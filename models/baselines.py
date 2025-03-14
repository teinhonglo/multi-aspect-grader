import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import AutoModel, AutoConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import os
from transformers.models.wav2vec2 import Wav2Vec2PreTrainedModel
from transformers.models.bert import BertPreTrainedModel
from modules.net_models import MeanPooling, AttentionPooling, AttentionPooling2
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder
from losses import OridinalEntropy

class PredictionHead(nn.Module):
    def __init__(self, config, input_dim=None, output_dim=None):
        super(PredictionHead, self).__init__()
        
        if input_dim is None:
            input_dim = config.hidden_size
        if output_dim is None:
            output_dim = config.num_labels

        self.dense = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(config.final_dropout)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    embeds: Optional[torch.FloatTensor] = None

class AutoGraderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, model_args, class_weight=None, config=None, text_config=None, pretrained=False): 

        # config
        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_args["model_path"],
                num_labels=model_args["num_labels"],
                problem_type=model_args["problem_type"],
                final_dropout=model_args["final_dropout"]
            )
            self.text_config = AutoConfig.from_pretrained(
                model_args["text_model_path"]
            )
        else:
            self.config = config
            self.text_config = text_config

        super().__init__(self.config)
        
        # model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_args["model_path"], config=self.config)
        
        else:
            self.model = AutoModel.from_config(self.config)

        #self.speech_attn_pool = AttentionPooling(in_dim=self.config.hidden_size + self.text_config.hidden_size)
        self.pool_type = model_args["pool_type"]
        
        if self.pool_type == "attn":
            self.speech_pool = AttentionPooling(in_dim=self.config.hidden_size)
        elif self.pool_type == "attn2":
            self.speech_pool = AttentionPooling2(in_dim=self.config.hidden_size)
        elif self.pool_type == "mean":
            self.speech_pool = MeanPooling()
        
        if "pred_head" in model_args: 
            self.pred_head = model_args["pred_head"] 
        else:
            self.pred_head = "default"
        
        if self.pred_head == "default":
            # prediction head
            self.prediction_head = PredictionHead(config=self.config, 
                                                  input_dim=self.config.hidden_size)
        elif self.pred_head == "norm_head":
            self.prediction_head = nn.Sequential(nn.LayerNorm(self.config.hidden_size), nn.Dropout(self.config.final_dropout), nn.Linear(self.config.hidden_size, self.config.num_labels))

        # other
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()
        
        # NOTE: freeze feature encoder
        self.freeze_feature_extractor()
        if "freeze_k_layers" in model_args:
            self.freeze_k_layers(model_args["freeze_k_layers"])
        
        if self.config.problem_type == "oridnal_regression":
            self.loss_ode_fct = OridinalEntropy(lambda_d_phn=1.0, lambda_t_phn=1.0, margin=1.0, ignore_index=-1)
    
    def freeze_k_layers(self, k):
        if k > len(self.model.encoder.layers):
            k = None
        
        for parameter in self.model.encoder.layers[:k].parameters():
            parameter.requires_grad = False

    def freeze_feature_extractor(self):
        self.model.feature_extractor._freeze_parameters()

    def load_pretrained_wav2vec2(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor._freeze_parameters()

    def forward(self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        input_ids=None,
        text_attention_mask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None,
        delivery=None,
        delivery_mask=None,
        language_use=None,
        language_use_mask=None
    ):
        # wav2vec2
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        
        # create mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
         
        # response (audio)
        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        #hidden_states = torch.cat([hidden_states, text_hidden_states], dim=-1)
        expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_padding_mask] = 0.0
        
        if self.pool_type in ["attn", "attn2"]:
            hidden_states, _ = self.speech_pool(x=hidden_states, attn=hidden_states, mask=padding_mask)
        elif self.pool_type == "mean":
            hidden_states, _ = self.speech_pool(x=hidden_states, mask=padding_mask)
        
        logits = self.prediction_head(hidden_states)
        
        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weight)
                # labels 1-8 to 0-7
                # NOTE: teemi: 1-9 to 0-8
                labels = labels - 1
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "cefr_regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
                cefr_labels = torch.floor(labels)
                loss_cefr = loss_fct(logits, cefr_labels)
                loss = loss + 0.5 * loss_cefr
            elif self.config.problem_type == "oridnal_regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
                loss_ode = self.loss_ode_fct(hidden_states, labels, labels)
                loss = loss + 0.5 * loss_ode

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeds=hidden_states
        )

class AutoTextGraderModel(BertPreTrainedModel):
    def __init__(self, model_args, class_weight=None, config=None, text_config=None, pretrained=False): 

        # config
        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_args["model_path"],
                num_labels=model_args["num_labels"],
                problem_type=model_args["problem_type"],
                final_dropout=model_args["final_dropout"]
            )
            self.text_config = AutoConfig.from_pretrained(
                model_args["text_model_path"],
                num_labels=model_args["num_labels"],
                problem_type=model_args["problem_type"],
                final_dropout=model_args["final_dropout"]
            )
        else:
            self.config = config
            self.text_config = text_config

        super().__init__(self.config)
        
        # model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_args["text_model_path"], config=self.text_config)
        else:
            self.model = AutoModel.from_config(self.text_config)

        self.pool_type = model_args["pool_type"]
        
        if self.pool_type in ["attn", "attn2"]:
            self.speech_pool = AttentionPooling(in_dim=self.text_config.hidden_size)
        elif self.pool_type == "mean":
            self.speech_pool = MeanPooling()
          
        # prediction head
        if "pred_head" in model_args: 
            self.pred_head = model_args["pred_head"] 
        else:
            self.pred_head = "default"
        
        if self.pred_head == "default":
            self.prediction_head = PredictionHead(config=self.config, 
                                                  input_dim=self.config.hidden_size)
        elif self.pred_head == "norm_head":
            self.prediction_head = nn.Sequential(nn.LayerNorm(self.config.hidden_size), nn.Dropout(self.config.final_dropout), nn.Linear(self.config.hidden_size, self.config.num_labels))

        # other
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()
        
        if self.config.problem_type == "oridnal_regression":
            self.loss_ode_fct = OridinalEntropy(lambda_d_phn=1.0, lambda_t_phn=1.0, margin=1.0, ignore_index=-1)
        
    def freeze_k_layers(self, k):
        if k > len(self.model.encoder.layers):
            k = None
        
        for parameter in self.model.encoder.layers[:k].parameters():
            parameter.requires_grad = False

    def forward(self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        input_ids=None,
        text_attention_mask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None,
        delivery=None,
        delivery_mask=None,
        language_use=None,
        language_use_mask=None
    ):  
        outputs = self.model(
            input_ids,
            attention_mask=text_attention_mask
        )
        
        hidden_states = outputs["last_hidden_state"] # B, T, H 
        hidden_states = hidden_states[:,0,:] # B, H 
        
        logits = self.prediction_head(hidden_states)
        
        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weight)
                # labels 1-8 to 0-7
                # NOTE: teemi: 1-9 to 0-8
                labels = labels - 1
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "oridnal_regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
                loss_ode = self.loss_ode_fct(hidden_states, labels, labels)
                loss = loss + 0.5 * loss_ode

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeds=hidden_states
        )

class AutoAudioTextGraderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, model_args, class_weight=None, config=None, text_config=None, pretrained=False): 

        # config
        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_args["model_path"],
                num_labels=model_args["num_labels"],
                problem_type=model_args["problem_type"],
                final_dropout=model_args["final_dropout"]
            )
            self.text_config = AutoConfig.from_pretrained(
                model_args["text_model_path"]
            )
        else:
            self.config = config
            self.text_config = text_config

        super().__init__(self.config)
        
        # model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_args["model_path"], config=self.config)
            self.text_model = AutoModel.from_pretrained(model_args["text_model_path"], config=self.text_config)
        else:
            self.model = AutoModel.from_config(self.config)
            self.text_model = AutoModel.from_config(self.text_config)

        self.pool_type = model_args["pool_type"]
        
        if self.pool_type == "attn":
            self.speech_pool = AttentionPooling(in_dim=self.config.hidden_size)
        elif self.pool_type == "attn2":
            self.speech_pool = AttentionPooling2(in_dim=self.config.hidden_size)
        elif self.pool_type == "mean":
            self.speech_pool = MeanPooling()

        self.proj_layer = nn.Linear(self.config.hidden_size + self.text_config.hidden_size, self.config.hidden_size)
        
        if "pred_head" in model_args: 
            self.pred_head = model_args["pred_head"] 
        else:
            self.pred_head = "default"
        
        if self.pred_head == "default":
            # prediction head
            self.prediction_head = PredictionHead(config=self.config, 
                                                  input_dim=self.config.hidden_size)
        elif self.pred_head == "norm_head":
            self.prediction_head = nn.Sequential(nn.LayerNorm(self.config.hidden_size), nn.Dropout(self.config.final_dropout), nn.Linear(self.config.hidden_size, self.config.num_labels))

        # other
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()
        
        # NOTE: freeze feature encoder
        self.freeze_feature_extractor()
        if "freeze_k_layers" in model_args:
            self.freeze_k_layers(model_args["freeze_k_layers"])
        
        if self.config.problem_type == "oridnal_regression":
            self.loss_ode_fct = OridinalEntropy(lambda_d_phn=1.0, lambda_t_phn=1.0, margin=1.0, ignore_index=-1)
    
    def freeze_k_layers(self, k):
        if k > len(self.model.encoder.layers):
            k = None
        
        for parameter in self.model.encoder.layers[:k].parameters():
            parameter.requires_grad = False

    def freeze_feature_extractor(self):
        self.model.feature_extractor._freeze_parameters()

    def load_pretrained_wav2vec2(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor._freeze_parameters()

    def forward(self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        input_ids=None,
        text_attention_mask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None,
        delivery=None,
        delivery_mask=None,
        language_use=None,
        language_use_mask=None
    ):
        # create mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        text_attention_mask = (
            text_attention_mask if text_attention_mask is not None else torch.ones_like(input_ids, dtype=torch.long)
        )
        # wav2vec2
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
         
        # response (audio)
        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        #hidden_states = torch.cat([hidden_states, text_hidden_states], dim=-1)
        expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_padding_mask] = 0.0
        
        if self.pool_type in ["attn", "attn2"]:
            hidden_states, _ = self.speech_pool(x=hidden_states, attn=hidden_states, mask=padding_mask)
        elif self.pool_type == "mean":
            hidden_states, _ = self.speech_pool(x=hidden_states, mask=padding_mask)
        # text
        text_outputs = self.text_model(
            input_ids,
            attention_mask=text_attention_mask
        )
        
        text_hidden_states = text_outputs["last_hidden_state"] # B, T, H 
        text_hidden_states = text_hidden_states[:,0,:]

        hidden_states = self.proj_layer(torch.cat([hidden_states,text_hidden_states], dim=-1))

        logits = self.prediction_head(hidden_states)
        
        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weight)
                # labels 1-8 to 0-7
                # NOTE: teemi: 1-9 to 0-8
                labels = labels - 1
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "oridnal_regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
                loss_ode = self.loss_ode_fct(hidden_states, labels, labels)
                loss = loss + 0.5 * loss_ode

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeds=hidden_states
        )
