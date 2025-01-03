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
from modules.net_models import MeanPooling, AttentionPooling
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder

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
        elif self.pool_type == "mean":
            self.speech_pool = MeanPooling()
        # prediction head
        #self.prediction_head = PredictionHead(config=self.config, 
        #                                      input_dim=(self.config.hidden_size + self.text_config.hidden_size))
        self.prediction_head = PredictionHead(config=self.config, 
                                              input_dim=self.config.hidden_size)

        # other
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()
        
        # NOTE: freeze feature encoder
        self.freeze_feature_extractor()
        if "freeze_k_layers" in model_args:
            self.freeze_k_layers(model_args["freeze_k_layers"])
    
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
        hidden_states[~padding_mask] = 0.0
        
        if self.pool_type == "attn":
            hidden_states, _ = self.speech_pool(x=hidden_states, attn=hidden_states, mask=padding_mask)
        elif self.pool_type == "mean":
            hidden_states, _ = self.speech_pool(x=hidden_states, mask=padding_mask)
        
        logits = self.prediction_head(hidden_states)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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

