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

class AutoMAGraderModel(Wav2Vec2PreTrainedModel):
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

        # content
        self.content_encoder = TransformerEncoder(dim=self.config.hidden_size, num_heads=1, mlp_hidden_dim=3*self.config.hidden_size, use_rope=True, max_seq_len=5000, act_layer="nn.SiLU", mlp_mdl="Mlp2")
        self.content_pooling = AttentionPooling(in_dim=self.config.hidden_size)

        # delivery
        self.delivery_encoder = TransformerEncoder(dim=15, num_heads=1, mlp_hidden_dim=3*15, use_rope=False, max_seq_len=500, act_layer="nn.SiLU", mlp_mdl="Mlp2")
        self.delivery_pooling = AttentionPooling(in_dim=15)

        # language use
        self.language_use_encoder = TransformerEncoder(dim=263, num_heads=1, mlp_hidden_dim=3*263, use_rope=False, max_seq_len=500, act_layer="nn.SiLU", mlp_mdl="Mlp2")
        self.language_use_pooling = AttentionPooling(in_dim=263)

        self.proj_layer = nn.Linear(self.config.hidden_size + 15 + 263, self.config.hidden_size)
        # prediction head
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
        
        # prompt
        prompt_outputs = self.text_model(
            prompt_input_ids,
            attention_mask=prompt_attention_mask
        )
        
        # create mask
        # response
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        padding_mask = padding_mask.bool()
        # text
        text_attention_mask = (
            text_attention_mask if text_attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        )
        text_attention_mask = text_attention_mask.bool()
        # prompt
        prompt_attention_mask = (
            prompt_attention_mask if prompt_attention_mask is not None else torch.ones_like(prompt_input_ids, dtype=torch.bool)
        )
        prompt_attention_mask = prompt_attention_mask.bool()
        # delivery
        delivery_mask = (
            delivery_mask if delivery_mask is not None else ~(torch.ones_like(delivery, dtype=torch.long) == 0).all(dim=2).bool()
        )
        delivery_mask = delivery_mask.bool()
        # language use
        language_use_mask = (
            language_use_mask if language_use_mask is not None else ~(torch.ones_like(language_use, dtype=torch.long) == 0).all(dim=2).bool()
        )
        language_use_mask = language_use_mask.bool()

        # content (audio)
        content_vector = self.content_encoder(x=hidden_states, mask=~padding_mask)
        content_vector, _ = self.content_pooling(x=content_vector, attn=content_vector, mask=padding_mask)

        # delivery
        delivery_vector = self.delivery_encoder(x=delivery, mask=~delivery_mask)
        delivery_vector, _ = self.delivery_pooling(x=delivery_vector, attn=delivery_vector, mask=delivery_mask)

        # language use
        language_use_vector = self.language_use_encoder(x=language_use, mask=~language_use_mask)
        language_use_vector, _ = self.language_use_pooling(x=language_use_vector, attn=language_use_vector, mask=language_use_mask)
        
        # content
        # prompt (text)
        prompt_hidden_states = prompt_outputs["last_hidden_state"] # B, T, H       
        prompt_vector = prompt_hidden_states[:,0,:] # B, H 
        
        fusion_hidden_states = torch.cat([content_vector, delivery_vector, language_use_vector], dim=-1)
        fusion_hidden_states = prompt_vector + self.proj_layer(fusion_hidden_states)
        logits = self.prediction_head(fusion_hidden_states)

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

