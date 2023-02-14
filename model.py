import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
#from transformers import Wav2Vec2Model, PretrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import os
from datasets import load_from_disk
        
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class PredictionHead(nn.Module):
    def __init__(self, config):
        super(PredictionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2vec2GraderModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config, class_weight=None):
        super(Wav2vec2GraderModel, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.class_weight = class_weight

        self.wav2vec2 = Wav2Vec2Model(config)
        self.prediction_head = PredictionHead(config)

        self.init_weights()

    def freeze(self, module):
        for parameter in module.parameters():
            parameter.requires_grad = False

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def load_pretrained_wav2vec2(self, state_dict):
        self.wav2vec2.load_state_dict(state_dict)
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        #hidden_states = self.pooler(hidden_states, input_values['attention_mask'])
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

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Wav2vec2GraderPrototypeModel(Wav2vec2GraderModel):

    def __init__(self, config, class_weight=None, num_prototypes=3, dist="sed"):
        super(Wav2vec2GraderPrototypeModel, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.class_weight = class_weight

        self.wav2vec2 = Wav2Vec2Model(config)

        # NOTE: prototype-related
        self.num_prototypes = num_prototypes
        self.prototype = nn.Embedding(self.num_labels * self.num_prototypes, self.wav2vec2.config.hidden_size)
        self.dist = dist
        if self.dist == "scos":
            self.w = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(-5.0))

        self.init_weights()
    
    def init_prototypes(self, tr_dataset, path, nj=8, device="cuda", eps=1e-6) -> None:
        # init with wav2vec2
        print("[INFO] Initialize prototypes with wav2vec2 ...")

        embed_path = path + "/prototype_initials.pt"
        if not os.path.exists(embed_path):

            prototype_initials = torch.full((self.num_labels, self.wav2vec2.config.hidden_size), fill_value=eps)
            
            def compute_embeddings(batch):
                input_values = torch.as_tensor(batch["input_values"], device=device).unsqueeze(0)
                labels = torch.as_tensor(batch["labels"], dtype=torch.long)

                with torch.no_grad():
                    outputs = self.wav2vec2(input_values)
                    hidden_states = outputs[0]
                    # [1, 768] -> [768]
                    embeddings = torch.mean(hidden_states, dim=1).squeeze(0)

                lv = batch['labels']-1
                prototype_initials[lv] += embeddings.detach().cpu()
            
            # compute all embeddings
            self.wav2vec2.eval()
            self.wav2vec2.to(device)
            tr_dataset = tr_dataset.map(compute_embeddings)
            torch.cuda.empty_cache()
            self.wav2vec2.train()

            # take avg of same level embeddings
            tr_labels = torch.as_tensor(tr_dataset['labels'])
            for lv in range(self.num_labels):
                lv_num = torch.count_nonzero((tr_labels == lv)) + eps
                prototype_initials[lv] = prototype_initials[lv] / lv_num
                
            # add noise
            var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
            prototype_initials = prototype_initials.repeat(self.num_prototypes, 1) # repeat num_prototypes in dim 1
            noise = (var ** 0.5) * torch.randn(prototype_initials.size())
            prototype_initials = prototype_initials + noise  # Add Gaussian noize

            # save embeddings
            torch.save(prototype_initials, embed_path)

        else:
            print("[INFO] {} exists, using it...".format(embed_path))
            prototype_initials = torch.load(embed_path)

        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

    
    def negative_sed(self, a, b):
        ''' negative square euclidean distance
        - input
            a: batch x D
            b: (num_label * num_proto) x D
        - output
            logits: batch x num_label
        '''

        # calculate centroid of prototypes
        b = b.reshape(self.num_labels, self.num_prototypes, -1)
        b = b.mean(dim=1)

        n = a.shape[0]
        m = b.shape[0]
        if a.size(1) != b.size(1):
            raise Exception
     
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)

        return logits
    
    def cosine_sim(self, a, b, scale=False):
        ''' cosine similarity
        - input
            a: batch x D
            b: (num_label * num_proto) x D
        - output
            logits: batch x num_label
        '''

        a = nn.functional.normalize(a)
        b = nn.functional.normalize(b)
        logits = torch.mm(a, b.T)
        if scale:
            torch.clamp(self.w, 1e-6)
            logits = self.w * logits + self.b
        logits = logits.reshape(-1, self.num_prototypes, self.num_labels)
        logits = logits.mean(dim=1)

        return logits

    def forward(self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        # calculate distance
        if self.dist == "sed":
            logits = self.negative_sed(hidden_states, self.prototype.weight)
        elif self.dist == "cos":
            logits = self.cosine_sim(hidden_states, self.prototype.weight)
        elif self.dist == "scos":
            logits = self.cosine_sim(hidden_states, self.prototype.weight, scale=True)
        else:
            raise ValueError("dist choices [sed, cos], {} is provided.".format(self.dist))

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

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )