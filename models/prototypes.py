import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import AutoModel, AutoConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

import os
from transformers.models.wav2vec2 import Wav2Vec2PreTrainedModel
from transformers.models.bert import BertPreTrainedModel
from modules.net_models import MeanPooling, AttentionPooling
from modules.encoders import TransformerEncoder
from modules.decoders import TransformerDecoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from losses import OrdinalRegressionLoss, CumulativeLinkLoss

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

class AutoGraderPrototypeModel(nn.Module):
    def __init__(self, model_args, class_weight=None, config=None, text_config=None, pretrained=False):
        super(AutoGraderPrototypeModel, self).__init__()

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

        # model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_args["model_path"], config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # NOTE: prototype-related
        self.num_prototypes = model_args["num_prototypes"]
        self.prototype = nn.Embedding(model_args["num_labels"] * self.num_prototypes, self.model.config.hidden_size)
        self.dist = model_args["dist"]
        if self.dist == "scos":
            self.w = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(-5.0))
        self.final_dropout = nn.Dropout(self.config.final_dropout)

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

    def get_prototype(self):
        return self.prototype.weight.reshape(self.num_labels, self.num_prototypes, -1).detach().cpu()

    def init_prototypes(self, tr_dataset, path, device="cuda", eps=1e-6) -> None:
        # init with wav2vec2
        print("[INFO] Initialize prototypes with wav2vec2 ...")

        embed_path = path + "/prototype_initials_var0.05.pt"
        if not os.path.exists(embed_path) or True:

            prototype_initials = torch.full((self.num_labels, self.model.config.hidden_size), fill_value=eps)

            def compute_embeddings(batch):
                input_values = torch.as_tensor(batch["input_values"], device=device).unsqueeze(0)
                labels = torch.as_tensor(batch["labels"], dtype=torch.long)

                with torch.no_grad():
                    outputs = self.model(input_values)
                    hidden_states = outputs[0]
                    # [1, 768] -> [768]
                    embeddings = torch.mean(hidden_states, dim=1).squeeze(0)

                lv = labels-1
                prototype_initials[lv] += embeddings.detach().cpu()

            # compute all embeddings
            self.model.eval()
            self.model.to(device)
            tr_dataset = tr_dataset.map(compute_embeddings)
            torch.cuda.empty_cache()
            self.model.train()

            # avg same level embeddings
            tr_labels = torch.as_tensor(tr_dataset['labels'])-1
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
        #nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal
    
    def negative_sed(self, a, b):
        ''' negative square euclidean distance
        - input
            a: batch x D
            b: (num_label * num_proto) x D
        - output
            logits: batch x num_label
        '''
        
        n = a.shape[0]
        m = b.shape[0]
        #if a.size(1) != b.size(1):
        #    raise Exception
     
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        
        # calculate centroid of prototypes
        logits = logits.reshape(-1, self.num_prototypes, self.num_labels)
        logits = logits.mean(dim=1)

        return logits
    
    def negative_sed2(self, a, b):
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
        hidden_states = self.final_dropout(hidden_states)
        hidden_states = torch.mean(hidden_states, dim=1)
        # calculate distance
        if self.dist == "sed":
            logits = self.negative_sed(hidden_states, self.prototype.weight)
        elif self.dist == "sed2":
            logits = self.negative_sed2(hidden_states, self.prototype.weight)
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

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeds=hidden_states
        )


class AutoGraderPrototypeRegModel(Wav2Vec2PreTrainedModel):
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
        
        # NOTE: prototype-related
        self.num_prototypes = model_args["num_prototypes"]
        self.num_cefr_levels = model_args["num_cefr_levels"]
        print(f"Num prototypes {self.num_prototypes} and Num cefr {self.num_cefr_levels}")
        self.prototype = nn.Embedding(self.num_cefr_levels * self.num_prototypes, self.model.config.hidden_size)
        self.dist = model_args["dist"]
        self.use_softmax = True if "use_softmax" not in model_args else model_args["use_softmax"]
        print(f"use_softmax {self.use_softmax}")
        
        if self.dist == "scos":
            self.w = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(-5.0))
        self.softmax = nn.Softmax(dim=1)
        
        input_dim = self.config.hidden_size + self.num_cefr_levels
        if self.pred_head == "default":
            # prediction head
            self.prediction_head = PredictionHead(config=self.config, 
                                                  input_dim=input_dim)
        elif self.pred_head == "norm_head":
            self.prediction_head = nn.Sequential(nn.LayerNorm(input_dim), nn.Dropout(self.config.final_dropout), nn.Linear(input_dim, self.config.num_labels))

        # other
        if self.config.num_labels != 1:
            raise Exception(f"self.num_labels is only allowed to 1, your self.num_labels is {self.num_labels}")
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()
        
        # NOTE: freeze feature encoder
        self.freeze_feature_extractor()
        if "freeze_k_layers" in model_args:
            self.freeze_k_layers(model_args["freeze_k_layers"])
        
        if self.config.problem_type == "ordinal_regression":
            self.loss_odr_fct = OrdinalRegressionLoss(num_class=8, train_cutpoints=False, scale=20.0)
        elif self.config.problem_type == "cumulative_link_loss":
            self.loss_cll_fct = CumulativeLinkLoss()
    
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
    
    def get_prototype(self):
        return self.prototype.weight.reshape(self.num_labels, self.num_prototypes, -1).detach().cpu()

    def init_prototypes(self, tr_dataset, path, device="cuda", eps=1e-10) -> None:
        # init with wav2vec2
        print("[INFO] Initialize prototypes with wav2vec2 ...")

        embed_path = path + "/prototype_initials_var0.05.pt"
        if not os.path.exists(embed_path) or True:

            prototype_initials = torch.full((self.num_cefr_levels, self.model.config.hidden_size), fill_value=eps)

            def compute_embeddings(batch):
                input_values = torch.as_tensor(batch["input_values"], device=device).unsqueeze(0)
                if self.num_cefr_levels == 8:
                    # NOTE: for slate, 0-7, +1, 1-8 (minus 1 after this operation) 
                    labels = torch.as_tensor(batch["labels"] * 2 - 4, dtype=torch.long) + 1
                elif self.num_cefr_levels == 4:
                    labels = torch.as_tensor(batch["labels"], dtype=torch.long) -2 + 1

                with torch.no_grad():
                    outputs = self.model(input_values)
                    hidden_states = outputs[0]
                    # [1, T, 768] -> [1, 768] -> [768]
                    embeddings = torch.mean(hidden_states, dim=1).squeeze(0)

                lv = labels - 1
                prototype_initials[lv] += embeddings.detach().cpu()

            # compute all embeddings
            self.model.eval()
            self.model.to(device)
            tr_dataset = tr_dataset.map(compute_embeddings)
            torch.cuda.empty_cache()
            self.model.train()

            # avg same level embeddings
            # NOTE: for slate, 0-7
            tr_labels = torch.as_tensor(tr_dataset['labels'], dtype=torch.float)
            for lv in range(self.num_cefr_levels):
                if self.num_cefr_levels == 8:
                    lv_val = (lv + 4) / 2
                    lv_num = torch.count_nonzero((tr_labels == lv_val)) + eps
                elif self.num_cefr_levels == 4:
                    lv_val = lv + 2
                    lv_num = ((tr_labels >= lv_val) & (tr_labels <= (lv_val + 0.5))).sum().item() + eps
                print(lv, lv_val, lv_num)
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
        #nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal
        
    def init_scaler(self, tr_dataset):
        # for regression
        print("[INFO] Initialize scalar (regression) ...")
        tr_labels = np.array(tr_dataset['labels'])
        scaler = StandardScaler()
        self.scaler = scaler.fit(tr_labels.reshape(-1, 1))
    
    def init_mean_var(self, tr_vector_mean, tr_vector_var):
        # for regression
        print("[INFO] Initialize mean and var ...")
        self.tr_vector_mean = tr_vector_mean
        self.tr_vector_var = tr_vector_var
    
    def negative_sed(self, a, b):
        ''' negative square euclidean distance
        - input
            a: batch x D
            b: (num_label * num_proto) x D
        - output
            logits: batch x num_label
        '''
        
        n = a.shape[0]
        m = b.shape[0]
        #if a.size(1) != b.size(1):
        #    raise Exception
     
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        
        # calculate centroid of prototypes
        logits = logits.reshape(-1, self.num_prototypes, self.num_labels)
        logits = logits.mean(dim=1)

        return logits
    
    def negative_sed2(self, a, b):
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

    def pairwise_ranking_loss(self, preds, targets, margin=0.0):
        """
        Simple pairwise ranking loss (RankNet style) to encourage correct ordering.
        preds: Tensor of shape (B,)
        targets: Tensor of shape (B,)
        """
        B = preds.size(0)
        loss = 0.0
        count = 0
        for i in range(B):
            for j in range(B):
                if targets[i] > targets[j] + margin:
                    pred_diff = preds[i] - preds[j]
                    loss += F.logsigmoid(pred_diff)  # log(sigmoid(pred_i - pred_j))
                    count += 1
        if count > 0:
            loss = -loss / count  # take mean and flip to loss
        else:
            loss = torch.tensor(0.0, device=preds.device)
        return loss

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
        expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_padding_mask] = 0.0
        
        if self.pool_type in ["attn", "attn2"]:
            hidden_states, _ = self.speech_pool(x=hidden_states, attn=hidden_states, mask=padding_mask)
        elif self.pool_type == "mean":
            hidden_states, _ = self.speech_pool(x=hidden_states, mask=padding_mask)
       
        B, H = hidden_states.shape 
        # calculate distance
        if self.dist == "sed":
            proto_logits = self.negative_sed(hidden_states, self.prototype.weight)
        elif self.dist == "sed2":
            proto_logits = self.negative_sed2(hidden_states, self.prototype.weight)
        elif self.dist == "cos":
            proto_logits = self.cosine_sim(hidden_states, self.prototype.weight)
        elif self.dist == "scos":
            proto_logits = self.cosine_sim(hidden_states, self.prototype.weight, scale=True)
        else:
            raise ValueError("dist choices [sed, cos], {} is provided.".format(self.dist))
     
        proto_logits = proto_logits.view(B, self.num_cefr_levels) 
        if self.use_softmax: 
            proto_logits_softmax = self.softmax(proto_logits)
            hidden_states = torch.cat([hidden_states, proto_logits_softmax], dim=-1)
        else:
            hidden_states = torch.cat([hidden_states, proto_logits], dim=-1)
        
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
            elif self.config.problem_type == "ordinal_regression":
                logits = logits.squeeze(-1)
                labels = ((labels - 2.0) * 2).long()
                loss_ode = self.loss_odr_fct(logits, labels)
                loss = loss_ode
            elif self.config.problem_type == "prototype_regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)
                loss = loss_fct(logits, labels)
                loss_ce_fct = CrossEntropyLoss(weight=self.class_weight)
                
                if self.num_cefr_levels == 8:
                    # 0-7
                    labels_cls = (labels * 2 - 4).long()
                elif self.num_cefr_levels == 4:
                    # 0-3
                    labels_cls = (labels).long() - 2

                loss_pt = loss_ce_fct(proto_logits.view(-1, self.num_cefr_levels), labels_cls.view(-1))
                loss = loss + 0.5 * loss_pt
            elif self.config.problem_type == "pairwise_ranking_regression":
                # MSE + pairwise ranking loss
                logits = logits.squeeze(-1)
                mse_loss = F.mse_loss(logits, labels)
                ranking_loss = self.pairwise_ranking_loss(logits, labels)
                loss = mse_loss + 0.3 * ranking_loss  # lambda = 0.3
            elif self.config.problem_type == "scaled_regression":
                loss_fct = MSELoss()
                logits = logits.squeeze(-1)         # (B,)
                labels = labels.view(-1)            # (B,)

                # 1. 標準化 labels（無需保留計算圖）
                with torch.no_grad():
                    labels_np = labels.detach().cpu().numpy().reshape(-1, 1)
                    labels_scaled_np = self.scaler.transform(labels_np).flatten()
                    labels_scaled = torch.tensor(labels_scaled_np, dtype=torch.float32, device=labels.device)

                # 2. logits 使用 scale 的 mean/std 做 inline 標準化，不打斷計算圖
                mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=logits.device)
                std = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=logits.device)
                logits_scaled = (logits - mean) / std

                # 3. 計算 loss，保留計算圖
                loss = loss_fct(logits_scaled, labels_scaled)
            elif self.config.problem_type == "weighted_focal_regression":
                activate='sigmoid'
                beta=.2 
                gamma=1
                logits = logits.squeeze(-1)
                loss = (logits - labels) ** 2
                loss *= (torch.tanh(beta * torch.abs(logits - labels))) ** gamma if activate == 'tanh' else \
                    (2 * torch.sigmoid(beta * torch.abs(logits - labels)) - 1) ** gamma
                loss = torch.mean(loss)
            elif self.config.problem_type == "test_time_adaptation":
                logits = logits.squeeze(-1)
                hidden_states_mean = torch.mean(hidden_states, dim=0)
                hidden_states_var = torch.var(hidden_states, dim=0)
                
                self.tr_vector_mean = self.tr_vector_mean.to(logits.device)
                self.tr_vector_var = self.tr_vector_var.to(logits.device)
                
                mean_loss = torch.mean(torch.pow(self.tr_vector_mean - hidden_states_mean, 2))
                var_loss = torch.mean(torch.pow(self.tr_vector_var - hidden_states_var, 2))
                loss = mean_loss + var_loss
            elif self.config.problem_type == "cumulative_link_loss":
                # labels 1-8 to 0-7
                labels = labels - 1
                loss = self.loss_cll_fct(logits.view(-1, self.num_labels), labels.view(-1))

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

