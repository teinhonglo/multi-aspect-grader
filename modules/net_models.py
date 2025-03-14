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

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask):
        '''
            x: (B, T, H1)
            attn: (B, T, H2)
            x_mask: (B)
        '''
        print(mask)
        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings, input_mask_expanded

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.GELU(),
        )

    def forward(self, x, attn, mask=None):
        '''
            x: (B, T, H1)
            attn: (B, T, H2)
            x_mask: (B)
        '''
        w = self.attention(attn).float()

        if mask is not None:
            w[mask==0] = float('-inf')
            
        w = torch.softmax(w, dim=1)
        x = torch.sum(w * x, dim=1)
        return x, w

class AttentionPooling2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 1)
        )

    def forward(self, x, attn, mask=None):
        '''
            x: (B, T, H1)
            attn: (B, T, H2)
            x_mask: (B)
        '''
        w = self.attention(attn).float()

        if mask is not None:
            w[mask==0] = float('-inf')
            
        w = torch.softmax(w, dim=1)
        x = torch.sum(w * x, dim=1)
        return x, w
