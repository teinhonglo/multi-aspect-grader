import sys
import math
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import _VF, Tensor
from torch.nn.init import xavier_uniform_
from modules.position_embeddings import RotaryPositionalEmbeddings
from modules.scaling import BiasNorm, RMSNorm

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)  

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_rope=False, max_seq_len=50):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5
        self.use_rope = use_rope
        self.attn_weights = None

        # RoPE
        if self.use_rope:
            self.rotary_pos_embed = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=max_seq_len)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _reset_parameters(self):
        xavier_uniform_(self.qkv.weight, gain=1 / math.sqrt(2))

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute([0, 2, 1, 3])

    def forward(self, x, mask=None, position_ids=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # # make torchscript happy (cannot use tensor as tuple)

        # RoPE
        if self.use_rope:
            q = self.rotary_pos_embed(q, input_pos=position_ids)
            k = self.rotary_pos_embed(k, input_pos=position_ids)

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(
                mask.unsqueeze(1).unsqueeze(2),
                -1e4,
            )

        self.attn_weights = attn.clone()
        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Swish MLP
class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.act_fn = act_layer()
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        swish = self.act_fn(self.gate_proj(x))
        x = swish * self.up_proj(x)
        x = self.down_proj(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim=None, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., use_rope=False, max_seq_len=50,
                 drop_path=0., act_layer="nn.GELU", norm_layer="nn.LayerNorm", mlp_mdl="Mlp"):
        super().__init__()
        # NOTE: attention
        act_layer= eval(act_layer)
        norm_layer = eval(norm_layer)
        mlp_mdl = eval(mlp_mdl)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_rope=use_rope, proj_drop=drop, max_seq_len=max_seq_len)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        
        if mlp_hidden_dim is None:
            mlp_ratio = 4
            mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = mlp_mdl(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        # attention
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

