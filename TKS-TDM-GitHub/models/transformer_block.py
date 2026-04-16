"""
transformer_block.py — Transformer building blocks for TKS-TDM.

Includes:
  - Mlp            : Feed-Forward Network (FFN) used inside each Transformer encoder layer.
  - Attention      : Multi-head Self-Attention (MSA).
  - TransformerEncoderLayer : Pre-norm Transformer block (MSA + FFN + residual + LayerNorm),
                              used in both KPSM iterations and the CCM stack.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Mlp(nn.Module):
    """Feed-Forward Network (FFN) with GELU activation.

    Structure: Linear → GELU → Dropout → Linear → Dropout
    Used inside every TransformerEncoderLayer in KPSM and CCM.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 act_layer=nn.GELU,
                 drop: float = 0.):
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


class Attention(nn.Module):
    """Multi-head Self-Attention (MSA).

    Standard scaled dot-product attention with h=num_heads heads.
    Used inside TransformerEncoderLayer.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder block.

    Applies:
        x = x + DropPath(MSA(LayerNorm(x)))
        x = x + DropPath(FFN(LayerNorm(x)))

    Used in:
      - KPSM: one layer per key-point selection iteration.
      - CCM : a stack of (depth - num_iters) layers for condition classification.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
