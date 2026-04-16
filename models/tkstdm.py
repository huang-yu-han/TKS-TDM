"""
tkstdm.py — TKS-TDM: Temporal Key Selection-Transformer Diagnostic Model.

Full model integrating three modules described in the paper:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Input: x ∈ R^(B × C_in × L)   (B=batch, C_in=9, L=5120)         │
  │                                                                     │
  │  ① MSFRM (Multichannel Signal Feature Representation Module)       │
  │     Hierarchical Conv1d → MaxPool × 2 → Bottleneck × 2            │
  │     Output: F ∈ R^(B × emb × L/dr)   (emb=288, dr=8 → L/dr=640) │
  │                                                                     │
  │  ② KPSM  (Key Point Selection Module)                             │
  │     N=num_iters iterations of KPSMLayer                            │
  │     Initialises n=num_points uniform key positions P_1             │
  │     Each iter: sample → encode pos → fuse → Transformer → offset  │
  │     Output: T_N ∈ R^(B × n × emb)   (n=8 optimised key features) │
  │                                                                     │
  │  ③ CCM   (Condition Classification Module)                        │
  │     Prepend learnable class token T_cls → [T_cls ; T_N]           │
  │     Stack of (depth - num_iters) TransformerEncoderLayers          │
  │     Extract T_cls after final LayerNorm → Linear head → logits    │
  │     Output: y ∈ R^(B × num_classes)                               │
  └─────────────────────────────────────────────────────────────────────┘

References:
  Paper: TKS-TDM, Section III (Proposed Method).
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .kpsm import KPSMLayer
from .transformer_block import TransformerEncoderLayer


# ---------------------------------------------------------------------------
# MSFRM building blocks
# ---------------------------------------------------------------------------

def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1,
             groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


class BottleneckLayer(nn.Module):
    """1-D ResNet bottleneck block used inside the MSFRM stem.

    Structure (compress → transform → expand + residual):
        1×1 Conv → BN → ReLU
        3×3 Conv → BN → ReLU
        1×1 Conv → BN
        + shortcut (1×1 Conv → BN if channel mismatch)
        → ReLU

    This is the "compress-transform-expand" architecture described in
    the MSFRM section of the paper.
    """

    def __init__(self, in_channels: int, inter_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = _conv1x1(in_channels, inter_channels)
        self.bn1   = nn.BatchNorm1d(inter_channels)
        self.conv2 = _conv3x3(inter_channels, inter_channels)
        self.bn2   = nn.BatchNorm1d(inter_channels)
        self.conv3 = _conv1x1(inter_channels, out_channels)
        self.bn3   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                _conv1x1(in_channels, out_channels),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# Full TKS-TDM model
# ---------------------------------------------------------------------------

class TKSTDM(nn.Module):
    """Temporal Key Selection-Transformer Diagnostic Model (TKS-TDM).

    Args:
        signal_length  (int)  : Length of each input signal (L, default 5119).
        num_points     (int)  : Number of key sampling points n (default 8).
        in_channels    (int)  : Number of input sensor channels C_in (default 9).
        downsample_ratio(int) : MSFRM temporal downsampling factor dr (default 8).
        num_classes    (int)  : Number of fault categories (default 16).
        num_iters      (int)  : Number of KPSM iterations N (default 6).
        depth          (int)  : Total Transformer depth = num_iters + CCM_depth
                                (default 14, so CCM has 8 layers).
        embed_dim      (int)  : Embedding dimension emb (default 288).
        num_heads      (int)  : MSA attention heads h (default 6).
        mlp_ratio      (float): FFN hidden-dim ratio (default 3.0).
        offset_gamma   (float): Offset scaling factor γ (default 1.0).
        offset_bias    (bool) : Bias in offset FC layer (default True).
        drop_rate      (float): Dropout rate (default 0.0).
        attn_drop_rate (float): Attention dropout rate (default 0.0).
        drop_path_rate (float): Stochastic depth rate (default 0.0).
    """

    def __init__(self,
                 signal_length: int = 5119,
                 num_points: int = 8,
                 in_channels: int = 9,
                 downsample_ratio: int = 8,
                 num_classes: int = 16,
                 num_iters: int = 6,
                 depth: int = 14,
                 embed_dim: int = 288,
                 num_heads: int = 6,
                 mlp_ratio: float = 3.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 offset_gamma: float = 1.0,
                 offset_bias: bool = True):
        super().__init__()
        assert num_iters >= 1, "num_iters must be at least 1"
        assert depth > num_iters, "depth must be greater than num_iters"

        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.num_points  = num_points
        self.feat_size   = signal_length // downsample_ratio   # L / dr

        # ------------------------------------------------------------------
        # ① MSFRM — Multichannel Signal Feature Representation Module
        # ------------------------------------------------------------------
        # Input: (B, C_in, L)  →  Output: F ∈ (B, emb, L/dr)
        #
        # Hierarchy:
        #   Conv1d(C_in→64, k=7, s=2)  → BN → ReLU
        #   MaxPool1d(k=3, s=2)
        #   MaxPool1d(k=3, s=2)
        #   BottleneckLayer(64 → 64 → emb)
        #   BottleneckLayer(emb → 64 → emb)
        # ------------------------------------------------------------------
        self.msfrm = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3,
                      stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            BottleneckLayer(64,       64, embed_dim),
            BottleneckLayer(embed_dim, 64, embed_dim),
        )

        # ------------------------------------------------------------------
        # ② KPSM — Key Point Selection Module
        # ------------------------------------------------------------------
        # Shared learnable positional encoding layer W_1 (scalar → emb)
        self.pos_encoder = nn.Linear(1, embed_dim)

        # Initial uniform sampling positions P_1 (registered as buffer)
        self.register_buffer('init_points', self._uniform_points())

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.kpsm_layers = nn.ModuleList()
        for i in range(num_iters):
            self.kpsm_layers.append(
                KPSMLayer(feat_size=self.feat_size,
                          dim=embed_dim,
                          num_heads=num_heads,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[i],
                          norm_layer=norm_layer,
                          position_layer=self.pos_encoder,   # shared W_1
                          pred_offset=(i < num_iters - 1),   # last iter: no offset
                          gamma=offset_gamma,
                          offset_bias=offset_bias))

        # ------------------------------------------------------------------
        # ③ CCM — Condition Classification Module
        # ------------------------------------------------------------------
        # Learnable class token T_cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # Stack of standard Transformer encoder layers
        ccm_depth = depth - num_iters
        self.ccm_layers = nn.ModuleList([
            TransformerEncoderLayer(dim=embed_dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[i + num_iters],
                                    norm_layer=norm_layer)
            for i in range(ccm_depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialisation
        self.apply(self._init_weights)
        for layer in self.kpsm_layers:
            layer.reset_offset_weight()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _uniform_points(self) -> torch.Tensor:
        """Generate n uniformly-spaced initial sampling positions P_1."""
        step = self.feat_size / self.num_points
        coords = torch.tensor([i * step + step / 2
                                for i in range(self.num_points)])   # (n,)
        # Shape: (1, n, 1) — batch dim will be expanded at forward time
        return coords.view(1, -1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor):
        """Extract features from input signal.

        Args:
            x: (B, C_in, L)   normalised multi-sensor signal.

        Returns:
            cls_feat : (B, emb)   class-token feature vector f_cls.
            final_pts: (B, n, 1) final key-point positions P_N.
        """
        B = x.size(0)

        # ① MSFRM: (B, C_in, L) → F: (B, emb, L/dr)
        F = self.msfrm(x)

        # ② KPSM: iterative key-point selection
        points  = self.init_points.expand(B, -1, -1)   # (B, n, 1)
        kps_out = None
        offset  = None
        for layer in self.kpsm_layers:
            kps_out, offset, points = layer(F, points, offset, kps_out)
        # kps_out: (B, n, emb) — T_N

        # ③ CCM: prepend class token, stack Transformers
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, emb)
        seq = torch.cat([cls, kps_out], dim=1)              # (B, n+1, emb)
        for layer in self.ccm_layers:
            seq = layer(seq)
        seq = self.norm(seq)

        cls_feat = seq[:, 0]      # extract T_cls  → (B, emb)
        return cls_feat, points

    def forward(self, x: torch.Tensor):
        """Full forward pass.

        Args:
            x: (B, C_in, L)   normalised multi-sensor signal.

        Returns:
            logits : (B, num_classes)  classification logits.
            points : (B, n, 1)         final key-point positions.
        """
        feat, points = self.forward_features(x)
        logits = self.head(feat)
        return logits, points
