"""
kpsm.py — Key Point Selection Module (KPSM) for TKS-TDM.

The KPSM iteratively refines N=num_iters learnable temporal sampling positions
(key points) over the compressed feature map produced by the MSFRM.

At each iteration t:
  1. Sample features at current positions via differentiable 1-D linear
     interpolation (DifferentiableSampler).
  2. Encode positions with a learnable linear layer (positional encoding).
  3. Fuse sampled features + positional encoding + previous iteration output.
  4. Refine with a single TransformerEncoderLayer.
  5. Predict position offsets with a linear layer; update sampling positions.

The KPSM is fully differentiable — gradients propagate from the
classification loss all the way back through the interpolation kernel to
update sampling positions end-to-end.

References:
  Paper: TKS-TDM, Section III-B (Key Point Selection Module).
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .transformer_block import TransformerEncoderLayer


# ---------------------------------------------------------------------------
# Differentiable 1-D Linear Interpolation Sampler
# ---------------------------------------------------------------------------

class _SamplingFunction(Function):
    """Custom autograd function for differentiable 1-D linear interpolation.

    Forward pass  : bilinear (linear) interpolation at fractional positions.
    Backward pass : scatter gradients to input; compute offset gradient via
                    γ · Σ( grad_out · (x[idx1] − x[idx0]) ).
    """

    @staticmethod
    def forward(ctx, input, point, offset, gamma: float):
        """
        Args:
            input  : (B, C, L)   compressed feature map from MSFRM.
            point  : (B, n, 1)   current sampling positions (pixel units).
            offset : (B, n, 1)   position offsets predicted by previous layer.
            gamma  : float       offset scaling factor (offset_gamma in config).

        Returns:
            output : (B, n, C)   sampled feature vectors at key points.
        """
        ctx.gamma = float(gamma)
        if offset is None:
            offset = torch.zeros_like(point)
        ctx.save_for_backward(input, point, offset)

        B, C, L = input.size()
        n = point.size(1)

        locs = torch.clamp(point + gamma * offset, 0, L - 1)
        idx0 = torch.clamp(torch.floor(locs).long(), 0, L - 1)
        idx1 = torch.clamp(torch.ceil(locs).long(),  0, L - 1)
        w1 = locs - idx0.float()
        w0 = 1.0 - w1

        output = torch.zeros(B, n, C, device=input.device)
        for b in range(B):
            i0 = idx0[b, :, 0]   # (n,)
            i1 = idx1[b, :, 0]
            a0 = w0[b, :, 0].unsqueeze(1)   # (n, 1)
            a1 = w1[b, :, 0].unsqueeze(1)
            # input[b]: (C, L) → index along L → (C, n) → transpose → (n, C)
            output[b] = (a0 * input[b, :, i0].T + a1 * input[b, :, i1].T)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, point, offset = ctx.saved_tensors
        gamma = ctx.gamma

        grad_input  = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)

        B, C, L = input.size()
        n = point.size(1)

        locs = torch.clamp(point + gamma * offset, 0, L - 1)
        idx0 = torch.clamp(torch.floor(locs).long(), 0, L - 1)
        idx1 = torch.clamp(torch.ceil(locs).long(),  0, L - 1)
        w1 = locs - idx0.float()
        w0 = 1.0 - w1

        for b in range(B):
            for p in range(n):
                i0 = idx0[b, p, 0]
                i1 = idx1[b, p, 0]
                a0 = w0[b, p, 0]
                a1 = w1[b, p, 0]
                grad_input[b, :, i0].add_(a0 * grad_output[b, p])
                grad_input[b, :, i1].add_(a1 * grad_output[b, p])
                if i0 != i1:
                    diff = input[b, :, i1] - input[b, :, i0]
                    grad_offset[b, p, 0] += gamma * torch.sum(
                        grad_output[b, p] * diff)
        return grad_input, None, grad_offset, None


_sampling_fn = _SamplingFunction.apply


@torch.jit.script
def _jit_sample(input, point, offset, gamma: float):
    """TorchScript JIT version of the sampler (used during inference)."""
    B, C, L = input.shape
    n = point.shape[1]
    locs = torch.clamp(point + gamma * offset, 0, L - 1)
    idx0 = torch.clamp(torch.floor(locs).long(), 0, L - 1)
    idx1 = torch.clamp(torch.ceil(locs).long(),  0, L - 1)
    w1 = locs - idx0.float()
    w0 = 1.0 - w1
    output = torch.zeros(B, n, C, device=input.device)
    for b in range(B):
        for p in range(n):
            i0 = idx0[b, p, 0]
            i1 = idx1[b, p, 0]
            output[b, p] = (w0[b, p, 0] * input[b, :, i0] +
                            w1[b, p, 0] * input[b, :, i1])
    return output


class DifferentiableSampler(nn.Module):
    """Differentiable 1-D temporal sampler.

    Uses the custom autograd version during training (for gradient flow to
    offset predictions) and the JIT-compiled version during inference (faster).

    Args:
        gamma (float): Offset scaling factor (γ in the paper).
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, point, offset=None):
        if offset is None:
            offset = torch.zeros_like(point)
        if self.training:
            return _sampling_fn(input, point, offset, self.gamma)
        else:
            return _jit_sample(input, point, offset, self.gamma)


# ---------------------------------------------------------------------------
# Single KPSM Iteration Layer
# ---------------------------------------------------------------------------

class KPSMLayer(nn.Module):
    """One iteration of the Key Point Selection Module (KPSM).

    At iteration t, this layer:
      1. Samples features T_t' from the MSFRM feature map F at positions P_t.
      2. Encodes positions P_t → P_t' via a learnable linear layer W_1.
      3. Fuses T_t', P_t', and the previous iteration's output T_{t-1}.
      4. Processes the fused tensor through a TransformerEncoderLayer → T_t.
      5. Predicts offsets O_t from T_t; updates P_{t+1} = P_t + O_t.

    Args:
        feat_size    (int)  : Length of the compressed feature sequence (L/dr).
        dim          (int)  : Embedding dimension (emb).
        num_heads    (int)  : Number of attention heads in MSA.
        mlp_ratio    (float): FFN hidden-dim ratio.
        position_layer       : Shared learnable linear for positional encoding
                               (W_1 in paper); created internally if None.
        pred_offset  (bool) : Whether to predict offsets (False for last iter).
        gamma        (float): Offset scaling factor.
        offset_bias  (bool) : Whether the offset FC layer has a bias.
        **kwargs            : Extra args forwarded to TransformerEncoderLayer.
    """

    def __init__(self,
                 feat_size: int,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 position_layer=None,
                 pred_offset: bool = True,
                 gamma: float = 1.0,
                 offset_bias: bool = False):
        super().__init__()
        self.feat_size = float(feat_size)

        self.transformer = TransformerEncoderLayer(
            dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, act_layer, norm_layer)

        self.sampler = DifferentiableSampler(gamma)

        # Shared positional encoding layer W_1: scalar position → emb vector
        self.position_layer = (position_layer
                               if position_layer is not None
                               else nn.Linear(1, dim))

        # Offset prediction layer M_1
        self.offset_layer = (nn.Linear(dim, 1, bias=offset_bias)
                             if pred_offset else None)

    def reset_offset_weight(self):
        """Zero-initialise offset layer weights (stable training start)."""
        if self.offset_layer is not None:
            nn.init.constant_(self.offset_layer.weight, 0)
            if self.offset_layer.bias is not None:
                nn.init.constant_(self.offset_layer.bias, 0)

    def forward(self, x, point, offset=None, prev_out=None):
        """
        Args:
            x       : (B, emb, L/dr)  MSFRM feature map F.
            point   : (B, n, 1)       current sampling positions P_t.
            offset  : (B, n, 1)       previous offset (None → zeros).
            prev_out: (B, n, emb)     T_{t-1} from previous iteration.

        Returns:
            out     : (B, n, emb)     fused key-point features T_t.
            new_off : (B, n, 1)       predicted offsets O_t (or None).
            new_pt  : (B, n, 1)       updated positions P_{t+1}.
        """
        if offset is None:
            offset = torch.zeros_like(point)

        # Step 1 — differentiable sampling: T_t' ∈ R^(B × n × emb)
        sampled = self.sampler(x, point, offset)

        # Step 2 — positional encoding: P_t' ∈ R^(B × n × emb)
        new_point = point + offset.detach()
        pos_enc = self.position_layer(new_point / self.feat_size)

        # Step 3 — fusion
        fused = sampled + pos_enc
        if prev_out is not None:
            fused = fused + prev_out

        # Step 4 — Transformer encoder: T_t
        out = self.transformer(fused)

        # Step 5 — offset prediction
        new_off = self.offset_layer(out) if self.offset_layer is not None else None

        return out, new_off, new_point
