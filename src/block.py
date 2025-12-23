from typing import Literal

import torch.nn as nn
from torch import Tensor 
from torch.amp import autocast

from src.mlp import MLPBlock
from src.kv_cache import KVCache
from src.attention import CausalAttentionBlock
from gpu_setup import device, use_amp

class CausalTransformerBlock(nn.Module):
    """Causal transformer block stacking attention and MLP blocks."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        rope_theta: float,
        dropout_p: float,
        d_ffn: int,
        use_mlp_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_proj_bias: bool = False,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        qk_norm_type: Literal["L2", "RMS"] = "L2",
        use_windowed_attn: bool = True,
        use_flash_attn: bool = True,
        use_mqa: bool = False,
    ):
        self.mlp = MLPBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_p=dropout_p,
            rms_norm_eps=rms_norm_eps,
            use_mlp_bias=use_mlp_bias
        )
        self.attn = CausalAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            softmax_scale=softmax_scale,
            rope_theta=rope_theta,
            dropout_p=dropout_p,
            rms_norm_eps=rms_norm_eps,
            use_proj_bias=use_proj_bias,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
            qk_norm_type=qk_norm_type,
            use_windowed_attn=use_windowed_attn,
            use_flash_attn=use_flash_attn,
            use_mqa=use_mqa
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        *,
        use_causal: bool = True,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        left_window: int = -1,
        right_window: int = -1,
    ) -> Tensor:
        """Run forward pass."""
        with autocast(device_type=device.type, enabled=use_amp):
            return self.mlp(self.attn(
                x,
                padding_mask=padding_mask,
                use_causal=use_causal,
                use_cache=use_cache,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                left_window=left_window,
                right_window=right_window
            ))
