import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from gpu_setup import device

class RoPE(nn.Module):
    """RoPE for attention module."""
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be divisible by 2, got {head_dim}")

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
            )
        )

        # use persistent=False to avoid state_dict errors
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies rotary position embeddings to input Q/K tensor."""
        with autocast(device_type=device.type, enabled=False): # no amp for RoPE
            T = x.size(1)

            pos = torch.arange(T, device=x.device, dtype=x.dtype)
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
            embedding = torch.cat([freqs, freqs], dim=-1) # [T, head_dim]

            cos_embed = embedding.cos()[None, :, None, :]
            sin_embed = embedding.sin()[None, :, None, :]

            x_even, x_odd = x[..., ::2], x[..., 1::2] # 2i for even, 2i+1 for odd
            rotated = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

            return x * cos_embed + rotated * sin_embed # [B, T, num_heads, head_dim]
        