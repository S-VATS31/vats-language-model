import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from gpu_setup import device, dtype

class RMSNorm(nn.Module):
    """RMSNorm module for RMS normalization, better alternative to LayerNorm."""
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def _norm(self, x: Tensor) -> Tensor:
        """Compute x/RMS(x)."""
        return x / x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
    
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass of RMSNorm."""
        with autocast(device_type=device.type, enabled=False): # no amp for rmsnorm
            assert x.size(-1) == self.d_model
            return self.weight * self._norm(x)
