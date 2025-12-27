import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from gpu_setup import device, dtype, use_amp
from src.rms_norm import RMSNorm

class MLP(nn.Module):
    """MLP module utilizing SwiGLU."""
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_p: float,
        use_mlp_bias: bool = False
    ):
        super().__init__()

        self.weight1 = nn.Linear(
            d_model, d_ffn, 
            bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.weight2 = nn.Linear(
            d_ffn, d_model, # project back to d_model
            bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.weight3 = nn.Linear(
            d_model, d_ffn,
            bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def _optimized_swiglu(self, x: Tensor) -> Tensor:
        """Optimized SwiGLU utilizing GPU kernels."""
        pass

    def _swiglu(self, x: Tensor) -> Tensor:
        """PyTorch implementation of SwiGLU."""
        return self.weight2(F.silu(self.weight1(x)) * self.weight3(x))
    
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass of SwiGLU."""
        with autocast(device_type=device.type, enabled=use_amp):
            if x.device.type == "cuda":
                return self.dropout(self._optimized_swiglu(x))
            return self.dropout(self._swiglu(x))


class MLPBlock(nn.Module):
    """MLP block applying normalization, residuals, and dropout."""
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_p: int,
        rms_norm_eps: float,
        use_mlp_bias: bool = False
    ):
        super().__init__()

        self.mlp = MLP(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_p=dropout_p,
            use_mlp_bias=use_mlp_bias
        )
        self.rms_norm = RMSNorm(
            d_model=d_model, eps=rms_norm_eps
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass of MLP block."""
        with autocast(device_type=device.type, enabled=use_amp):
            return x + self.dropout(self.mlp(self.rms_norm(x)))
