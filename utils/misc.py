import torch
import torch.nn as nn

def count_params(model: nn.Module) -> int:
    # all params should be trainable for pretraining
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

def get_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype
