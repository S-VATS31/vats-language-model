import math

import torch
from torch import Tensor

def compute_perplexity(loss: Tensor | float) -> float:
    """Compute perplexity for scalar loss tensor or float."""
    if isinstance(loss, Tensor):
        return torch.exp(loss.detach()).item()
    return math.exp(loss)
