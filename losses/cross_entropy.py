import torch
import torch.nn.functional as F
from torch import Tensor

def compute_loss(
    logits: Tensor, 
    labels: Tensor,
    *,
    ignore_index: int = -100
) -> Tensor:
    """Compute loss using cross entropy.
    
    Args:
        logits (Tensor): Logits tensor of shape [b, t, v].
        labels (Tensor): Labels tensor of shape [b, t].
        ignore_index (int): Value to not compute loss for.

    Returns:
        Tensor: Scalar loss tensor.
    """
    assert logits.shape[:-1] == labels.shape
    if labels.dtype != torch.long:
        labels = labels.long()
    logits = logits.view(-1, logits.size(-1)) # [b*t, v]
    labels = labels.view(-1) # [b*t]
    return F.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        reduction="mean"
    )
