import torch
import torch.nn.functional as F
from torch import Tensor

def _temp_sample(logits: Tensor, temp: float = 1.0) -> Tensor:
    """Apply temperature sampling to input logits."""
    if temp <= 0:
        return logits
    return logits / temp

def _top_k_sample(logits: Tensor, top_k: int = 0) -> Tensor:
    """Keep only the top-k logits, set others to -inf."""
    if top_k <= 0 or logits.size(-1) < top_k:
        return logits
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    min_top_k = top_k_values[..., -1, None]
    return torch.where(
        logits < min_top_k, 
        torch.tensor(float('-inf'), device=logits.device), 
        logits
    )

def _top_p_sample(logits: Tensor, top_p: float = 0.95) -> Tensor:
    """Keep only the top tokens with cumulative probability >= top_p."""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    logits_filtered = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return logits_filtered

def get_next_tokens(
    logits: Tensor,
    do_sample: bool = True,
    temp: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> Tensor:
    """Get next tokens, and apply sampling if possible.
    Greedy sampling is applied if `do_sample` is `False`.
    
    Args:
        logits (Tensor): Logits tensor of shape [b, v].
        do_sample (bool): Whether to apply sampling or not.
        temp (float): Temperature hyperparameter.
        top_k (int): Top-k hyperparameter.
        top_p (float): Top-p hyperparameter.

    Returns:
        Tensor: Generated tokens of size [b,].
    """
    if do_sample:
        logits = _temp_sample(logits, temp=temp)
        logits = _top_k_sample(logits, top_k=top_k)
        logits = _top_p_sample(logits, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs)
    return torch.argmax(logits, dim=-1)
