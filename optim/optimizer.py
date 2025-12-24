import torch
from torch.optim import AdamW
from src.model import CausalTransformer

def get_optimizer(
    model: CausalTransformer,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    fused: bool = False
) -> AdamW:
    """Setup and get optimizer."""
    return AdamW(
        params=model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=(fused and torch.cuda.is_available())
    )
