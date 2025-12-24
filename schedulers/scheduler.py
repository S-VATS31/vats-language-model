import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def cosine_scheduler(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
) -> LambdaLR:
    """Custom cosine scheduler with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        # linear warmup
        if current_step < num_cycles:
            return float(current_step) / float(num_warmup_steps)
        # cosine decay
        progress = float(
            current_step - num_warmup_steps
        ) / float(
            num_training_steps - num_warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))

    return LambdaLR(optimizer, lr_lambda)
