from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from configs.training_args import TrainingArgs
from configs.model_args.args_5M import ModelArgs
from src.model import CausalTransformer
from gpu_setup import device

def save_checkpoint(
    model: CausalTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    tokens_seen: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: GradScaler | None = None,
    is_best: bool = False,
    checkpoints_dir: Path = Path("lm_checkpoints")
) -> str:
    """Save checkpoint to .pt file.

    Args:
        model (nn.Module): Transformer architecture.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler: PyTorch scheduler.
        tokens_seen (int): Number of tokens seen so far.
        loss (float): Current loss to save checkpoint to.
        training_args (TrainingArgs): Training hyperparameters.
        model_args (ModelArgs): Model hyperparameters.
        scaler (GradScaler, optional): Save if GradScaler is not None.
        is_best (bool): Whether the current checkpoint contains the lowest validation loss or not.
        checkpoints_dir (str): Directory to where checkpoints will be saved.

    Returns:
        str: Returns path to save checkpoint so it can be loaded later.
    """
    checkpoints_dir.mkdir(exist_ok=True)
    try:
        checkpoint_data = {
            'tokens_seen': tokens_seen,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'training_args': training_args.__dict__,
            'model_args': model_args.__dict__,
        }
        
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        # Create filename
        filename = "best_model.pt" if is_best else f"checkpoint_tokens_seen_{tokens_seen}.pt"
        
        # Load checkpoint data to filename
        save_path = checkpoints_dir / filename
        torch.save(checkpoint_data, save_path)
        
        return str(save_path)

    except Exception as e:
        # TODO: add logging
        raise

def load_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None = None
) -> dict[str, Union[int, float, dict]]:
    """Load checkpoint from saved .pt file.
    
    Args:
        filename (str): Filename where checkpoint is saved.
        model (nn.Module): Transformer architecture.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LambdaLR): PyTorch scheduler.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.

    Returns:
        Dict[str, Union[int, float, dict]]:
            - Dict[str, int]: Number of tokens seen so far.
            - Dict[str, float]: Loss based on training.
            - Dict[str, dict]: Training arguments and model arguments.
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state dict if using AMP
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return {
            'tokens_seen': checkpoint['tokens_seen'],
            'loss': checkpoint['loss'],
            'training_args': checkpoint['training_args'],
            'model_args': checkpoint['model_args'],
        }
        
    except Exception as e:
        # TODO: add logging
        raise
