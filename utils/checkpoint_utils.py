from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from safetensors.torch import save_file, load_file

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
    checkpoints_dir: Path = Path("lm_checkpoints"),
) -> str:
    """Save training checkpoint."""
    checkpoints_dir.mkdir(exist_ok=True)
    ckpt_dir = checkpoints_dir / ("best" if is_best else f"checkpoint_{tokens_seen}")
    ckpt_dir.mkdir(exist_ok=True)

    try:
        model_path = ckpt_dir / "model.safetensors"
        save_file(model.state_dict(), model_path)

        trainer_state = {
            "tokens_seen": tokens_seen,
            "loss": loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_args": training_args.__dict__,
            "model_args": model_args.__dict__,
        }

        if scaler is not None:
            trainer_state["scaler_state_dict"] = scaler.state_dict()

        trainer_state_path = ckpt_dir / "trainer_state.pt"
        torch.save(trainer_state, trainer_state_path)

        return str(ckpt_dir)

    except Exception:
        # TODO: add logging
        raise

def load_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None = None,
) -> dict[str, Union[int, float, dict]]:
    """Load a training checkpoint saved by `save_checkpoint`."""
    checkpoint_dir = Path(checkpoint_dir)

    try:
        model_state = load_file(checkpoint_dir / "model.safetensors")
        model.load_state_dict(model_state, strict=True)

        trainer_state = torch.load(
            checkpoint_dir / "trainer_state.pt",
            map_location=device,
        )

        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        scheduler.load_state_dict(trainer_state["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in trainer_state:
            scaler.load_state_dict(trainer_state["scaler_state_dict"])

        return {
            "tokens_seen": trainer_state["tokens_seen"],
            "loss": trainer_state["loss"],
            "training_args": trainer_state["training_args"],
            "model_args": trainer_state["model_args"],
        }

    except Exception:
        # TODO: add logging
        raise
