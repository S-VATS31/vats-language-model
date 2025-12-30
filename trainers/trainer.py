import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.model import CausalTransformer
from losses.cross_entropy import compute_loss
from metrics.perplexity import compute_perplexity
from gpu_setup import device, use_amp

def train_step(
    model: CausalTransformer,
    batch: dict[str, Tensor],
    grad_accum_steps: int,
    ignore_index: int = -100,
    scaler: GradScaler | None = None
) -> tuple[float, float, int, bool]:
    """Train for a single step.

    Args:
        model (CausalTransformer): Transformer architecture to get logits from.
        batch (dict[str, Tensor]): Dictionary containing `input_ids`, `attention_mask`, and `labels`.
        grad_accum_steps (int): Gradient accumulation to stimulate a larger batch size.
            Defaults to 1. `grad_accum_steps`>1 activated gradient accumulation.
        ignore_index (int): Value to be ignored when computing loss.
        scaler (GradScaler, optional): Scaler to scale up gradient when using autocast.

    Returns:
        tuple:
            - float: Language modeling loss for a single step.
            - float: Perplexity for a single step.
            - int: Tokens seen, not including pad tokens.
            - bool: Whether the step was succesful.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # compute loss
    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(input_ids, attention_mask, use_cache=False)
        loss = compute_loss(logits, labels, ignore_index=ignore_index)
    perplexity = compute_perplexity(loss)
    loss /= grad_accum_steps

    # get tokens in step
    tokens_in_step = attention_mask.sum().item()

    # backprop
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return (
        loss.item() * grad_accum_steps,
        perplexity,
        tokens_in_step,
        True
    )
    
def train(
    model: CausalTransformer,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    max_train_tokens: int,
    logging_steps: int,
    max_failed_steps: int,
    grad_accum_steps: int,
    max_norm: float = 1.0,
    ignore_index: int = -100,
    scaler: GradScaler | None = None
) -> tuple[float, float, int, bool]:
    """Train for a set amount of tokens.
    
    Args:
        model (CausalTransformer): Transformer architecture.
        dataloader (DataLoader): Training dataloader containing examples.
        optimizer (AdamW): AdamW optimizer.
        scheduler (LambdaLR): Learning rate scheduler.
        max_train_tokens (int): Maximum number of tokens to train on.
        logging_steps (int): Logging is done every `logging_steps`.
        max_failed_steps (int): Maximum number of failed steps to end training.
        grad_accum_steps (int): Gradient accumulation steps to stimulate a larger batch size.
        max_norm (float): Value to clip gradients to.
            grad = grad*min((max_norm/total_norm+1e-6), 1).
        ignore_index (int): All padding tokens are set to this value.
            CE loss ignores all values set to `ignore_index`.
        scaler (GradScaler, optional): Gradient scaler when using autocast.

    Returns:
        tuple:
            - float: Average training loss.
            - float: Average training perplexity.
            - int: Total tokens seen.
            - bool: Whether to apply early stopping or not.
    """
    model.train()

    total_loss = 0
    total_ppl = 0
    successful_steps = 0
    failed_steps = 0
    total_tokens_seen = 0
    stop_early = False

    # get pbar and zero gradients
    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        loss, ppl, tokens_seen, success = train_step(
            model=model,
            batch=batch,
            grad_accum_steps=grad_accum_steps,
            ignore_index=ignore_index,
            scaler=scaler
        )
        
        # check for success
        if success:
            total_loss += loss
            total_ppl += ppl
            total_tokens_seen += tokens_seen
            successful_steps += 1
            pbar.set_postfix({"tokens_seen": total_tokens_seen})
            if total_tokens_seen >= max_train_tokens:
                # TODO: add logging
                stop_early = True
                break
        else:
            failed_steps += 1
            if failed_steps >= max_failed_steps:
                # TODO: add logging
                break

        # update weights
        if (step + 1) % grad_accum_steps == 0:
            if successful_steps > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # update logs
        if (step + 1) % logging_steps == 0 and successful_steps > 0:
            avg_loss = total_loss / successful_steps
            avg_perplexity = total_ppl / successful_steps
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "perplexity": f"{avg_perplexity:.4f}",
                "lr": f"{lr:.2e}",
                "failed_steps": failed_steps,
                "tokens_seen": total_tokens_seen
            })

    # final optimizer flush
    if (successful_steps % grad_accum_steps) != 0 and successful_steps > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # return infinite loss and ppl if no success steps
    if successful_steps == 0:
        # TODO: add logging
        return float("inf"), float("inf"), 0, False
    
    return (
        total_loss / successful_steps,
        total_ppl / successful_steps,
        total_tokens_seen,
        stop_early
    )
