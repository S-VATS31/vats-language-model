import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.model import CausalTransformer
from losses.cross_entropy import compute_loss
from metrics.perplexity import compute_perplexity
from gpu_setup import device, use_amp

def evaluate(
    model: CausalTransformer,
    dataloader: DataLoader,
    ignore_index: int = -100,
    max_batches: int | None = None
) -> tuple[float, float]:
    """Evaluate model for a set of batches.
    
    Args:
        model (CausalTransformer): Transformer architecture.
        dataloader (DataLoader): Evaluation dataloader containing examples.
        ignore_index (int): Ignore index for CE loss.
        max_batches (int, optional): Max batches to evaluate on.

    Returns:
        tuple:
            - float: Average evaluation loss.
            - float: Average evaluation perplexity.
    """
    model.eval()

    total_loss = 0
    total_ppl = 0
    succesful_batches = 0

    # get pbar
    pbar = tqdm(dataloader, desc="Evaluating")

    # no gradient tracking for eval
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            if max_batches is not None and step >= max_batches:
                break

            try:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(input_ids, attention_mask, use_cache=False)
                    loss = compute_loss(logits, labels, ignore_index=ignore_index)
                ppl = compute_perplexity(loss)

                # Accumulate loss and perplexity
                total_loss += loss.item()
                total_ppl += ppl
                succesful_batches += 1

            except Exception as e:
                if "out of memory" in str(e):
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if device.type == "mps":
                        torch.mps.empty_cache()
                return float("inf"), float("inf")
            
    if succesful_batches == 0:
        return float("inf"), float("inf")
    
    return (
        total_loss / succesful_batches,
        total_ppl / succesful_batches
    )