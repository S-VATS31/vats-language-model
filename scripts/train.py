import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from data.streaming import TextDataset
from src.model import CausalTransformer
from configs.model_args.args_5M import ModelArgs
from configs.training_args import TrainingArgs
from tokenizer.tokenizer import get_tokenizer
from optim.optimizer import get_optimizer
from schedulers.scheduler import cosine_scheduler
from trainers.trainer import train
from trainers.evaluator import evaluate
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from inference.generate import generate
from gpu_setup import device

logger = setup_logger(__name__, log_file="training.log")

def main(
    dataset_names: list[str],
    interleave_datasets: bool = True,
    resume_from_checkpoint: str | None = None,
    early_stopping_threshold: int = 3,
    split: str = "train",
    max_samples: int | None = None
) -> None:
    """Main function to train model.
    
    Args:
        dataset_name (str): Name of the dataset to be trained on.
        resume_from_checkpoint (str, optional): Resume from checkpoint if path given.
        early_stopping (int): Number of passes to wait for loss to decrease before stopping.
        text_field (str): Text field for given dataset.
        max_samples (int, optional): Number of samples to train on.
            `None` trains on entire dataset.
    """
    # initialize args
    model_args = ModelArgs()
    training_args = TrainingArgs()
    logger.info("Initialized model and training arguments.")

    # check if we can resume from checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_data = load_checkpoint(
            checkpoint_dir=resume_from_checkpoint,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            load_only_args=True
        )
        model_args = checkpoint_data.get("model_args", model_args)
        training_args = checkpoint_data.get("training_args", training_args)

    # initialize tokenizer
    tokenizer = get_tokenizer(model_args)
    logger.info("Initialized tokenizer.")

    # initialize model
    model = CausalTransformer(model_args)
    logger.info("Initialized model.")

    # initialize dataloader
    text_dataset = TextDataset(
        tokenizer=tokenizer,
        model_args=model_args,
        dataset_names=dataset_names,
        split=split,
        max_samples=max_samples,
        interleave=interleave_datasets
    )
    logger.info("Initialized text dataset.")

    # 90% train, 10% validation
    train_len = int(0.9 * len(text_dataset))
    eval_len = len(text_dataset) - train_len
    train_dataset, eval_dataset = random_split(text_dataset, [train_len, eval_len])

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=training_args.num_workers,
        pin_memory=training_args.pin_memory
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        num_workers=training_args.num_workers,
        pin_memory=training_args.pin_memory
    )
    logger.info("Initialized train and evaluation dataloaders.")

    # get total training steps
    num_training_steps = training_args.max_train_tokens // (
        training_args.batch_size * model_args.max_seq_len
    )

    # get optimizer, scheduler, gradscaler
    optimizer = get_optimizer(
        model=model,
        learning_rate=training_args.learning_rate,
        betas=training_args.betas,
        eps=training_args.epsilon,
        weight_decay=training_args.weight_decay,
        fused=training_args.fused
    )
    scheduler = cosine_scheduler(
        optimizer=optimizer,
        num_warmup_steps=int(training_args.warmup_ratio*num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=training_args.num_cycles
    )
    scaler = GradScaler() if device.type == "cuda" else None
    logger.info("Initialized optimizer, scheduler, and scaler.")

    # initialize losses and perplexity
    best_train_loss, best_train_ppl = float("inf"), float("inf")
    best_eval_loss, best_eval_ppl = float("inf"), float("inf")

    # initialize token args
    total_tokens_seen = 0
    last_logged_tokens = 0
    early_stopping_counter = 0
    stop_early = False
    last_save_tokens = 0
    last_generation_tokens = 0
    last_clear_cache_tokens = 0

    # update tokens seen and loss from checkpoint if available
    if resume_from_checkpoint is not None:
        checkpoint_data = load_checkpoint(
            checkpoint_dir=resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            load_only_args=False
        )
        total_tokens_seen = checkpoint_data.get("tokens_seen", total_tokens_seen)
        best_eval_loss = checkpoint_data.get("loss", best_eval_loss)

    logger.info("Starting training")
        
    # Training started
    while total_tokens_seen < training_args.max_train_tokens and not stop_early:
        train_loss, train_ppl, tokens_seen, stop_early = train(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            max_train_tokens=training_args.max_train_tokens,
            logging_steps=training_args.logging_steps,
            max_failed_steps=training_args.max_failed_steps,
            grad_accum_steps=training_args.grad_accum_steps,
            max_norm=training_args.max_norm,
            ignore_index=training_args.ignore_index,
            scaler=scaler
        )
        eval_loss, eval_ppl = evaluate(
            model=model, 
            dataloader=eval_loader, 
            ignore_index=training_args.ignore_index, 
            max_batches=training_args.max_eval_batches
        )
        total_tokens_seen += tokens_seen

        # Log every >= logging_tokens_freq tokens
        if total_tokens_seen - last_logged_tokens >= training_args.logging_tokens_freq:
            last_logged_tokens = total_tokens_seen
            # Update best stats
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if train_ppl < best_train_ppl:
                best_train_ppl = train_ppl
            if eval_ppl < best_eval_ppl:
                best_eval_ppl = eval_ppl

            # Save best checkpoint based on validation lm loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                early_stopping_counter = 0
                # Save best checkpoint
                best_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokens_seen=total_tokens_seen,
                    loss=eval_loss,
                    training_args=training_args,
                    model_args=model_args,
                    scaler=scaler,
                    is_best=True
                )
                logger.info(f"Saved new best checkpoint: {best_path}")
            else:
                early_stopping_counter += 1
                logger.info(f"Early stopping counter:   {early_stopping_counter}")
                logger.info(f"Early stopping threshold: {early_stopping_threshold}")

            if early_stopping_counter >= early_stopping_threshold:
                logger.info(f"Early stopping activated, best evaluation loss: {best_eval_loss}")
                break
        
        # Save regular checkpoint
        if total_tokens_seen - last_save_tokens >= training_args.logging_tokens_freq:
            last_save_tokens = total_tokens_seen
            save_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokens_seen=total_tokens_seen,
                loss=eval_loss,
                training_args=training_args,
                model_args=model_args,
                scaler=scaler,
                is_best=False
            )
            logger.info(f"Regular checkpoint saved to {save_path}")
            
        # Test generation every n tokens for coherent generation
        if total_tokens_seen - last_generation_tokens >= training_args.generation_frequency:
            last_generation_tokens = total_tokens_seen
            prompt = "Once upon a time, "
            generated_text = generate(prompt)
            logger.info(f"{prompt} -> {generated_text}")

        # Clear CUDA cache every n tokens
        if total_tokens_seen - last_clear_cache_tokens >= training_args.clear_cache_freq:
            last_clear_cache_tokens = total_tokens_seen
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if device.type == "mps":
                torch.mps.empty_cache()

    logger.info("Training Completed:")
    logger.info(f"Best Training Loss: {best_train_loss:.4f}")
    logger.info(f"Best Training Perplexity: {best_train_ppl:.4f}")
    logger.info(f"Best Evaluation Loss: {best_eval_loss:.4f}")
    logger.info(f"Best Evaluation Perplexity: {best_eval_ppl:.4f}")

if __name__ == "__main__":
    main(
        dataset_names=["tiiuae/falcon-refinedweb"],
        interleave_datasets=True,
        resume_from_checkpoint=None,
        early_stopping_threshold=3,
        split="train",
        max_samples=100_000
    )
