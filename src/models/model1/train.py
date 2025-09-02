from dataclasses import fields
from pathlib import Path
from datasets.iterable_dataset import StepExamplesIterable
import torch
import os
import tiktoken
import gc
import argparse

from gpt import GPT
from ctm import CTM
import time
import math

from logger import setup_logger
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from data.dataloader import DL


def demo_generation(model, enc, device, start_text="Once upon a time"):
    """
    Demo script to show how to use the model.generate() function with tiktoken.
    """
    # Encode the starting text
    input_ids = enc.encode(start_text, allowed_special={"<eod>"})
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print("Prompt:", start_text)
    print("-" * 40)

    # Generate continuation
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, max_new_tokens=50, temperature=0.8, top_k=40
        )[0].tolist()

    # Decode
    output_text = enc.decode(output_ids)
    print(output_text)
    # Print results
    return output_text


# Llearning rate scheduler for
def get_lr(iter, config):
    if iter < config.warmup_steps:
        return config.max_lr * (iter + 1) / config.warmup_steps
    if iter > config.max_steps:
        return config.min_lr

    decay_ratio = (iter - config.warmup_steps) / (
        config.max_steps - config.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


def build_checkpoint(
    step, model_sd, opt_sd, scaler_sd, train_losses, val_losses, arguments
):
    return {
        "step": step,
        "model_state_dict": model_sd,
        "optimizer_state_dict": opt_sd,
        "scaler_state_dict": scaler_sd,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "args": arguments,
    }


def main(args):
    project_dir = Path(__file__).resolve().parents[3]
    data_dir = project_dir / "data" / "raw"
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    prompts = [
        "If a train travels at 60 miles per hour for 2 hours, it will cover",
        "The capital of Japan is",
        "The first 5 prime numbers are",
        "In a distant future, humans discovered a way to travel faster than light. The first mission was",
        "Human: Hello, who are you? \nAI:",
        "Write a recipe for a simple sandwich using only three ingredients:",
    ]
    # GPU setup and memory management
    device = args.device
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    elif device == "mps":
        print(f"Using MPS (Apple Silicon)")
    # Setup models
    logger = setup_logger(log_file=args.logfile)
    gpt = GPT.from_pretrained("gpt2", args)
    # gpt = GPT(args)
    gpt.to(device)
    model = CTM(args)
    model.set_GPT(gpt)
    model.train()
    model.to(device)

    # Initialize lazy layers
    batch_size = args.batch_size
    max_length = args.block_size
    with torch.no_grad():
        dummy_x = torch.randint(
            0, args.vocab_size, (batch_size, max_length), device=device
        )
        model(dummy_x)

    # Optimizer setup
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        fused=(device == "cuda"),  # Use fused optimizer for CUDA
    )

    # Data loaders
    B, T = args.batch_size, args.block_size
    total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // (B * T)

    train_loader = DL(B, T, train_dir, logger)
    val_loader = DL(B, T, val_dir, logger)

    # Mixed precision setup
    scaler = torch.amp.GradScaler(device)

    # Training metrics
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    logger.info(f"CTM Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {B}, Sequence length: {T}")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"Effective batch size: {total_batch_size}")

    # Training loop
    with tqdm(total=args.num_batches) as pbar:
        for step in range(args.num_batches):

            t0 = time.time()

            # Training step
            model.train()
            optimizer.zero_grad()
            loss_accum = 0.0

            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    predictions, certainties = model(x, targets=y)
                    loss, _ = model.calc_loss(predictions, certainties, y)
                    loss = loss / grad_accum_steps  # Scale loss for accumulation

                scaler.scale(loss).backward()
                loss_accum += loss.detach()

                # Memory cleanup for GPU
                if device in ["cuda", "mps"]:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    elif device == "mps":
                        torch.mps.empty_cache()
                del predictions, certainties
                gc.collect()

            # Gradient clipping and optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Learning rate scheduling
            lr = get_lr(step, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            # Logging and metrics
            t1 = time.time()
            # Validation step
            if step % args.val_interval == 0:
                model.eval()
                val_loss_accum = 0.0
                val_steps = min(20, args.val_steps)  # Limit validation steps

                with torch.no_grad():
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch()
                        if x_val is None or y_val is None:
                            continue

                        x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(
                            device, non_blocking=True
                        )

                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            pred_val, cert_val = model(x_val, targets=y_val)
                            val_loss, _ = model.calc_loss(pred_val, cert_val, y_val)

                        val_loss_accum += val_loss.detach()

                val_loss_avg = val_loss_accum / val_steps
                val_losses.append(val_loss_avg.item())

                # Save best model
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    checkpoint = build_checkpoint(
                        step,
                        model.state_dict(),
                        optimizer.state_dict(),
                        scaler.state_dict(),
                        train_losses,
                        best_val_loss,
                        args,
                    )
                    prompt_text = ""
                    # for prompt in prompts:
                    #     prompt_text += (
                    #         demo_generation(model, enc, device, prompt) + "\n"
                    #     )

                    torch.save(checkpoint, model_dir / "best_model.pth")
                    logger.info(
                        f"New best model saved. Val loss: {val_loss_avg:.4f}\n{prompt_text}"
                    )

            dt = (t1 - t0) * 1000  # milliseconds
            tokens_per_sec = (B * T * grad_accum_steps) / (t1 - t0)
            train_losses.append(loss_accum.item())

            if step % args.log_interval == 0:
                logger.info(
                    f"Step {step:4d} | "
                    f"Train Loss: {loss_accum:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"dt: {dt:.1f}ms | "
                    f"tok/s: {tokens_per_sec:.0f}"
                )

                if len(val_losses) > 0:
                    logger.info(f"Val Loss: {val_losses[-1]:.4f}")

            # Periodic checkpointing
            if step % args.checkpoint_interval == 0 and step > 0:
                checkpoint = build_checkpoint(
                    step,
                    model.state_dict(),
                    optimizer.state_dict(),
                    scaler.state_dict(),
                    train_losses,
                    val_losses,
                    args,
                )
                torch.save(checkpoint, model_dir / f"checkpoint_step_{step}.pth")
                logger.info(f"Checkpoint saved at step {step}")
            pbar.set_description(
                f"Batch {step:4d} | "
                f"Loss: {loss_accum:.4f} | "
                f"Val: {best_val_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )
            pbar.update(1)

        # Final model save and generation demo
        final_checkpoint = build_checkpoint(
            args.num_batches,
            model.state_dict(),
            optimizer.state_dict(),
            scaler.state_dict(),
            train_losses,
            val_losses,
            args,
        )
        torch.save(final_checkpoint, model_dir / "final_model.pth")

        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            logger.info(f"Best val loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model with CTM.")

    # Data and Model parameters
    parser.add_argument("--dataset", type=str, default="", help="Dataset name")
    parser.add_argument(
        "--logfile", type=str, default="trainthink.log", help="Log File name"
    )

    parser.add_argument("--device", type=str, default="mps", help="Device name")
    parser.add_argument("--iterations", type=int, default=12, help="CTM iterations")
    # Training parameters
    parser.add_argument(
        "--num_batches",
        type=int,
        default=4000,
        help="Total Number of batches in Training Run",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for data loader"
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=2**10,
        help="Total batch size for gradient accumulation",
    )
    parser.add_argument("--block_size", type=int, default=128, help="Block size")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bias", action="store_true", help="Use bias in layers")
    parser.add_argument(
        "--no-bias", dest="bias", action="store_false", help="Don't use bias in layers"
    )
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--max_lr", type=float, default=6e-4, help="Maximum learning rate"
    )
    parser.add_argument(
        "--min_lr_ratio", type=float, default=0.1, help="Ratio of min_lr to max_lr"
    )
    parser.add_argument(
        "--hidden_dimensions",
        type=int,
        default=4,
        help="Hidden Dimensions in Neuron Level Models",
    )
    parser.add_argument(
        "--memory_length",
        type=int,
        default=16,
        help="Input Memory Length with Neuron Level Models",
    )
    parser.add_argument("--warmup_steps", type=int, default=200, help="Warmup steps")
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Max steps for learning rate decay"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")

    parser.add_argument(
        "--val_interval", type=int, default=200, help="Validation interval"
    )
    parser.add_argument(
        "--val_steps", type=int, default=10, help="Number of validation steps"
    )
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=1000, help="Checkpoint interval"
    )

    parser.set_defaults(bias=True)

    args = parser.parse_args()
    args.min_lr = args.max_lr * args.min_lr_ratio
    return args


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = parse_args()
    main(args)
