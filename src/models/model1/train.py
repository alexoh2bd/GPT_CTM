from dataclasses import fields
from pathlib import Path
import torch
import os
import tiktoken
import argparse
import gc

from gpt import GPT
from ctm import CTM
import time
import math

from logger import setup_logger
from config import GPTConfig, MODEL_PATH
from dataloader import DL
from tqdm import tqdm


def print_mps_memory():
    if hasattr(torch.mps, "current_allocated_memory"):
        allocated = torch.mps.current_allocated_memory()
        print(f"MPS Memory Allocated: {allocated / 1024**3:.2f} GB")


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


def main(args):
    batch_size = (
        args.batch_size
    )  # if gradient_accumulation_steps > 1, this is the micro-batch size
    max_length = args.block_size

    torch.autograd.set_detect_anomaly(True)  # debug weird slow ops

    # Setup
    logger = setup_logger()
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        wd=args.wd,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        betas=(args.beta1, args.beta2),
        device=args.device,
    )
    device = args.device

    # gpt = GPT.from_pretrained("gpt2")
    gpt = GPT(config)
    optimizer = gpt.configure_optimizers(
        weight_decay=args.wd,
        learning_rate=args.max_lr,
        betas=config.betas,
        device_type=device,
    )
    gpt.train()
    model = CTM(config)
    model.set_GPT(gpt)

    model.train()
    model.to(device)
    # --- Initialize lazy modules with a dummy forward pass ---
    # The CTM model has lazy layers that need to be initialized before the optimizer can be configured.
    # We can do this by running a dummy batch through the model once.
    with torch.no_grad():
        dummy_x = torch.randint(
            0, args.vocab_size, (batch_size, max_length), device=device
        )
        dummy_y = torch.randint(
            0, args.vocab_size, (batch_size, max_length), device=device
        )
        model(dummy_x, targets=dummy_y)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=args.max_lr, betas=config.betas, fused=device == "cuda"
    )
    # Data Loader specs
    total_batch_size = args.total_batch_size
    B = args.batch_size
    T = args.block_size
    grad_accum_steps = total_batch_size // (B * T)
    train_loader = DL(B=B, T=T)

    logger.info(
        f"\n CTM GPT Model compiled. Model contains {sum(p.numel() for p in gpt.parameters())} parameters. \nUsing Device {device} with {torch.mps.device_count()} devices. \nInitiating Training Loop."
    )

    scaler = torch.amp.GradScaler(device)

    # with tqdm(
    #     total=args.num_batches, initial=0, leave=False, position=0, dynamic_ncols=True
    # ) as pbar:
    for i in range(args.num_batches):  # iterate on batches
        logger.info(f"Batch number {i}")
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for mini_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                predictions, certainties, synch, pre_act, post_act, att_track = model(
                    x, targets=y
                )  # (B,T,vocab size)
                loss, certainty_index = model.calc_loss(predictions, certainties, y)

                logger.info(
                    f"Loss: {loss:0.3f}. Certainty={certainty_index.float().mean().item():0.2f}+-{certainty_index.float().std().item():0.2f} ({certainty_index.min().item():d}<->{certainty_index.max().item():d})"
                )
            # torch.mps.empty_cache()

            scaler.scale(loss).backward()

            loss = loss / grad_accum_steps  # "Normalizing" the gradients steps
            loss_accum += loss.detach()
            # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            gc.collect()
            # torch.mps.empty_cache()
        # Dynamically set learning rate for this iteration
        lr = get_lr(i, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        t1 = time.time()
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        summary = f"---> Loss: {loss_accum:.2f}, dt: {(t1-t0)*1000:.2f}ms, tokens/sec: {tokens_per_sec:.2f}. \n"
        # pbar.set_description(summary)
        logger.info(summary)
    torch.save(model.state_dict(), MODEL_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model with CTM.")

    # Data and Model parameters
    parser.add_argument("--dataset", type=str, default="", help="Dataset name")
    parser.add_argument("--device", type=str, default="mps", help="Device name")

    parser.add_argument("--block_size", type=int, default=128, help="Block size")
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--bias", action="store_true", help="Use bias in layers")
    parser.add_argument(
        "--no-bias", dest="bias", action="store_false", help="Don't use bias in layers"
    )
    parser.set_defaults(bias=True)

    # Training parameters
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
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
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--max_lr", type=float, default=6e-4, help="Maximum learning rate"
    )
    parser.add_argument(
        "--min_lr_ratio", type=float, default=0.1, help="Ratio of min_lr to max_lr"
    )
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Max steps for learning rate decay"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")

    args = parser.parse_args()
    args.min_lr = args.max_lr * args.min_lr_ratio
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
