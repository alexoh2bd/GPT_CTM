from dataclasses import dataclass
import torch
from pathlib import Path


CWD_PATH = Path.cwd()

MODEL_PATH = CWD_PATH / "models"


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    wd: float = 0.1  # Weight Decay
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 10
    max_steps: int = 50
    betas: tuple[float] = (0.9, 0.95)
    device: torch.device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
