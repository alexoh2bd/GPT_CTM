from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import tiktoken
from model import GPT,GPTConfig
import time
import math

def print_mps_memory():
    if hasattr(torch.mps, 'current_allocated_memory'):
        allocated = torch.mps.current_allocated_memory()
        print(f"MPS Memory Allocated: {allocated / 1024**3:.2f} GB")

class DataLoader:
    def __init__(self, B, T):
        self.B=B
        self.T=T
        datapath = Path.cwd() / "data" / "input.txt"

        enc = tiktoken.get_encoding('gpt2')
        with open(datapath, 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.curr_pos = 0


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_pos:self.curr_pos+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.curr_pos += B*T
        if self.curr_pos + B*T+1 > len(self.tokens):
            self.curr_pos = 0
        return x, y

max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(iter):
    if iter < warmup_steps:
        return max_lr * (iter + 1) / warmup_steps
    if iter > max_steps:
        return min_lr
    
    decay_ratio = (iter - warmup_steps) / (max_steps - warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5 * (1.0 + math.cos(math.pi*decay_ratio))
    return min_lr+coeff * (max_lr - min_lr)



if __name__ == "__main__":
    # poor man's data loader
    dataset = ""
    batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    data_dir = os.path.join("data", dataset)
    n_embd = 128
    max_length = 30
    num_return_sequences = 5

    device="cpu"
    if torch.cuda.is_available():
        device="gpu"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(F"Using Device {device}")

    # model = GPT.from_pretrained("gpt2")
    model = GPT(GPTConfig())
    model.train()
    model.to(device)
    
    # compile the model, reduces Python overhead, and GPU read/writes
    model = torch.compile(model, backend ='aot_eager') # aot_eager backend for mps


    total_batch_size = 2**19
    B=4
    T=1024
    grad_accum_steps = total_batch_size // (B*T)
    train_loader = DataLoader(B=B,T=T)
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate = 6e-4, betas = (0.9,0.95), device_type=device)

    for step in range(20):
        t0 = time.time()
        optimizer.zero_grad()
        for mini_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype = torch.bfloat16):
                logits, loss = model(x, targets=y) # (B,T,vocab size)   
            loss = loss / grad_accum_steps # "Normalizing" the gradients steps
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        # Dynamically set learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        t1 = time.time()
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
        print(f"Step {step}, Loss: {loss}, norm: {norm:.4f}, dt: {(t1-t0)*1000:.2f}ms, toks/sec: {tokens_per_sec:.2f} ")
