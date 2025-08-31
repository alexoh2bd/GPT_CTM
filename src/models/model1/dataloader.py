from pathlib import Path
import torch
import tiktoken

class DL:  # DataLoader
    def __init__(self, B, T):
        self.B = B
        self.T = T
        datapath = Path.cwd() / "data" / "input.txt"
        enc = tiktoken.get_encoding("gpt2")
        with open(datapath, "r") as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.curr_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_pos : self.curr_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.curr_pos += B * T
        if self.curr_pos + B * T + 1 > len(self.tokens):
            self.curr_pos = 0
        return x, y
