import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

import inspect
from config import GPTConfig


# Attention Class
class attn(nn.Module):
    # queries, keys, values B,T,C
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(self.config.n_embd, self.config.n_embd * 3)
        # projection
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        # print("I", k.shape)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        # print("FJ", k.shape)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash scaled dot product attention
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Articulated scaled dot product self-attention for practice
        # att = q @ k.transpose(-2, -1) / (n_embd**-0.5)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1) @ v

        out = att.transpose(1, 2).contiguous().view(B, T, C)  # Returns shape to B, T, C
        out = self.c_proj(out)
        return out


# MLP class
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(self.config.n_embd, self.config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# Implement One Block of Attention
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.attn = attn(config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(config)
        # self.ctm = CTM(config)

    def forward(self, x):
        out = x + self.attn(self.ln_1(x))
        out = out + self.mlp(self.ln_2(out))
        return out


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )  # B, T, C
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence length {T}, T {T} < block size {self.config.block_size}"
        tok_enc = self.transformer.wte(x)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)  # shape (T)
        pos_enc = self.transformer.wpe(pos)
        out = tok_enc + pos_enc

        for block in self.transformer.h:
            out = block(out)
        out_emb = out
        out_emb = self.transformer.ln_f(out_emb)
        # print("out",out.shape)
        hidden_embeddings = out_emb
        logits = self.lm_head(out_emb)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.1
            )
        return logits, loss, hidden_embeddings

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Weight Decay 2D matrices
        # Do not weight decay biases, 1D layernorms
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # fused AdamW into a single kernels
        # Better for CUDA
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        model_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        model_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        model_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**model_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discarding the mask/buffer

        # init the GPT2 model type from Huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # discarding the mask/buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # discarding the mask/buffer

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert (
                    sd_hf[k].shape[::-1] == sd[k].shape
                ), f"{k}, {sd_hf[k].shape}, {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        print("Pulled Pretrained Weights from HuggingFace")
        return model
