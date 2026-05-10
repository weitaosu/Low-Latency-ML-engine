"""Encoder-only transformer for ETT forecasting. Frozen at Step 4.

Every op here corresponds 1:1 to a kernel that stages 0-5 will re-implement
in C++. Do not edit without re-doing the C++ correctness checks.
"""
import math
import torch
import torch.nn as nn


def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = scores.softmax(dim=-1) @ v
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_out(attn)
        h = self.ln2(x)
        x = x + self.ff2(self.act(self.ff1(h)))
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_vars=7, seq_len=96, horizon=96,
                 d_model=128, n_heads=4, n_layers=2, d_ff=256):
        super().__init__()
        self.n_vars, self.seq_len, self.horizon = n_vars, seq_len, horizon
        self.d_model = d_model
        self.input_proj = nn.Linear(n_vars, d_model)
        self.register_buffer('pos', sinusoidal_pe(seq_len, d_model))
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, horizon * n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.input_proj(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = h.mean(dim=1)
        out = self.head(h)
        return out.view(B, self.horizon, self.n_vars)


CONFIGS = {
    "small":  dict(n_vars=7, seq_len=96, horizon=96, d_model=128, n_heads=4, n_layers=2, d_ff=256),
    "medium": dict(n_vars=7, seq_len=96, horizon=96, d_model=192, n_heads=6, n_layers=4, d_ff=768),
}


def build_model(size: str) -> TimeSeriesTransformer:
    return TimeSeriesTransformer(**CONFIGS[size])
