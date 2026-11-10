"""
Transformer Block with Pre-LayerNorm Architecture.
Sequence: x -> x + Attention(LN1(x)) -> x + MLP(LN2(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import GPTConfig
from .attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Decoder Block:
    1. LayerNorm -> Multi-Head Causal Self-Attention -> Residual Addition
    2. LayerNorm -> Feed-Forward Network (MLP) -> Residual Addition
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention block with residual connection
        x = x + self.attn(self.ln_1(x))
        # Pre-LN feed-forward block with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x
