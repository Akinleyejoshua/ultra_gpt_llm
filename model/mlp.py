"""
Feed-Forward Network (MLP) for Transformer Block.
Structure: Linear(n_embd, 4*n_embd) -> GELU(approx="tanh") -> Linear(4*n_embd, n_embd) -> Dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import GPTConfig


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network:
    Expands channel dimension by 4x, applies GELU with tanh approximation,
    projects back to n_embd, and applies residual dropout.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
