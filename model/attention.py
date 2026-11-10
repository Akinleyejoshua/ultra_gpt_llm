"""
Multi-Head Causal Self-Attention Module with FlashAttention & Manual Fallback.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism:
    - Batched Q, K, V linear projections.
    - Scaled dot-product attention utilizing torch.nn.functional.scaled_dot_product_attention
      (FlashAttention / memory-efficient attention) with automated manual fallback.
    - Strict lower-triangular causal autoregressive masking.
    - Output projection and residual dropout.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        # Batched key, query, value projections in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask for manual attention fallback
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        # Detect FlashAttention / SDPA support
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.
        Input shape: (B, T, C)
        Output shape: (B, T, C)
        """
        B, T, C = x.size()

        # Compute query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention with FlashAttention / Fallback
        if self.has_sdpa:
            # Native PyTorch SDPA automatically chooses FlashAttention, memory efficient, or math kernel
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual fallback
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble all head outputs side by side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
