"""
Full Decoder-Only Generative Pre-trained Transformer (GPT) Model.
Implements token & positional embeddings, stacked TransformerBlocks, final LayerNorm,
tied LM head, parameter initialization, forward pass with loss, and parameter counting.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from .block import TransformerBlock


class GPT(nn.Module):
    """
    Decoder-Only Generative Pre-trained Transformer (GPT).
    Direct raw PyTorch implementation adhering strictly to GPT-2 design specifications.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights according to GPT-2 standard
        self.apply(self._init_weights)

        # Apply special scaled initialization to the residual projections (c_proj)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights: normal(0, 0.02) for weights, zeros for biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> Tuple[int, int]:
        """
        Returns (total_params, trainable_params).
        If non_embedding is True, subtracts the positional embedding parameters.
        (Token embeddings are tied with lm_head, so they count once).
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            if self.transformer.wpe.weight.requires_grad:
                n_trainable -= self.transformer.wpe.weight.numel()
        return n_params, n_trainable

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for GPT model.
        Args:
            idx: Tensor of token indices with shape (Batch, Seq_Len)
            targets: Optional tensor of target token indices with shape (Batch, Seq_Len)
        Returns:
            logits: (Batch, Seq_Len, Vocab_Size) or (Batch, 1, Vocab_Size) during generation
            loss: scalar cross-entropy loss if targets is provided, otherwise None
        """
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            t = idx.size(1)

        # Safety clamp to strictly guarantee indices are within embedding bounds
        idx = torch.clamp(idx, 0, self.config.vocab_size - 1)

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward token & position embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, compute cross-entropy loss across all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference optimization: only project the last token logit to save memory
            logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Shrink model block_size (context window) if needed."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
