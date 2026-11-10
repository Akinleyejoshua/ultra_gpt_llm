"""
Cosine Learning Rate Scheduler with Linear Warmup.
"""

from __future__ import annotations

import math
from typing import Optional
import torch

from config import GPTConfig


class CosineWarmupScheduler:
    """
    Computes learning rate with:
    1. Linear warmup from 0 to peak learning_rate over warmup_iters.
    2. Cosine decay from peak learning_rate to min_lr over (lr_decay_iters - warmup_iters).
    3. Flat min_lr past lr_decay_iters.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        min_lr: float,
        warmup_iters: int,
        lr_decay_iters: int,
    ) -> None:
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters

    def get_lr_for_step(self, step: int) -> float:
        """Compute the learning rate for a specific global step."""
        # 1. Linear warmup
        if step < self.warmup_iters:
            return self.learning_rate * (step + 1) / (self.warmup_iters + 1e-8)
        # 2. Past decay schedule -> min_lr
        if step > self.lr_decay_iters:
            return self.min_lr
        # 3. In-between cosine decay
        decay_ratio = (step - self.warmup_iters) / max(
            1, (self.lr_decay_iters - self.warmup_iters)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def step(self, step: int) -> float:
        """Update optimizer learning rates to match schedule for current step."""
        lr = self.get_lr_for_step(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
