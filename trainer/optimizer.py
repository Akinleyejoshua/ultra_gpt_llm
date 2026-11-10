"""
Optimizer Configuration with Decoupled Weight Decay.
Applies weight decay strictly to 2D tensors (weights), excluding 1D biases and LayerNorms.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 6e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """
    Categorizes parameters into decayed and non-decayed groups:
    - 2D parameters (e.g. Linear weights, embedding weights) receive weight_decay.
    - 1D parameters (biases, LayerNorm scale/bias) receive 0.0 weight_decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D normalization weights do not decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimized parameters: {len(decay_params)} tensors with decay ({num_decay_params:,} params), "
        f"{len(no_decay_params)} tensors without decay ({num_nodecay_params:,} params)."
    )

    # Use fused AdamW kernel if running on CUDA and supported
    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=1e-8,
        **extra_args,
    )
    return optimizer
