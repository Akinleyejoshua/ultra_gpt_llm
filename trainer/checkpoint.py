"""
Checkpoint Management: Rolling latest checkpoint and periodic versioned snapshots.
Handles full state restoration for model, optimizer, scaler, and training metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Saves and restores training checkpoints:
    - Maintains a rolling 'latest_checkpoint.pt'
    - Creates periodic versioned snapshots 'checkpoint_step_{step}.pt'
    - Preserves model state, optimizer state, scaler state, step count, and loss.
    """

    def __init__(self, checkpoint_dir: str | Path = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.checkpoint_dir / "latest_checkpoint.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        step: int,
        loss: float,
        config: Any,
        is_snapshot: bool = False,
    ) -> Path:
        """Serialize complete training state."""
        raw_model = model.module if hasattr(model, "module") else model
        state = {
            "step": step,
            "loss": loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config.to_dict() if hasattr(config, "to_dict") else config,
        }

        # Save rolling latest checkpoint
        torch.save(state, self.latest_path)

        # Save periodic versioned snapshot if requested
        if is_snapshot:
            snapshot_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save(state, snapshot_path)
            logger.info(f"Saved versioned checkpoint snapshot to {snapshot_path}")
            return snapshot_path

        return self.latest_path

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Restore state from checkpoint. If path not specified, defaults to latest_checkpoint.pt.
        Returns: (step, loss, config_dict)
        """
        target_path = Path(checkpoint_path) if checkpoint_path else self.latest_path
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {target_path}")

        logger.info(f"Loading checkpoint from {target_path} on device {device}...")
        checkpoint = torch.save if False else torch.load(target_path, map_location=device)

        raw_model = model.module if hasattr(model, "module") else model
        ckpt_state = checkpoint["model_state_dict"]
        model_dict = raw_model.state_dict()

        # Check for architectural dimension compatibility (e.g. expanding block_size or vocab)
        compatible_dict = {}
        dimension_adjusted = False

        for k, v in ckpt_state.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    compatible_dict[k] = v
                elif k == "transformer.wpe.weight" and v.shape[1] == model_dict[k].shape[1]:
                    # Positional embedding expansion (e.g. block_size 64 -> 256)
                    dimension_adjusted = True
                    min_pos = min(v.shape[0], model_dict[k].shape[0])
                    model_dict[k][:min_pos] = v[:min_pos]
                    compatible_dict[k] = model_dict[k]
                elif k in ("transformer.wte.weight", "lm_head.weight") and v.shape[1] == model_dict[k].shape[1]:
                    # Vocabulary expansion
                    dimension_adjusted = True
                    min_v = min(v.shape[0], model_dict[k].shape[0])
                    model_dict[k][:min_v] = v[:min_v]
                    compatible_dict[k] = model_dict[k]

        if dimension_adjusted:
            logger.info("Transferred compatible weights across expanded architectural dimensions.")
            model_dict.update(compatible_dict)
            raw_model.load_state_dict(model_dict)
        else:
            raw_model.load_state_dict(ckpt_state)

        if not dimension_adjusted:
            if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    logger.warning("Could not restore optimizer state, initializing fresh optimizer.")

            if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception:
                    pass

        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", 0.0)
        config_dict = checkpoint.get("config", {})

        logger.info(f"Successfully restored checkpoint from step {step} with loss {loss:.4f}.")
        return step, loss, config_dict
