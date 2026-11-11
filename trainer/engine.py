"""
Production Training Engine with AMP, Telemetry, and Loss Visualization.
Includes gradient clipping, tokens/sec calculation, NaN early-exit safeguard, and matplotlib plotting.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import GPTConfig
from .checkpoint import CheckpointManager
from .lr_scheduler import CosineWarmupScheduler
from .optimizer import configure_optimizers

logger = logging.getLogger(__name__)


class Trainer:
    """
    Production-grade Training Engine for Decoder-Only GPT:
    - Native Automatic Mixed Precision (AMP) with GradScaler.
    - Decoupled AdamW optimization & Cosine Warmup LR schedule.
    - Gradient clipping and memory leak mitigation.
    - Real-time telemetry (train/val loss, lr, elapsed time, tokens/sec).
    - Post-training Matplotlib visualization of training trajectories.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GPTConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = config.device
        self.model.to(self.device)

        # Optimizer setup with decoupled weight decay
        self.optimizer = configure_optimizers(
            self.model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            device_type=self.device,
        )

        # Cosine warmup scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            learning_rate=config.learning_rate,
            min_lr=config.min_lr,
            warmup_iters=config.warmup_iters,
            lr_decay_iters=config.lr_decay_iters,
        )

        # Mixed precision setup
        # For CUDA: 'fp16' or 'bf16'; for CPU/MPS: default to float32
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.pt_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }.get(config.mixed_precision, torch.float32)

        self.use_amp = self.device_type == "cuda" and config.mixed_precision in ("fp16", "bf16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and config.mixed_precision == "fp16"))

        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)

        # History tracking for telemetry and plotting
        self.history: Dict[str, List[float]] = {
            "steps": [],
            "train_loss": [],
            "val_steps": [],
            "val_loss": [],
            "lr": [],
            "tokens_per_sec": [],
        }

    @torch.no_grad()
    def estimate_loss(self) -> Tuple[float, float]:
        """Estimate mean loss on train and validation sets over eval_iters batches."""
        self.model.eval()
        losses = {"train": 0.0, "val": 0.0}

        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            total_loss = 0.0
            count = 0
            for i, (x, y) in enumerate(loader):
                if i >= self.config.eval_iters:
                    break
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device_type, dtype=self.pt_dtype, enabled=self.use_amp):
                    _, loss = self.model(x, y)
                total_loss += loss.item()
                count += 1
            losses[split] = total_loss / max(1, count)

        self.model.train()
        return losses["train"], losses["val"]

    def train(self, resume_from_checkpoint: Optional[bool] = None) -> Dict[str, List[float]]:
        """
        Main training loop with telemetry logging, checkpointing, and safeguards.
        - If resume_from_checkpoint is True or None (auto), and latest_checkpoint.pt exists,
          it automatically resumes training from the latest weights, step, and optimizer state!
        - If resume_from_checkpoint is False, starts afresh from step 0.
        """
        start_step = 0
        should_resume = (
            resume_from_checkpoint is True
            or (resume_from_checkpoint is None and self.checkpoint_mgr.latest_path.exists())
        )

        if should_resume and self.checkpoint_mgr.latest_path.exists():
            print(f"\n[Auto-Resume] Found existing checkpoint at: {self.checkpoint_mgr.latest_path}")
            start_step, last_loss, _ = self.checkpoint_mgr.resume(
                self.model, self.optimizer, self.scaler, device=self.device
            )
            print(f"  -> Successfully restored weights at Step {start_step:,} (Loss: {last_loss:.4f})")
            if self.config.max_iters <= start_step:
                added = 1000
                self.config.max_iters = start_step + added
                self.config.lr_decay_iters = self.config.max_iters
                print(f"  -> Target step {start_step:,} already achieved. Extending training to {self.config.max_iters:,} steps (+{added:,}).")
        elif resume_from_checkpoint is True:
            raise FileNotFoundError(f"Cannot resume: No checkpoint found at {self.checkpoint_mgr.latest_path}")

        self.model.train()
        data_iter = iter(self.train_loader)
        tokens_per_batch = self.config.batch_size * self.config.block_size
        t0 = time.time()
        start_time = time.time()

        print(f"\n--- Launching UltraGPT Training on {self.device.upper()} ---")
        print(f"Max Iters: {self.config.max_iters} | Batch Size: {self.config.batch_size} | Context: {self.config.block_size}")
        print(f"Mixed Precision: {self.config.mixed_precision} | AMP Enabled: {self.use_amp}\n")

        # Throttled progress bar: updates at most once per second to prevent VS Code Jupyter buffer crashes
        pbar = tqdm(
            range(start_step, self.config.max_iters),
            desc="Training UltraGPT",
            unit="step",
            mininterval=1.0,
            dynamic_ncols=True,
        )

        ema_loss = None
        last_val_loss = None

        try:
            for step in pbar:
                step_start = time.time()

                # Dynamic LR step
                lr = self.scheduler.step(step)

                # Fetch microbatch with cycling
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                # Forward pass under AMP autocast
                with torch.amp.autocast(device_type=self.device_type, dtype=self.pt_dtype, enabled=self.use_amp):
                    _, loss = self.model(x, y)

                loss_val = loss.item()

                # Early-exit safeguard for numerical instability / NaN
                if math.isnan(loss_val) or math.isinf(loss_val):
                    raise FloatingPointError(f"Training aborted: Loss exploded to {loss_val} at step {step}!")

                # Backward pass with GradScaler
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                # Gradient clipping to stabilize Transformer gradients
                if self.config.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Calculate real-time EMA loss and instantaneous step throughput
                ema_loss = loss_val if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_val)
                step_dt = max(time.time() - step_start, 1e-5)
                step_tok_s = tokens_per_batch / step_dt

                # Update live real-time progress bar (throttled to 1 update/sec)
                pbar.set_postfix(
                    {
                        "step_loss": f"{loss_val:.4f}",
                        "ema_loss": f"{ema_loss:.4f}",
                        "val_loss": f"{last_val_loss:.4f}" if last_val_loss is not None else "n/a",
                        "lr": f"{lr:.2e}",
                        "tok/s": f"{step_tok_s:,.0f}",
                    }
                )

                # Periodic Evaluation & Telemetry
                if (step + 1) % self.config.eval_interval == 0 or step == self.config.max_iters - 1:
                    t1 = time.time()
                    dt = t1 - t0
                    tokens_processed = tokens_per_batch * self.config.eval_interval
                    tokens_per_sec = tokens_processed / max(dt, 1e-6)
                    t0 = time.time()

                    train_eval_loss, val_loss = self.estimate_loss()
                    last_val_loss = val_loss

                    self.history["steps"].append(step + 1)
                    self.history["train_loss"].append(train_eval_loss)
                    self.history["val_steps"].append(step + 1)
                    self.history["val_loss"].append(val_loss)
                    self.history["lr"].append(lr)
                    self.history["tokens_per_sec"].append(tokens_per_sec)

                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    pbar.write(
                        f"[Step {step + 1:5d}/{self.config.max_iters}] | "
                        f"Train: {train_eval_loss:.4f} | "
                        f"Val: {val_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Avg Speed: {tokens_per_sec:7,.0f} tok/s | "
                        f"Elapsed: {elapsed_str}"
                    )

                    # Periodic snapshot checkpoint
                    is_snapshot = ((step + 1) % self.config.save_checkpoint_interval == 0)
                    self.checkpoint_mgr.save(
                        self.model,
                        self.optimizer,
                        self.scaler,
                        step=step + 1,
                        loss=val_loss,
                        config=self.config,
                        is_snapshot=is_snapshot,
                    )

                # Prevent CUDA memory fragmentation leak
                if self.device_type == "cuda" and (step + 1) % 500 == 0:
                    torch.cuda.empty_cache()
        finally:
            pbar.close()
            del data_iter
            self.optimizer.zero_grad(set_to_none=True)
            if self.device_type == "cuda":
                torch.cuda.empty_cache()

        print(f"\nTraining completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}.")
        return self.history

    def plot_metrics(self, save_path: Optional[str | Path] = "training_metrics.png") -> plt.Figure:
        """Plot loss curves (train vs. validation) and learning rate trajectory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss Curve
        ax1.plot(self.history["steps"], self.history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        ax1.plot(self.history["val_steps"], self.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
        ax1.set_title("Training & Validation Loss Curve", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Iteration Step", fontsize=11)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=11)

        # Learning Rate Schedule
        ax2.plot(self.history["steps"], self.history["lr"], label="Learning Rate", color="#2ca02c", lw=2)
        ax2.set_title("Cosine Warmup Learning Rate Trajectory", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Iteration Step", fontsize=11)
        ax2.set_ylabel("Learning Rate", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Metrics plot saved to {save_path}")
        return fig
