"""
Centralized Configuration & Hyperparameter Definitions for Decoder-Only GPT.
Production-grade dataclass supporting presets, user overrides, and serialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GPTConfig:
    # ---------------------------------------------------------
    # 1. Model Architecture Dimensions
    # ---------------------------------------------------------
    vocab_size: int = 50257       # Vocabulary size (default: GPT-2 BPE size)
    n_embd: int = 768             # Embedding dimension / hidden size
    n_layer: int = 12             # Number of Transformer blocks
    n_head: int = 12              # Number of attention heads (n_embd must be divisible by n_head)
    block_size: int = 1024        # Maximum context length (sequence length)
    dropout: float = 0.0          # Dropout probability across embeddings, attention & MLP
    bias: bool = True             # Enable bias in Linears and LayerNorms (like GPT-2)

    # ---------------------------------------------------------
    # 2. Training Hyperparameters
    # ---------------------------------------------------------
    batch_size: int = 64          # Micro-batch size per optimization step
    epochs: Optional[int] = None  # Number of complete passes over the dataset (optional)
    learning_rate: float = 6e-4   # Peak learning rate
    min_lr: float = 6e-5          # Minimum learning rate for cosine decay schedule
    warmup_iters: int = 200       # Iterations of linear warmup
    lr_decay_iters: int = 2000    # Total iterations over which learning rate decays
    max_iters: int = 2000         # Total training steps
    eval_interval: int = 100      # Validate model every N iterations
    eval_iters: int = 20          # Number of batches sampled during evaluation
    grad_clip: float = 1.0        # Maximum gradient norm for clipping
    weight_decay: float = 0.1     # Decoupled weight decay for 2D weight tensors
    mixed_precision: str = "fp16" # 'bf16', 'fp16', or 'no'

    # ---------------------------------------------------------
    # 3. System, Device & Storage
    # ---------------------------------------------------------
    device: str = "auto"          # 'auto', 'cuda', 'mps', or 'cpu'
    seed: int = 1337              # Deterministic random seed
    checkpoint_dir: str = "checkpoints" # Directory for rolling & versioned checkpoints
    save_checkpoint_interval: int = 500 # Interval for versioned snapshot creation
    dataset_path: str = "data/dataset.txt" # Path to primary raw/cleaned dataset
    preset_name: Optional[str] = None # Name of preset if loaded from one

    def __post_init__(self) -> None:
        """Validate architectural integrity and resolve runtime device."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"Embedding dimension n_embd={self.n_embd} must be divisible by n_head={self.n_head}"
            )
        if self.device == "auto":
            self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect available acceleration backend: CUDA -> MPS -> CPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def get_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        Standard architecture presets:
          - gpt-nano:  ~1.0M parameters (rapid CPU prototyping / unit tests)
          - gpt-micro: ~10.5M parameters (single consumer GPU quick training)
          - gpt-mini:  ~25.8M parameters (solid desktop / edge training)
          - gpt2-small: ~124M parameters (standard OpenAI GPT-2 architecture)
        """
        return {
            "gpt-nano": {
                "n_layer": 3,
                "n_head": 3,
                "n_embd": 48,
                "block_size": 128,
                "vocab_size": 50257,
                "dropout": 0.0,
                "bias": True,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "min_lr": 1e-4,
                "warmup_iters": 50,
                "lr_decay_iters": 1000,
                "max_iters": 1000,
            },
            "gpt-micro": {
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
                "block_size": 256,
                "vocab_size": 50257,
                "dropout": 0.1,
                "bias": True,
                "batch_size": 32,
                "learning_rate": 8e-4,
                "min_lr": 8e-5,
                "warmup_iters": 100,
                "lr_decay_iters": 2000,
                "max_iters": 2000,
            },
            "gpt-mini": {
                "n_layer": 6,
                "n_head": 6,
                "n_embd": 192,
                "block_size": 512,
                "vocab_size": 50257,
                "dropout": 0.1,
                "bias": True,
                "batch_size": 32,
                "learning_rate": 6e-4,
                "min_lr": 6e-5,
                "warmup_iters": 200,
                "lr_decay_iters": 3000,
                "max_iters": 3000,
            },
            "gpt2-small": {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
                "vocab_size": 50257,
                "dropout": 0.1,
                "bias": True,
                "batch_size": 64,
                "learning_rate": 6e-4,
                "min_lr": 6e-5,
                "warmup_iters": 200,
                "lr_decay_iters": 5000,
                "max_iters": 5000,
            },
        }

    @classmethod
    def from_preset(cls, preset_name: str, **overrides: Any) -> GPTConfig:
        """Instantiate GPTConfig using a designated preset with custom parameter overrides."""
        presets = cls.get_presets()
        if preset_name not in presets:
            raise KeyError(
                f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
            )
        preset_args = presets[preset_name].copy()
        preset_args.update(overrides)
        preset_args["preset_name"] = preset_name
        return cls(**preset_args)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_json(self, filepath: str | Path) -> None:
        """Serialize configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str | Path) -> GPTConfig:
        """Load configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
