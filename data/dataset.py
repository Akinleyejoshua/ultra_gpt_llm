"""
PyTorch Dataset and DataLoader Pipeline for Causal Language Modeling.
Features dynamic pinning, randomized context sampling, causal label shifting (x=tokens[:-1], y=tokens[1:]),
and automated 90/10 train/validation splits.
"""

from __future__ import annotations

from typing import Tuple, Union, Sequence
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """
    Dataset for Causal Language Modeling.
    Slices input tokens into chunks of length (block_size + 1) to yield shifted (x, y) pairs:
    x = chunk[:-1]
    y = chunk[1:]
    """

    def __init__(
        self,
        tokens: Union[torch.Tensor, Sequence[int]],
        block_size: int,
    ) -> None:
        if isinstance(tokens, torch.Tensor):
            self.tokens = tokens.long()
        else:
            self.tokens = torch.tensor(tokens, dtype=torch.long)

        self.block_size = block_size
        if len(self.tokens) <= self.block_size:
            raise ValueError(
                f"Token sequence length ({len(self.tokens)}) must be greater than block_size ({self.block_size})."
            )

        # Number of non-overlapping or offset samples available
        self.total_samples = len(self.tokens) - self.block_size

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract chunk of block_size + 1 tokens
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloaders(
    tokens: Union[torch.Tensor, Sequence[int]],
    block_size: int,
    batch_size: int,
    split_ratio: float = 0.9,
    pin_memory: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, GPTDataset, GPTDataset]:
    """
    Splits tokens into 90/10 train/val splits and constructs PyTorch DataLoaders
    with dynamic pinning and memory efficiency.
    """
    if not isinstance(tokens, torch.Tensor):
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    else:
        tokens_tensor = tokens.long()

    n = len(tokens_tensor)
    split_idx = int(n * split_ratio)

    train_tokens = tokens_tensor[:split_idx]
    val_tokens = tokens_tensor[split_idx:]

    train_dataset = GPTDataset(train_tokens, block_size=block_size)
    val_dataset = GPTDataset(val_tokens, block_size=block_size)

    # Dynamic pinning flag: only pin memory if CUDA is available and requested
    use_pin_memory = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset
