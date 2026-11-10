"""
Standalone Production Training CLI for Decoder-Only GPT.
Usage:
    python train.py --preset gpt-nano --max_iters 500 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from config import GPTConfig
from data.cleaner import TextCleaner
from data.dataset import create_dataloaders
from data.scraper import WebScraper
from data.tokenizer import BPETokenizer, CharTokenizer
from model.gpt import GPT
from trainer.engine import Trainer


def prepare_sample_dataset(dataset_path: Path) -> None:
    """Prepares educational corpus if no dataset exists."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found at {dataset_path}. Ingesting high-quality corpus via WebScraper...")

    scraper = WebScraper(output_path=dataset_path)
    topics = [
        "Artificial intelligence",
        "Transformer (deep learning architecture)",
        "Generative pre-trained transformer",
        "Large language model",
    ]
    chars = scraper.scrape_batch(topics)
    print(f"Dataset ingestion complete: {chars:,} characters compiled at {dataset_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description="UltraGPT Training CLI")
    parser.add_argument("--search", nargs="+", default=None, help="Specific search queries/topics to scrape before training")
    parser.add_argument("--preset", type=str, default="gpt-nano", help="Model preset: gpt-nano, gpt-micro, gpt-mini, gpt2-small")
    parser.add_argument("--dataset", type=str, default="data/dataset.txt", help="Path to text corpus")
    parser.add_argument("--epochs", type=int, default=None, help="Number of complete training passes over dataset (overrides max_iters)")
    parser.add_argument("--max_iters", type=int, default=None, help="Override maximum training iterations")
    parser.add_argument("--batch_size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--block_size", type=int, default=None, help="Override sequence context length")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["fp16", "bf16", "no"], help="AMP precision mode")
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda, mps, cpu, auto)")
    parser.add_argument("--bpe", action="store_true", help="Use GPT-2 Byte Pair Encoding tokenizer instead of character tokenizer")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.max_iters is not None:
        overrides["max_iters"] = args.max_iters
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.block_size is not None:
        overrides["block_size"] = args.block_size
    if args.mixed_precision is not None:
        overrides["mixed_precision"] = args.mixed_precision
    if args.device is not None:
        overrides["device"] = args.device

    config = GPTConfig.from_preset(args.preset, **overrides)

    # 1. Prepare data
    data_path = Path(args.dataset)
    if args.search:
        print(f"Scraping requested topics into {data_path}: {args.search}")
        scraper = WebScraper(output_path=data_path)
        scraper.scrape_batch(args.search)
    elif not data_path.exists() or data_path.stat().st_size == 0:
        prepare_sample_dataset(data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean text
    cleaned_text = TextCleaner.clean(text)

    # 2. Tokenizer setup
    if args.bpe:
        print("Using Byte-Pair Encoding (BPE) Tokenizer...")
        tokenizer = BPETokenizer()
        tokens = tokenizer.encode(cleaned_text)
        config.vocab_size = tokenizer.vocab_size
    else:
        print("Using Character-Level Tokenizer...")
        tokenizer = CharTokenizer(list(cleaned_text))
        tokens = tokenizer.encode(cleaned_text)
        config.vocab_size = tokenizer.vocab_size
        tokenizer.save(Path(config.checkpoint_dir) / "char_vocab.json")

    print(f"Corpus processed: {len(tokens):,} tokens | Vocab Size: {config.vocab_size:,}")

    # 3. Create DataLoaders
    train_loader, val_loader, _, _ = create_dataloaders(
        tokens=tokens,
        block_size=config.block_size,
        batch_size=config.batch_size,
        split_ratio=0.9,
    )

    if args.epochs is not None:
        steps_per_epoch = len(train_loader)
        config.epochs = args.epochs
        config.max_iters = args.epochs * steps_per_epoch
        config.lr_decay_iters = config.max_iters
        print(f"Epoch schedule: {args.epochs} epoch(s) x {steps_per_epoch:,} steps/epoch = {config.max_iters:,} total steps")

    # 4. Instantiate Model
    model = GPT(config)
    total_params, trainable_params = model.get_num_params()
    print(f"Initialized UltraGPT [{args.preset}]: {total_params:,} total parameters ({trainable_params:,} trainable).")

    # 5. Launch Training
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    history = trainer.train(resume_from_checkpoint=args.resume)
    trainer.plot_metrics(save_path="training_metrics.png")
    print("Done! Metrics and latest checkpoints successfully generated.")


if __name__ == "__main__":
    main()
