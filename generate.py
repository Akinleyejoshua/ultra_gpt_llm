"""
Interactive Generation & Chat CLI for UltraGPT.
Demonstrates streaming token-by-token generation with temperature, top-k, top-p, and repetition penalty.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import sys

import torch

from config import GPTConfig
from data.tokenizer import BPETokenizer, CharTokenizer
from inference.bot import ConversationBot
from model.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="UltraGPT Interactive Generation CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--preset", type=str, default="gpt-nano", help="Preset if no checkpoint exists")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K filtering limit")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus mass threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate per turn")
    parser.add_argument("--bpe", action="store_true", help="Use BPE Tokenizer")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"No checkpoint found at {ckpt_path}. Bootstrapping fresh model with preset '{args.preset}'...")
        config = GPTConfig.from_preset(args.preset)
        model = GPT(config)

    model.to(device)
    model.eval()

    # Load Tokenizer
    vocab_file = Path("checkpoints/char_vocab.json")
    if args.bpe:
        tokenizer = BPETokenizer()
    elif vocab_file.exists():
        tokenizer = CharTokenizer.load(vocab_file)
    else:
        import string
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(string.ascii_letters + string.digits + string.punctuation + " \n\t")

    bot = ConversationBot(
        model=model,
        tokenizer=tokenizer,
        system_prompt="You are UltraGPT, a helpful, coherent, and knowledgeable AI assistant.",
        device=device,
    )

    print("\n" + "=" * 60)
    print("UltraGPT Interactive Session Initialized.")
    print("Commands: ':reset' to clear conversation memory, ':exit' or 'Ctrl+C' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_msg = input("\nUser > ").strip()
            if not user_msg:
                continue
            if user_msg in (":exit", ":quit"):
                print("Session terminated. Goodbye!")
                break
            if user_msg == ":reset":
                bot.reset()
                print("[Conversation memory cleared]")
                continue

            print("\nUltraGPT > ", end="", flush=True)
            gen = bot.stream_generate(
                user_message=user_msg,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            for token_chunk in gen:
                print(token_chunk, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break


if __name__ == "__main__":
    main()
