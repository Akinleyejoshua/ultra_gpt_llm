"""
Build a dedicated, standalone inference.ipynb notebook for UltraGPT.
Focused purely on loading checkpoints, interactive streaming chat, sampling experiments,
and OpenAI-compatible serving.
"""

import json
from pathlib import Path

def create_inference_notebook():
    cells = []

    def md(text: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    md("""# UltraGPT: Dedicated Inference & Serving Studio
### Interactive Generation, Multi-Turn Chat, Sampling Exploration, and OpenAI API Serving

This notebook is a dedicated environment for running inference on your trained UltraGPT model. It automatically loads your saved checkpoint (`checkpoints/latest_checkpoint.pt`) and character vocabulary without running or touching any training pipelines.

---

### Features:
1. **Model Loader & Checkpoint Telemetry**: Inspects trained weights, steps, and loss.
2. **Sampling Exploration**: Compare Greedy, Top-K, Top-P (Nucleus), and Repetition Penalties.
3. **Interactive Conversational Bot**: Multi-turn dialogue with real-time token-by-token streaming.
4. **Sliding Window Memory Visualizer**: Demonstrates context management within `block_size`.
5. **OpenAI API Serving Layer**: In-process OpenAI-compatible FastAPI `/v1/chat/completions` verification.""")

    # Step 1: Environment & Imports
    md("""## 1. Environment & Hardware Detection
Detects CUDA GPU acceleration and imports dependencies.""")
    code("""import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

# Ensure project root is in path
if os.path.abspath(".") not in sys.path:
    sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from model.gpt import GPT
from data.tokenizer import CharTokenizer, BPETokenizer
from data.cleaner import DialogueFormatter
from inference.sampler import sample_next_token
from inference.bot import ConversationBot

# Device placement
device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
print(f"Active Compute Device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")""")

    # Step 2: Model Loader
    md("""## 2. Load Trained Checkpoint & Vocabulary
Restores model weights from `checkpoints/latest_checkpoint.pt` and vocabulary from `checkpoints/char_vocab.json`.""")
    code("""ckpt_path = Path("checkpoints/latest_checkpoint.pt")
vocab_path = Path("checkpoints/char_vocab.json")

if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please train a model first or check path.")

print(f"Loading checkpoint from: {ckpt_path} ...")
checkpoint = torch.load(ckpt_path, map_location=device)

# Load configuration and initialize architecture
cfg_dict = checkpoint["config"]
config = GPTConfig(**cfg_dict)
model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Load Tokenizer
if vocab_path.exists():
    tokenizer = CharTokenizer.load(vocab_path)
    print(f"Loaded CharTokenizer from: {vocab_path} (Vocab Size: {tokenizer.vocab_size})")
else:
    print("Warning: char_vocab.json not found, falling back to BPE Tokenizer")
    tokenizer = BPETokenizer()

total_params, trainable_params = model.get_num_params()

print("\\n" + "=" * 55)
print("UltraGPT Model Successfully Loaded into VRAM")
print("=" * 55)
print(f"  Preset:             {config.preset_name or 'Custom'}")
print(f"  Training Steps:     {checkpoint.get('step', 'N/A'):,}")
print(f"  Validation Loss:    {checkpoint.get('loss', 0.0):.4f}")
print(f"  Total Parameters:   {total_params:,}")
print(f"  Layers / Heads:     {config.n_layer} layers / {config.n_head} heads")
print(f"  Embedding Dim:      {config.n_embd}")
print(f"  Context Window:     {config.block_size} tokens")
print(f"  Device Placement:   {device.upper()}")
print("=" * 55)""")

    # Step 3: Sampling Experimentation
    md("""## 3. Sampling Strategy Exploration
Compare different decoding mechanisms on the exact same prompt:
- **Greedy Decoding** ($T=0$): Selects the highest-probability token deterministically.
- **Top-K Sampling**: Limits sampling to the top $k$ highest-scoring tokens.
- **Nucleus (Top-P) Sampling**: Dynamically trims the tail by keeping cumulative mass $p$.
- **Repetition Penalty**: Penalizes tokens already generated to avert degenerate loops.""")
    code("""@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt)
    current = list(tokens)
    generated = []
    block_size = getattr(model.config, "block_size", 64)

    for _ in range(max_new_tokens):
        ctx = current[-block_size:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1, :]

        next_tok = sample_next_token(
            logits=next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated,
        )

        if next_tok == tokenizer.eos_token_id:
            break

        current.append(next_tok)
        generated.append(next_tok)

    return tokenizer.decode(generated)

test_prompt = "Artificial intelligence is"
print(f"Prompt: '{test_prompt}'\\n")

print("--- 1. Greedy Decoding (temp=0.0) ---")
print(test_prompt + generate_text(model, tokenizer, test_prompt, temperature=0.0))

print("\\n--- 2. Balanced Nucleus (temp=0.7, top_p=0.9, top_k=40) ---")
print(test_prompt + generate_text(model, tokenizer, test_prompt, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1))

print("\\n--- 3. Creative Sampling (temp=1.0, top_p=0.95, top_k=60) ---")
print(test_prompt + generate_text(model, tokenizer, test_prompt, temperature=1.0, top_k=60, top_p=0.95, repetition_penalty=1.15))""")

    # Step 4: Interactive Conversational Bot
    md("""## 4. Context-Aware Conversational Bot (Real-Time Streaming)
The `ConversationBot` maintains conversation history, applies ChatML formatting (`<|im_start|>...<|im_end|>`), manages sliding-window memory to stay within `block_size`, and yields tokens in real time.""")
    code("""# Initialize Conversational Agent
bot = ConversationBot(
    model=model,
    tokenizer=tokenizer,
    system_prompt="You are UltraGPT, a helpful and precise AI assistant.",
    device=device,
)

def chat(user_message: str, max_tokens: int = 80, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9):
    print(f"\\nUser > {user_message}")
    print("UltraGPT > ", end="", flush=True)
    for token_piece in bot.stream_generate(
        user_message=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.1,
    ):
        print(token_piece, end="", flush=True)
    print()

# Multi-Turn Conversation Test
chat("What is a neural network?")
chat("Can you explain how attention works?")
chat("What are its main benefits?")""")

    # Step 5: Sliding-Window Context Memory Visualizer
    md("""## 5. Sliding-Window Memory Visualizer
When conversation turns exceed `block_size`, the bot slides its window to discard the oldest user/assistant turns while **permanently preserving the system prompt**.""")
    code("""print(f"Active Model Context Window: {bot.block_size} tokens")
print(f"Current Conversation History Messages ({len(bot.messages)} turns):")
for i, msg in enumerate(bot.messages):
    print(f"  [{i}] {msg['role'].upper()}: {msg['content'][:60]}...")

# Inspect the raw formatted ChatML string that the model receives
raw_formatted = DialogueFormatter.format_conversation(bot.messages, add_generation_prompt=True)
token_count = len(tokenizer.encode(raw_formatted))
print(f"\\nTotal tokens formatted for next generation: {token_count} / {bot.block_size} tokens")
print("\\nRaw ChatML Template Preview:")
print("-" * 40)
print(raw_formatted)
print("-" * 40)""")

    # Step 6: OpenAI-Compatible Serving Layer Test
    md("""## 6. OpenAI-Compatible FastAPI Serving Layer Test
We verify the serving API in-process using Starlette's `TestClient` to test:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (OpenAI format payload)""")
    code("""from starlette.testclient import TestClient
from server.app import app, load_model_and_tokenizer

# Initialize server state with the active model and tokenizer
load_model_and_tokenizer(checkpoint_path="checkpoints/latest_checkpoint.pt", preset=config.preset_name or "gpt-nano")
client = TestClient(app)

print("--- 1. Testing GET /health ---")
resp_health = client.get("/health")
print(json.dumps(resp_health.json(), indent=2))

print("\\n--- 2. Testing GET /v1/models ---")
resp_models = client.get("/v1/models")
print(json.dumps(resp_models.json(), indent=2))

print("\\n--- 3. Testing POST /v1/chat/completions ---")
payload = {
    "model": "ultra-gpt",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in simple terms."}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 40,
    "stream": False
}
resp_chat = client.post("/v1/chat/completions", json=payload)
chat_data = resp_chat.json()
print("Status Code:", resp_chat.status_code)
print("Response Message:")
print("  Role:   ", chat_data["choices"][0]["message"]["role"])
print("  Content:", chat_data["choices"][0]["message"]["content"])
print("  Usage:  ", chat_data["usage"])
print("\\n>>> Inference Notebook Verification Complete! <<<")""")

    # Save notebook
    out_file = Path("inference.ipynb")
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    print(f"inference.ipynb created successfully with {len(cells)} cells.")

if __name__ == "__main__":
    create_inference_notebook()
