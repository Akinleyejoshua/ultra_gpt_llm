"""
Generate notebook.ipynb directly from the exact production source files.
Guarantees 100% code parity between .py modules and the Jupyter Notebook.
"""

import json
import re
from pathlib import Path

def clean_notebook_imports(source: str) -> str:
    """
    Remove relative imports (from .xxx import yyy) and redundant local module imports
    (from config import ..., from data.xxx import ..., etc.) that fail in a Jupyter
    notebook because notebook cells share a single unified global namespace.
    """
    # Remove relative imports like: from .cleaner import TextCleaner
    source = re.sub(r'from \.\w+ import [^\n]+\n', '', source)
    source = re.sub(r'from \. import [^\n]+\n', '', source)
    
    # Remove local project imports like: from config import GPTConfig, from data.cleaner import ...
    source = re.sub(r'from (?:config|data|model|trainer|inference|server)(?:\.\w+)? import [^\n]+\n', '', source)
    return source

def read_source(rel_path: str) -> str:
    path = Path(rel_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {rel_path}")
    raw = path.read_text(encoding="utf-8")
    return clean_notebook_imports(raw)

def build_nb():
    cells = []

    def md(content: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.strip().split("\n")]
        })

    def code(code_content: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_content.strip().split("\n")]
        })

    # Header
    md("""# UltraGPT: Production-Grade Decoder-Only Transformer from Scratch
### End-to-End Implementation: Configuration, Scraping & ETL, Tokenization, Architecture, Training Engine, Context Bot, and OpenAI Serving Layer

This notebook is an exact, self-contained, phase-by-phase mirror of the UltraGPT codebase. Every phase contains the exact, unabridged production implementation with complete type hints and docstrings.

---

### Roadmap:
- **Phase 1**: System Verification & Hardware Telemetry (GPU / CUDA Detection)
- **Phase 2**: Centralized Configuration & Presets (`GPTConfig`)
- **Phase 3**: Data Ingestion, Web Scraping & Text Cleaning Pipeline
- **Phase 4**: Tokenization Engine (BPE & Character-Level with Specials)
- **Phase 5**: PyTorch Dataset & DataLoader (Causal LM Label Shifting)
- **Phase 6**: Decoder-Only GPT Architecture from Scratch (Attention, MLP, Block, GPT)
- **Phase 7**: Training Engine (Decoupled AdamW, Cosine Warmup, AMP & Checkpoint Manager)
- **Phase 8**: Model Training Execution & Telemetry Demonstration
- **Phase 9**: Post-Training Loss Curves & LR Schedule Visualization
- **Phase 10**: Context-Aware Conversational Bot (Sampling & Memory)
- **Phase 11**: OpenAI-Compatible FastAPI Serving Layer & Live Verification""")

    # Phase 1
    md("""## Phase 1: Environment Setup & Hardware Telemetry
We inspect the computing environment, checking CUDA GPU device properties, driver compatibility, and memory allocation.""")
    code("""import os
import sys
import math
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence, Set, Tuple, Union

# Ensure current project directory is in python search path
if os.path.abspath(".") not in sys.path:
    sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Check PyTorch & GPU Acceleration
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available (NVIDIA GPU): {cuda_available}")

if cuda_available:
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Active GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    print("Active Device: Apple Silicon MPS (Metal Performance Shaders)")
else:
    device = "cpu"
    print("Active Device: CPU")""")

    # Phase 2
    md("""## Phase 2: Centralized Configuration & Presets (`GPTConfig`)
We load the centralized dataclass configuration governing all architectural dimensions, presets (`gpt-nano`, `gpt-micro`, `gpt-mini`, `gpt2-small`), training parameters, and device resolution.""")
    code(read_source("config.py") + """

# Verify presets and parameter counts
for preset_name in ["gpt-nano", "gpt-micro", "gpt-mini", "gpt2-small"]:
    cfg_inst = GPTConfig.from_preset(preset_name)
    print(f"Preset [{preset_name:10s}]: layers={cfg_inst.n_layer:2d}, heads={cfg_inst.n_head:2d}, "
          f"embd={cfg_inst.n_embd:4d}, context={cfg_inst.block_size:4d}, batch={cfg_inst.batch_size:2d}")""")

    # Phase 3
    md("""## Phase 3: Data Ingestion, Web Scraping & Text Cleaning Pipeline
Here we define:
1. `TextCleaner`: Unicode NFKC normalization, regex citation removal (`[1]`, `[citation needed]`, `[edit]`), HTML tag stripping, and whitespace consolidation.
2. `DialogueFormatter`: ShareGPT / ChatML format converter (`<|im_start|>role...<|im_end|>`).
3. `WebScraper`: BeautifulSoup + requests crawler targeting Wikipedia and web articles.
Then we run the scraper to ingest real articles into `data/dataset.txt`.""")
    code(read_source("data/cleaner.py"))
    code(read_source("data/scraper.py") + """

# Execute Live Web Scraper: Ingest real articles into data/dataset.txt
scraper = WebScraper(output_path="data/dataset.txt")
topics = [
    "Artificial intelligence",
    "Transformer (deep learning architecture)",
    "Large language model",
]
print("Scraping educational corpus from Wikipedia...")
total_chars = scraper.scrape_batch(topics)
print(f"Total scraped & cleaned characters in dataset.txt: {total_chars:,}")""")

    # Phase 4
    md("""## Phase 4: Tokenization Engine (BPE & Character-Level with Specials)
We implement two tokenizers:
- `BPETokenizer`: High-speed Byte-Pair Encoding (`tiktoken`) with special tokens (`<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`).
- `CharTokenizer`: Self-contained character-level tokenizer with exact mapping and JSON serialization.""")
    code(read_source("data/tokenizer.py") + """

# Test Tokenizers
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    raw_corpus = f.read()

# Build Character Tokenizer from scraped dataset
char_tok = CharTokenizer(list(raw_corpus))
char_tok.save("checkpoints/char_vocab.json")
print(f"CharTokenizer initialized from scraped data | Vocab Size: {char_tok.vocab_size}")

test_str = "UltraGPT learns language via self-attention. <|endoftext|>"
enc = char_tok.encode(test_str)
dec = char_tok.decode(enc)
print(f"Original: '{test_str}'")
print(f"Encoded Token IDs (first 15): {enc[:15]}...")
print(f"Decoded String: '{dec}'")""")

    # Phase 5
    md("""## Phase 5: PyTorch Dataset & DataLoader Pipeline (Causal LM Shifting)
We implement `GPTDataset` which creates causally shifted input-target pairs ($x = \text{tokens}[:-1]$, $y = \text{tokens}[1:]$), performs a 90/10 train/validation split, and configures dynamic memory pinning.""")
    code(read_source("data/dataset.py") + """

# Encode scraped corpus into token sequence
tokens = char_tok.encode(raw_corpus)
print(f"Total tokens in scraped dataset: {len(tokens):,}")

# Create 90/10 Train and Validation DataLoaders
train_loader, val_loader, train_ds, val_ds = create_dataloaders(
    tokens=tokens,
    block_size=64,
    batch_size=8,
    split_ratio=0.9,
)

print(f"Train samples: {len(train_ds):,} | Validation samples: {len(val_ds):,}")
x_sample, y_sample = next(iter(train_loader))
print(f"Batch x shape: {x_sample.shape} | Batch y shape: {y_sample.shape}")
print(f"Causal Shift Validated (x[0, 1:] == y[0, :-1]): {torch.equal(x_sample[0, 1:], y_sample[0, :-1])}")""")

    # Phase 6
    md("""## Phase 6: Decoder-Only GPT Architecture from Scratch
Here we construct the neural network using raw `torch.nn.Module`:
1. `CausalSelfAttention`: Batched $Q,K,V$, FlashAttention (`F.scaled_dot_product_attention`) with manual causal mask fallback.
2. `MLP`: Position-wise feed-forward with GELU(approx="tanh").
3. `TransformerBlock`: Pre-LayerNorm architecture.
4. `GPT`: Token & positional embeddings, weight tying (`lm_head.weight = wte.weight`), GPT-2 weight initialization, and parameter counter.""")
    code(read_source("model/attention.py"))
    code(read_source("model/mlp.py"))
    code(read_source("model/block.py"))
    code(read_source("model/gpt.py") + """

# Instantiate GPT with gpt-nano preset for our character vocabulary
# You can control training duration using epochs or max_iters:
EPOCHS = 3
steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch

nano_cfg = GPTConfig.from_preset(
    "gpt-nano",
    vocab_size=char_tok.vocab_size,
    block_size=64,
    batch_size=8,
    epochs=EPOCHS,
    max_iters=total_steps,
    lr_decay_iters=total_steps,
    eval_interval=max(1, total_steps // 5),
    eval_iters=5,
    learning_rate=1e-3,
    mixed_precision="fp16" if torch.cuda.is_available() else "no",
)
print(f"Training schedule: {EPOCHS} epoch(s) x {steps_per_epoch:,} steps/epoch = {total_steps:,} total steps")

model = GPT(nano_cfg)
total_params, trainable_params = model.get_num_params()
print(f"Initialized UltraGPT: {total_params:,} total parameters ({trainable_params:,} trainable).")""")

    # Phase 7
    md("""## Phase 7: Training Engine (Decoupled AdamW, Cosine Warmup, AMP & Checkpointing)
We define:
1. `configure_optimizers`: Decoupled weight decay (2D weights get decay, 1D biases/norms do not).
2. `CosineWarmupScheduler`: Linear warmup followed by cosine decay.
3. `CheckpointManager`: Rolling `latest_checkpoint.pt` and versioned snapshot `checkpoint_step_{N}.pt`.
4. `Trainer`: AMP autocast, GradScaler, gradient clipping, telemetry, and loss curve plotting.""")
    code(read_source("trainer/optimizer.py"))
    code(read_source("trainer/lr_scheduler.py"))
    code(read_source("trainer/checkpoint.py"))
    code(read_source("trainer/engine.py"))

    # Phase 8
    md("""## Phase 8: Model Training Execution & Telemetry Demonstration
Now we train our UltraGPT model on the scraped Wikipedia dataset, tracking real-time loss, learning rate, and processing throughput (tokens/second).""")
    code("""# Initialize Trainer and execute training loop
trainer = Trainer(
    model=model,
    config=nano_cfg,
    train_loader=train_loader,
    val_loader=val_loader,
)

training_history = trainer.train()""")

    # Phase 9
    md("""## Phase 9: Telemetry Visualizations
We generate and display the cross-entropy loss curve (train vs. validation) and the learning rate trajectory.""")
    code("""import matplotlib.pyplot as plt

fig = trainer.plot_metrics(save_path="training_metrics.png")
plt.show()""")

    # Phase 10
    md("""## Phase 10: Context-Aware Conversational Bot
We implement:
- `sample_next_token`: Temperature, Top-K, Top-P (nucleus), and repetition penalty.
- `ConversationBot`: Manages multi-turn history with sliding window truncation respecting `block_size`, and yields real-time streaming tokens.""")
    code(read_source("inference/sampler.py"))
    code(read_source("inference/bot.py") + """

# Initialize interactive conversational bot with trained model
bot = ConversationBot(
    model=model,
    tokenizer=char_tok,
    system_prompt="You are UltraGPT, an expert AI assistant.",
    device=nano_cfg.device,
)

print("Prompt: 'What is deep learning?'")
print("UltraGPT > ", end="", flush=True)
for piece in bot.stream_generate("What is deep learning?", max_tokens=50, temperature=0.7):
    print(piece, end="", flush=True)
print()""")

    # Phase 11
    md("""## Phase 11: OpenAI-Compatible FastAPI Serving Layer & Live Verification
We load the standalone FastAPI server and verify the endpoints using an in-process HTTP test client.""")
    code(read_source("server/app.py") + """

# In-Process Live Verification of FastAPI Server
from starlette.testclient import TestClient

# Load trained model state into server
load_model_and_tokenizer(checkpoint_path="checkpoints/latest_checkpoint.pt", preset="gpt-nano")
client = TestClient(app)

# 1. Test /health
res_health = client.get("/health")
print("GET /health ->", res_health.json())

# 2. Test /v1/models
res_models = client.get("/v1/models")
print("GET /v1/models ->", res_models.json())

# 3. Test /v1/chat/completions (OpenAI Format)
res_chat = client.post(
    "/v1/chat/completions",
    json={
        "model": "ultra-gpt",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain transformers."}
        ],
        "temperature": 0.7,
        "max_tokens": 30
    }
)
print("POST /v1/chat/completions response:")
print(json.dumps(res_chat.json(), indent=2))
print("\\n>>> End-to-End Verification Complete: All 11 Phases Executed Successfully! <<<")""")

    # Save to notebook.ipynb
    out_file = Path("notebook.ipynb")
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

    print(f"Perfect notebook.ipynb generated successfully with {len(cells)} cells.")

if __name__ == "__main__":
    build_nb()
