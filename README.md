# UltraGPT: Production-Grade Decoder-Only Transformer from Scratch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, modular, and production-grade PyTorch implementation of a **Decoder-Only Generative Pre-trained Transformer (GPT)** built entirely from mathematical first principles. Adhering strictly to the OpenAI GPT-2 architecture specification, UltraGPT includes automated data ingestion & cleaning pipelines, an optimized training engine with AMP, context-aware conversational generation, an OpenAI-compatible FastAPI serving layer, and an end-to-end phase-by-phase Jupyter Notebook.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Project Architecture](#project-architecture)
3. [Environment Setup & Installation](#environment-setup--installation)
   - [NVIDIA GPU Installation (CUDA)](#1-nvidia-gpu-installation-cuda---recommended)
   - [CPU / Apple Silicon Installation](#2-cpu--apple-silicon-installation)
   - [Installing Additional Dependencies](#3-install-core-dependencies)
4. [Centralized Configuration & Presets](#centralized-configuration--presets)
5. [Data Ingestion & ETL Pipeline](#data-ingestion--etl-pipeline)
6. [Model Architecture](#model-architecture)
7. [Training Engine & Telemetry](#training-engine--telemetry)
8. [Context-Aware Conversational Bot](#context-aware-conversational-bot)
9. [OpenAI-Compatible FastAPI Server](#openai-compatible-fastapi-server)
10. [Phase-by-Phase Jupyter Notebook](#phase-by-phase-jupyter-notebook)
11. [Quickstart CLI Guide](#quickstart-cli-guide)
12. [Verification & API Testing](#verification--api-testing)

---

## Key Features

- **Zero Black-Box Wrappers**: Every layer (`CausalSelfAttention`, `MLP`, `TransformerBlock`, `GPT`) is constructed using raw `torch.nn.Module`.
- **FlashAttention & Fallback**: Native `torch.nn.functional.scaled_dot_product_attention` with automatic graceful fallback to standard scaled dot-product attention for unsupported hardware.
- **Decoupled Weight Decay**: Custom AdamW parameter grouping separating 2D weight matrices from 1D biases and LayerNorms.
- **Dynamic Context Management**: Context window tracking with sliding-window truncation ensuring conversation history stays within context limits while permanently preserving system instructions.
- **OpenAI API Compatibility**: Production-ready FastAPI serving layer supporting both standard JSON responses and Server-Sent Events (SSE) streaming token output.
- **Self-Contained Notebook**: [`notebook.ipynb`](notebook.ipynb) running all 11 phases in a sequential, reproducible workflow.

---

## Project Architecture

```
ultra_gpt/
├── config.py                 # Centralized dataclass configuration & architecture presets
├── requirements.txt          # Core dependencies specification
├── README.md                 # Complete system documentation & guides
├── train.py                  # Standalone CLI training engine
├── generate.py               # Standalone interactive conversational CLI
├── run_server.py             # Uvicorn entrypoint for FastAPI serving
├── notebook.ipynb            # Phase-by-phase executable Jupyter Notebook
├── data/
│   ├── __init__.py
│   ├── scraper.py            # Web crawler for Wikipedia / raw text URLs
│   ├── cleaner.py            # NFKC Unicode normalization, regex artifact removal, ChatML
│   ├── tokenizer.py          # BPE (tiktoken) & standalone character-level tokenizers
│   └── dataset.py            # Causal LM shifting (x=t[:-1], y=t[1:]), dynamic pinning
├── model/
│   ├── __init__.py
│   ├── attention.py          # Multi-Head CausalSelfAttention with FlashAttention/fallback
│   ├── mlp.py                # Position-wise MLP with GELU(approx="tanh")
│   ├── block.py              # Pre-LayerNorm Transformer Block
│   └── gpt.py                # Core GPT model, weight tying & parameter counter
├── trainer/
│   ├── __init__.py
│   ├── optimizer.py          # AdamW with 2D weight decay decoupling
│   ├── lr_scheduler.py       # Linear warmup + Cosine decay schedule
│   ├── checkpoint.py         # Rolling (latest) and versioned snapshot checkpoint manager
│   └── engine.py             # AMP training loop, telemetry, safeguards & loss plots
├── inference/
│   ├── __init__.py
│   ├── sampler.py            # Temp, Top-K, Nucleus (Top-P) & repetition penalty sampler
│   └── bot.py                # Context-aware conversational bot with sliding window memory
└── server/
    ├── __init__.py
    └── app.py                # Standalone OpenAI-compatible FastAPI server (SSE + JSON)
```

---

## Environment Setup & Installation

### 1. NVIDIA GPU Installation (CUDA) - Recommended

To leverage hardware acceleration, automatic mixed precision (AMP), and FlashAttention on NVIDIA GPUs, install the CUDA-enabled build of PyTorch matching your installed NVIDIA driver:

#### A. Check your NVIDIA Driver Version
Open PowerShell or Terminal and check your driver and maximum supported CUDA version:
```powershell
nvidia-smi
```

#### B. Install PyTorch with CUDA Support
- **For CUDA 12.1 / 12.4 (Modern Drivers >= 525.xx):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **For CUDA 11.8 (Drivers >= 450.80 / 472.xx, such as Quadro T2000 / GTX 10-series):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

#### C. Verify GPU Detection
Verify that PyTorch identifies your CUDA device:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### 2. CPU / Apple Silicon Installation

If running on CPU or macOS (Apple Silicon MPS):
```bash
pip install torch
```

---

### 3. Install Core Dependencies

Once PyTorch is installed, install the supporting libraries:
```bash
pip install -r requirements.txt
```

Or install them directly:
```bash
pip install fastapi "uvicorn[standard]" pydantic beautifulsoup4 requests matplotlib numpy tqdm tiktoken
```

---

## Centralized Configuration & Presets

Configuration is controlled via [`config.py`](config.py) through the `GPTConfig` dataclass:

```python
from config import GPTConfig

# Instantiate standard presets
cfg = GPTConfig.from_preset("gpt-nano")
# Or with custom overrides
cfg_custom = GPTConfig.from_preset("gpt-micro", block_size=512, dropout=0.05)
```

### Standard Presets:
| Preset | Layers | Heads | Embedding Dim | Context Window | Parameters | Target Workload |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `gpt-nano` | 3 | 3 | 48 | 128 | ~1.0M | Instant CPU debugging & unit tests |
| `gpt-micro` | 4 | 4 | 128 | 256 | ~10.5M | Single-GPU rapid pre-training |
| `gpt-mini` | 6 | 6 | 192 | 512 | ~25.8M | Desktop & edge workstation training |
| `gpt2-small`| 12 | 12 | 768 | 1024 | ~124.4M | Standard OpenAI GPT-2 architecture |

---

## Data Ingestion & ETL Pipeline

Located in [`data/`](data/):
1. **Scraping ([`data/scraper.py`](data/scraper.py))**: Extracts clean paragraphs from specified Wikipedia articles or raw URLs using `BeautifulSoup` with custom User-Agent headers.
2. **Text Cleaning ([`data/cleaner.py`](data/cleaner.py))**:
   - `unicodedata.normalize("NFKC", text)` eliminates Unicode variance.
   - Regex-based removal of citation tags (`[1]`, `[citation needed]`, `[edit]`), HTML residual tags, non-printable characters, and excessive whitespace.
3. **Dialogue Formatter ([`data/cleaner.py`](data/cleaner.py))**: Formats conversational turns into structured ChatML:
   ```
   <|im_start|>system\nYou are a helpful AI assistant.<|im_end|>
   <|im_start|>user\nWhat is a transformer?<|im_end|>
   <|im_start|>assistant\n
   ```
4. **Tokenizers ([`data/tokenizer.py`](data/tokenizer.py))**:
   - `BPETokenizer`: GPT-2 Byte-Pair Encoding (`tiktoken`) supporting special tokens `<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`.
   - `CharTokenizer`: Completely standalone character-level tokenizer with JSON vocabulary serialization.
5. **Causal Dataset ([`data/dataset.py`](data/dataset.py))**:
   - Chunks sequences of length `block_size + 1` into shifted pairs:
     $$x = \text{tokens}[i : i + \text{block\_size}], \quad y = \text{tokens}[i + 1 : i + \text{block\_size} + 1]$$
   - 90/10 train/validation split with dynamic CUDA memory pinning (`pin_memory=True`).

---

## Model Architecture

Located in [`model/`](model/):
- **Causal Self-Attention ([`model/attention.py`](model/attention.py))**:
  - Linear projection $W_{\text{qkv}}$ mapping $d_{\text{model}} \to 3 \times d_{\text{model}}$.
  - FlashAttention via `torch.nn.functional.scaled_dot_product_attention`.
  - Strict lower-triangular causal autoregressive mask:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V, \quad M_{ij} = \begin{cases} 0 & i \ge j \\ -\infty & i < j \end{cases}$$
- **Feed-Forward MLP ([`model/mlp.py`](model/mlp.py))**:
  - Projection to $4 \times d_{\text{model}} \to \text{GELU}(\text{approximate}="tanh") \to \text{Linear} \to \text{Dropout}$.
- **Transformer Block ([`model/block.py`](model/block.py))**:
  - Pre-LayerNorm formulation:
    $$x \leftarrow x + \text{Attention}(\text{LayerNorm}_1(x))$$
    $$x \leftarrow x + \text{MLP}(\text{LayerNorm}_2(x))$$
- **Core GPT ([`model/gpt.py`](model/gpt.py))**:
  - Token embedding (`wte`) + learned positional embedding (`wpe`).
  - **Weight Tying**: Output linear projection layer shares weights with input token embeddings (`lm_head.weight = wte.weight`).
  - GPT-2 standard parameter initialization ($\mathcal{N}(0, 0.02)$ with scaled standard deviation $0.02 / \sqrt{2 \cdot n_{\text{layer}}}$ for residual projections).
  - Trainable and total parameter counting utility.

---

## Training Engine & Telemetry

Located in [`trainer/`](trainer/):
- **Decoupled AdamW ([`trainer/optimizer.py`](trainer/optimizer.py))**: Parameters are partitioned:
  - 2D weight tensors (linear layers, embeddings) receive `weight_decay=0.1`.
  - 1D tensors (biases, LayerNorm scale and shifts) receive `weight_decay=0.0`.
- **Cosine Warmup LR Schedule ([`trainer/lr_scheduler.py`](trainer/lr_scheduler.py))**:
  - Linear warmup for `warmup_iters` steps from 0 to peak learning rate.
  - Cosine decay down to `min_lr` over `lr_decay_iters`.
- **Automatic Mixed Precision (AMP)**: Uses native `torch.amp.autocast` (`fp16` or `bf16`) with `torch.amp.GradScaler`.
- **Checkpoint Manager ([`trainer/checkpoint.py`](trainer/checkpoint.py))**:
  - Rolling `latest_checkpoint.pt` containing model weights, optimizer state, scaler state, step, and loss.
  - Periodic snapshot checkpoints: `checkpoint_step_{step}.pt`.
  - Seamless state restoration and resume capability.
- **Telemetry & Visualization ([`trainer/engine.py`](trainer/engine.py))**:
  - Logs step, train loss, val loss, LR, elapsed time, and processing throughput (tokens/second).
  - Numerical safeguard: Aborts early on NaN/Inf loss explosion.
  - Automatically exports loss curves and learning rate trajectory using `matplotlib`.

---

## Context-Aware Conversational Bot

Located in [`inference/`](inference/):
- **Advanced Sampler ([`inference/sampler.py`](inference/sampler.py))**:
  - **Temperature Scaling**: Adjusts output entropy.
  - **Top-K Filtering**: Limits candidate tokens to top $k$ choices.
  - **Top-P (Nucleus) Sampling**: Retains the minimal set of tokens whose cumulative probability exceeds threshold $p$.
  - **Repetition Penalty**: Scales logits of previously generated tokens to prevent degeneration.
- **Context Management ([`inference/bot.py`](inference/bot.py))**:
  - Retains multi-turn chat history.
  - Dynamic sliding-window memory: Automatically trims oldest dialogue turns to fit strictly within `block_size` while permanently preserving the system prompt.
  - Real-time token-by-token streaming generator.

---

## OpenAI-Compatible FastAPI Server

Located in [`server/`](server/):
A standalone, high-performance API server adhering to the official OpenAI API specification.

### Supported Endpoints:
1. `POST /v1/chat/completions`:
   - Accepts standard OpenAI payload (`model`, `messages`, `temperature`, `max_tokens`, `top_p`, `stream`).
   - Supports non-streaming standard JSON response.
   - Supports streaming token output via Server-Sent Events (`text/event-stream`).
2. `GET /v1/models`:
   - Returns model metadata, owner, and active preset information.
3. `GET /health`:
   - Returns health status, active compute device (`cuda`, `mps`, or `cpu`), vocabulary size, and parameter counts.

---

## Phase-by-Phase Jupyter Notebook

The included [`notebook.ipynb`](notebook.ipynb) provides a self-contained, sequential walk-through of the entire system:

1. **Phase 1: Environment Setup & Hardware Telemetry**
2. **Phase 2: Centralized Configuration & Hyperparameters (GPTConfig)**
3. **Phase 3: Data Ingestion, Scraping & Cleaning Pipeline**
4. **Phase 4: Tokenizers (BPE & Character-Level with Specials)**
5. **Phase 5: PyTorch Dataset & DataLoader (Causal LM Shifting)**
6. **Phase 6: Decoder-Only GPT Architecture from Scratch**
7. **Phase 7: Training Engine with Decoupled AdamW, Cosine Warmup, AMP & Checkpointing**
8. **Phase 8: Model Training Execution & Telemetry Demonstration**
9. **Phase 9: Post-Training Telemetry Visualization (Loss Curves & LR Schedule)**
10. **Phase 10: Context-Aware Conversational Bot (Top-k, Top-p, Repetition Penalty, Streaming)**
11. **Phase 11: OpenAI-Compatible FastAPI Serving Layer & Live Verification**

To run the notebook:
```bash
jupyter lab notebook.ipynb
# or
jupyter notebook notebook.ipynb
```

---

## Quickstart CLI Guide

### 1. Scraping Custom Datasets
You can scrape any Wikipedia topics, search queries, or web URLs into `data/dataset.txt`:
```powershell
# A. Scrape specific keywords/topics:
python scrape.py --search "Quantum computing" "Neural networks" "Astrophysics"

# B. Interactive mode (prompts you in the terminal):
python scrape.py --interactive

# C. Scrape direct URLs:
python scrape.py --urls "https://en.wikipedia.org/wiki/Deep_learning"
```

### 2. Training from Scratch
Run pre-training using any preset:
```powershell
# Automatically scrapes search topics and trains:
python train.py --preset gpt-nano --search "Artificial intelligence" "Deep learning" --max_iters 500 --batch_size 16
```

### 2. Interactive Terminal Chat
Launch an interactive streaming chat session with the trained model:
```powershell
python generate.py --preset gpt-nano --temperature 0.8 --top_k 50 --top_p 0.9 --repetition_penalty 1.1
```
*Commands inside chat:*
- Type `:reset` to clear dialogue history.
- Type `:exit` to terminate the session.

### 3. Launching the OpenAI-Compatible API Server
Start the Uvicorn server on port 8000:
```powershell
python run_server.py --port 8000 --preset gpt-nano
```

---

## Verification & API Testing

### Check Health & Device Placement:
```bash
curl http://127.0.0.1:8000/health
```
*Example response:*
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "preset": "gpt-nano",
  "vocab_size": 100,
  "block_size": 128,
  "parameters": {
    "total": 1032192,
    "trainable": 1032192
  }
}
```

### Chat Completions Request (Standard JSON):
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ultra-gpt",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain self-attention in transformers."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Streaming Chat Completions (SSE):
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ultra-gpt",
    "messages": [
      {"role": "user", "content": "Tell me about large language models."}
    ],
    "stream": true,
    "max_tokens": 100
  }'
```

### Python OpenAI Client Integration:
Because the server adheres strictly to the OpenAI specification, you can connect using the official `openai` Python SDK:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

# Streaming request
response = client.chat.completions.create(
    model="ultra-gpt",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello UltraGPT!"}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

---

## License
MIT License. Feel free to modify and adapt for research and production applications.
