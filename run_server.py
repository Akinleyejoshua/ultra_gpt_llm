"""
Entrypoint script for launching the UltraGPT OpenAI-Compatible Serving API.
Usage:
    python run_server.py --port 8000 --preset gpt-nano
"""

from __future__ import annotations

import argparse
import uvicorn
from server.app import app, load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="UltraGPT FastAPI Serving Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--preset", type=str, default="gpt-nano", help="Architecture preset")
    parser.add_argument("--bpe", action="store_true", help="Use BPE Tokenizer instead of CharTokenizer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Initializing UltraGPT (preset={args.preset}, checkpoint={args.checkpoint})...")
    load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        preset=args.preset,
        use_bpe=args.bpe,
    )
    print(f"Launching Uvicorn server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)
