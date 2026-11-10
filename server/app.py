"""
OpenAI-Compatible FastAPI Serving Layer for UltraGPT.
Endpoints:
- POST /v1/chat/completions (Supports both JSON response and Server-Sent Events streaming)
- GET  /v1/models (Metadata & active architecture preset)
- GET  /health (Health check & device status)
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import GPTConfig
from data.cleaner import DialogueFormatter
from data.tokenizer import BPETokenizer, CharTokenizer
from model.gpt import GPT
from inference.sampler import sample_next_token


# =========================================================
# Pydantic Schemas strictly conforming to OpenAI API Spec
# =========================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ultra-gpt"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# =========================================================
# Global Server State & Lifespan Loader
# =========================================================

class ServerState:
    model: Optional[GPT] = None
    tokenizer: Optional[Union[BPETokenizer, CharTokenizer]] = None
    config: Optional[GPTConfig] = None
    device: str = "cpu"


state = ServerState()


def load_model_and_tokenizer(
    checkpoint_path: Optional[str] = None,
    preset: str = "gpt-nano",
    use_bpe: bool = False,
) -> None:
    """Load model and tokenizer into global server state."""
    # Determine config
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = ckpt.get("config", {})
        config = GPTConfig(**cfg_dict)
    else:
        config = GPTConfig.from_preset(preset)

    device = config.device
    model = GPT(config)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()

    # Determine tokenizer
    if use_bpe:
        try:
            tokenizer = BPETokenizer()
        except Exception:
            tokenizer = CharTokenizer()
    else:
        # Default self-contained CharTokenizer
        tokenizer = CharTokenizer()
        # Seed default printable chars
        import string
        tokenizer.build_vocab(string.ascii_letters + string.digits + string.punctuation + " \n\t")

    state.model = model
    state.config = config
    state.tokenizer = tokenizer
    state.device = device


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize default model if not already present
    if state.model is None:
        load_model_and_tokenizer(preset="gpt-nano")
    yield
    # Shutdown logic
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="UltraGPT OpenAI-Compatible API",
    description="Production-grade OpenAI-compatible Serving API for Decoder-Only GPT",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Server Endpoints
# =========================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health status and device telemetry."""
    if state.model is None or state.config is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    total_params, trainable_params = state.model.get_num_params()
    return {
        "status": "healthy",
        "device": state.device,
        "cuda_available": torch.cuda.is_available(),
        "preset": state.config.preset_name or "custom",
        "vocab_size": state.config.vocab_size,
        "block_size": state.config.block_size,
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
    }


@app.get("/v1/models", tags=["OpenAI"])
async def list_models() -> Dict[str, Any]:
    """OpenAI standard models list endpoint."""
    total_params = state.model.get_num_params()[0] if state.model else 0
    return {
        "object": "list",
        "data": [
            {
                "id": "ultra-gpt",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ultra-gpt-team",
                "permission": [],
                "root": "ultra-gpt",
                "parent": None,
                "metadata": {
                    "preset": state.config.preset_name if state.config else "gpt-nano",
                    "total_parameters": total_params,
                },
            }
        ],
    }


async def stream_chat_completions(
    req: ChatCompletionRequest,
    prompt_ids: List[int],
    req_id: str,
    created_time: int,
) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events (SSE) compatible with OpenAI streaming client."""
    # Yield initial chunk with role
    init_chunk = ChatCompletionChunk(
        id=req_id,
        created=created_time,
        model=req.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {init_chunk.model_dump_json()}\n\n"

    current_tokens = list(prompt_ids)
    generated_tokens: List[int] = []
    block_size = state.config.block_size
    stop_ids = {state.tokenizer.eos_token_id}
    if hasattr(state.tokenizer, "im_end_token_id"):
        stop_ids.add(state.tokenizer.im_end_token_id)

    max_toks = req.max_tokens or 256
    accumulated_str = ""

    for step in range(max_toks):
        context = current_tokens[-block_size:]
        x = torch.tensor([context], dtype=torch.long, device=state.device)

        with torch.no_grad():
            logits, _ = state.model(x)

        next_logits = logits[0, -1, :]
        next_token = sample_next_token(
            logits=next_logits,
            temperature=req.temperature or 0.8,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty or 1.1,
            generated_tokens=generated_tokens,
        )

        if next_token in stop_ids:
            break

        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        delta_piece = state.tokenizer.decode([next_token])
        accumulated_str += delta_piece

        # Check textual stop sequence
        if req.stop:
            stops = [req.stop] if isinstance(req.stop, str) else req.stop
            if any(st in accumulated_str for st in stops):
                break

        chunk = ChatCompletionChunk(
            id=req_id,
            created=created_time,
            model=req.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=delta_piece),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final termination chunk
    final_chunk = ChatCompletionChunk(
        id=req_id,
        created=created_time,
        model=req.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", tags=["OpenAI"])
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions endpoint.
    Processes multi-turn messages, applies ChatML dialogue formatting,
    and returns either standard JSON response or SSE token streaming.
    """
    if state.model is None or state.tokenizer is None or state.config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="UltraGPT model is not initialized",
        )

    # Format multi-turn messages into model ChatML template
    messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
    formatted_prompt = DialogueFormatter.format_conversation(messages_payload, add_generation_prompt=True)
    prompt_tokens = state.tokenizer.encode(formatted_prompt)

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            stream_chat_completions(request, prompt_tokens, req_id, created_time),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Synchronous full generation
    current_tokens = list(prompt_tokens)
    generated_tokens: List[int] = []
    block_size = state.config.block_size
    stop_ids = {state.tokenizer.eos_token_id}
    if hasattr(state.tokenizer, "im_end_token_id"):
        stop_ids.add(state.tokenizer.im_end_token_id)

    max_toks = request.max_tokens or 256
    accumulated_str = ""

    for _ in range(max_toks):
        context = current_tokens[-block_size:]
        x = torch.tensor([context], dtype=torch.long, device=state.device)

        with torch.no_grad():
            logits, _ = state.model(x)

        next_logits = logits[0, -1, :]
        next_token = sample_next_token(
            logits=next_logits,
            temperature=request.temperature or 0.8,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty or 1.1,
            generated_tokens=generated_tokens,
        )

        if next_token in stop_ids:
            break

        current_tokens.append(next_token)
        generated_tokens.append(next_token)

        piece = state.tokenizer.decode([next_token])
        accumulated_str += piece

        if request.stop:
            stops = [request.stop] if isinstance(request.stop, str) else request.stop
            if any(st in accumulated_str for st in stops):
                for st in stops:
                    accumulated_str = accumulated_str.split(st)[0]
                break

    clean_content = accumulated_str.strip()
    finish_reason = "stop" if len(generated_tokens) < max_toks else "length"

    return ChatCompletionResponse(
        id=req_id,
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=clean_content),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(generated_tokens),
            total_tokens=len(prompt_tokens) + len(generated_tokens),
        ),
    )
