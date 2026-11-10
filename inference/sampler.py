"""
Token Sampling Strategies for Autoregressive Generation.
Implements Temperature, Top-K, Nucleus (Top-P) Filtering, and Repetition Penalties.
"""

from __future__ import annotations

from typing import List, Optional, Set
import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[List[int]] = None,
) -> int:
    """
    Selects the next token from unnormalized logits:
    1. Applies repetition penalty to previously emitted tokens.
    2. Temperature scaling.
    3. Top-K filtering.
    4. Top-P (nucleus) cumulative mass filtering.
    5. Categorical sampling via torch.multinomial (or argmax if temperature=0).
    """
    # Flatten to 1D vector (vocab_size,)
    logits = logits.squeeze()

    # 1. Repetition Penalty
    if repetition_penalty != 1.0 and generated_tokens:
        seen_tokens = set(generated_tokens)
        for token_id in seen_tokens:
            if token_id < logits.size(0):
                val = logits[token_id].item()
                if val > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # 2. Greedy decoding when temperature is effectively zero
    if temperature <= 1e-4:
        return int(torch.argmax(logits).item())

    # Temperature Scaling
    logits = logits / temperature

    # 3. Top-K Filtering
    if top_k is not None and top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")

    # 4. Top-P (Nucleus) Filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # 5. Categorical Sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
