"""
Context-Aware Conversational Bot with Sliding Window Memory and Streaming Generation.
"""

from __future__ import annotations

from typing import Dict, Generator, List, Optional, Set
import torch
import torch.nn as nn

from data.cleaner import DialogueFormatter
from data.tokenizer import BaseTokenizer
from .sampler import sample_next_token


class ConversationBot:
    """
    Stateful conversational agent wrapper:
    - Manages multi-turn conversation history.
    - Context memory manager: slides/truncates oldest history to strictly fit within model block_size.
    - Supports streaming token-by-token generation or synchronous execution.
    - Configurable sampling (temp, top-k, top-p, repetition penalty, stop tokens).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        system_prompt: str = "You are UltraGPT, a helpful and precise AI assistant.",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.device = device
        self.block_size = getattr(model.config, "block_size", 1024)

        self.messages: List[Dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        """Clear conversational history, retaining system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _truncate_history(self, new_user_msg: str, max_new_tokens: int) -> str:
        """
        Dynamically trims conversation history so prompt tokens + max_new_tokens <= block_size.
        Always preserves system prompt (messages[0]).
        """
        candidate_messages = list(self.messages) + [{"role": "user", "content": new_user_msg}]
        max_prompt_len = self.block_size - max_new_tokens - 8  # headroom for formatting

        while len(candidate_messages) > 2:
            prompt_str = DialogueFormatter.format_conversation(candidate_messages, add_generation_prompt=True)
            tokens = self.tokenizer.encode(prompt_str)
            if len(tokens) <= max_prompt_len:
                break
            # Remove oldest message after system prompt
            candidate_messages.pop(1)

        prompt_str = DialogueFormatter.format_conversation(candidate_messages, add_generation_prompt=True)
        return prompt_str

    @torch.no_grad()
    def stream_generate(
        self,
        user_message: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
    ) -> Generator[str, None, str]:
        """
        Stream generation token by token:
        Yields newly emitted text fragments as they are generated.
        Returns the full assistant response at end.
        """
        self.model.eval()

        prompt_text = self._truncate_history(user_message, max_new_tokens=max_tokens)
        input_ids = self.tokenizer.encode(prompt_text)
        current_tokens = list(input_ids)
        generated_token_ids: List[int] = []

        # Default stop tokens
        stops: Set[str] = {DialogueFormatter.IM_END, "<|endoftext|>"}
        if stop_tokens:
            stops.update(stop_tokens)

        # Stop token IDs
        stop_ids = {self.tokenizer.eos_token_id}
        if hasattr(self.tokenizer, "im_end_token_id"):
            stop_ids.add(self.tokenizer.im_end_token_id)

        accumulated_text = ""

        for _ in range(max_tokens):
            # Take last block_size tokens and clamp to valid vocabulary range
            context_window = current_tokens[-self.block_size :]
            vocab_size = getattr(self.model.config, "vocab_size", 50257)
            context_window = [min(max(0, tid), vocab_size - 1) for tid in context_window]
            x = torch.tensor([context_window], dtype=torch.long, device=self.device)

            logits, _ = self.model(x)  # (1, 1, vocab_size)
            next_logits = logits[0, -1, :]

            next_token = sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_token_ids,
            )

            # Check stop token IDs
            if next_token in stop_ids:
                break

            current_tokens.append(next_token)
            generated_token_ids.append(next_token)

            # Decode newest token
            piece = self.tokenizer.decode([next_token])
            accumulated_text += piece

            # Check textual stop triggers
            stop_triggered = False
            for st in stops:
                if st in accumulated_text:
                    accumulated_text = accumulated_text.split(st)[0]
                    stop_triggered = True
                    break

            if stop_triggered:
                break

            yield piece

        clean_response = accumulated_text.strip()
        # Record turn in persistent history
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": clean_response})

        return clean_response

    def generate(
        self,
        user_message: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """Synchronously generate complete response for user message."""
        gen = self.stream_generate(
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        )
        full_text = ""
        for piece in gen:
            full_text += piece
        return full_text.strip()
