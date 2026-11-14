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

    def _truncate_history(self, new_user_msg: str, max_new_tokens: int = 512) -> str:
        """
        Dynamically trims conversation history so prompt tokens leave ample headroom for generation.
        Always preserves system prompt (messages[0]) and the latest user message.
        """
        candidate_messages = list(self.messages) + [{"role": "user", "content": new_user_msg}]
        # Reserve room in context window for generation:
        # At least 25% of block_size (or 32 tokens) is preserved for new output tokens
        reserved_tokens = max(32, min(max_new_tokens, self.block_size // 3))
        max_prompt_len = max(32, self.block_size - reserved_tokens)

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
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
        complete_sentence: bool = True,
    ) -> Generator[str, None, str]:
        """
        Stream generation token by token with full conversational completion:
        - Terminates naturally when the model emits End-of-Turn (<|im_end|>) or EOS.
        - Supports short responses (stops early on EOS) or long responses (generates up to max_tokens).
        - Includes adaptive sentence-boundary completion to prevent cutting off mid-sentence.
        - Yields newly emitted text fragments in real time.
        """
        self.model.eval()

        prompt_text = self._truncate_history(user_message, max_new_tokens=max_tokens)
        input_ids = self.tokenizer.encode(prompt_text)
        current_tokens = list(input_ids)
        generated_token_ids: List[int] = []

        # Default stop tokens
        stops: Set[str] = {DialogueFormatter.IM_END, "<|endoftext|>", "<|im_start|>"}
        if stop_tokens:
            stops.update(stop_tokens)

        # Stop token IDs
        stop_ids = {self.tokenizer.eos_token_id}
        if hasattr(self.tokenizer, "im_end_token_id"):
            stop_ids.add(self.tokenizer.im_end_token_id)

        accumulated_text = ""
        # Adaptive budget allows finishing the current thought if cut off mid-sentence
        adaptive_limit = 40 if complete_sentence else 0
        total_token_budget = max_tokens + adaptive_limit
        sentence_terminators = {".", "!", "?", "\n", ";", "```"}

        sentence_count = 0
        min_tokens_before_stop = 20

        for step in range(total_token_budget):
            # Take last block_size tokens and clamp to valid vocabulary range
            context_window = current_tokens[-self.block_size :]
            vocab_size = getattr(self.model.config, "vocab_size", 50257)
            context_window = [min(max(0, tid), vocab_size - 1) for tid in context_window]
            model_device = next(self.model.parameters()).device
            x = torch.tensor([context_window], dtype=torch.long, device=model_device)

            logits, _ = self.model(x)
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

            # Prevent leaking partial stop tags in stream
            if not any(st.startswith(piece) and len(piece) > 2 for st in stops):
                yield piece

            # Natural stopping: if the model reaches a sentence terminator (. or ! or ?) after min tokens
            if complete_sentence and step >= min_tokens_before_stop:
                if piece in (".", "!", "?"):
                    sentence_count += 1
                    # Stop if max_tokens reached or 2 complete sentences are delivered
                    if step >= max_tokens or sentence_count >= 2:
                        break

            # Hard budget check
            if step >= total_token_budget - 1:
                break

        clean_response = accumulated_text.strip()

        # Guarantee complete sentences: trim any trailing fragmented sentence or half-word
        if complete_sentence and clean_response:
            terminal_chars = (".", "!", "?", "```", "\n")
            if not any(clean_response.endswith(t) for t in terminal_chars):
                boundaries = [
                    clean_response.rfind(". "),
                    clean_response.rfind(".\n"),
                    clean_response.rfind("! "),
                    clean_response.rfind("?\n"),
                    clean_response.rfind("? "),
                    clean_response.rfind(".\t"),
                ]
                latest_b = max(boundaries)
                if latest_b != -1:
                    clean_response = clean_response[: latest_b + 1].strip()
                elif clean_response.rfind(".") != -1:
                    clean_response = clean_response[: clean_response.rfind(".") + 1].strip()
                else:
                    # If model did not emit a period, clean up trailing partial word and seal sentence
                    last_space = clean_response.rfind(" ")
                    if last_space > 10:
                        clean_response = clean_response[:last_space].rstrip(",;:- ") + "."
                    else:
                        clean_response = clean_response.rstrip(",;:- ") + "."

        # Record turn in persistent history
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": clean_response})

        return clean_response

    def generate(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
        complete_sentence: bool = True,
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
            complete_sentence=complete_sentence,
        )
        full_text = ""
        try:
            while True:
                full_text += next(gen)
        except StopIteration as e:
            if e.value:
                return e.value

        # Fallback to trimmed complete sentence
        clean = full_text.strip()
        if complete_sentence and clean:
            terminal_chars = (".", "!", "?", "```", "\n")
            if not any(clean.endswith(t) for t in terminal_chars):
                boundaries = [
                    clean.rfind(". "),
                    clean.rfind(".\n"),
                    clean.rfind("! "),
                    clean.rfind("?\n"),
                    clean.rfind("? "),
                    clean.rfind(".\t"),
                ]
                latest_b = max(boundaries)
                if latest_b != -1:
                    clean = clean[: latest_b + 1].strip()
                elif clean.rfind(".") != -1:
                    clean = clean[: clean.rfind(".") + 1].strip()
        return clean
