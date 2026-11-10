"""
Tokenizer Implementations: Byte-Pair Encoding (BPE) & Character-Level Fallback.
Both support special token handling (<|endoftext|>, <|pad|>, <|im_start|>, <|im_end|>).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding (BPE) Tokenizer wrapping GPT-2's vocabulary
    via tiktoken with explicit special tokens (<|endoftext|>, <|pad|>, <|im_start|>, <|im_end|>).
    """

    SPECIAL_TOKENS = {
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
        "<|im_start|>": 50258,
        "<|im_end|>": 50259,
    }

    def __init__(self) -> None:
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("gpt2")
        except ImportError:
            raise ImportError("tiktoken is required for BPETokenizer. Install via `pip install tiktoken`.")

        self._special_tokens = self.SPECIAL_TOKENS.copy()
        self._reverse_special_tokens = {v: k for k, v in self._special_tokens.items()}
        # Base gpt2 vocab has 50257 tokens (0..50256 where 50256 is <|endoftext|>)
        # With added pad, im_start, im_end, vocab_size becomes 50260
        self._vocab_size = 50260

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @property
    def pad_token_id(self) -> int:
        return self._special_tokens["<|pad|>"]

    @property
    def im_start_token_id(self) -> int:
        return self._special_tokens["<|im_start|>"]

    @property
    def im_end_token_id(self) -> int:
        return self._special_tokens["<|im_end|>"]

    def encode(self, text: str, allowed_special: Union[str, Set[str]] = "all") -> List[int]:
        """Encode text to token IDs handling standard and special tokens."""
        if allowed_special == "all":
            allowed = set(self._special_tokens.keys())
        elif isinstance(allowed_special, set):
            allowed = allowed_special
        else:
            allowed = set()

        return self._enc.encode(text, allowed_special=allowed)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to string."""
        return self._enc.decode(tokens)


class CharTokenizer(BaseTokenizer):
    """
    Self-contained Character-level Tokenizer with special token handling.
    Ideal for testing, zero-dependency deployment, and fast CPU training.
    """

    DEFAULT_SPECIALS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]

    def __init__(self, chars: Optional[List[str]] = None):
        self.specials = self.DEFAULT_SPECIALS.copy()
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        if chars:
            self.build_vocab(chars)

    def build_vocab(self, chars: Union[str, List[str]]) -> None:
        """Construct character-to-index and index-to-character tables."""
        unique_chars = sorted(list(set(chars)))
        # Filter out characters that might clash with special token brackets
        all_symbols = self.specials + unique_chars
        self.stoi = {ch: i for i, ch in enumerate(all_symbols)}
        self.itos = {i: ch for i, ch in enumerate(all_symbols)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi) if self.stoi else len(self.specials)

    @property
    def eos_token_id(self) -> int:
        return self.stoi.get("<|endoftext|>", 0)

    @property
    def pad_token_id(self) -> int:
        return self.stoi.get("<|pad|>", 1)

    @property
    def im_start_token_id(self) -> int:
        return self.stoi.get("<|im_start|>", 2)

    @property
    def im_end_token_id(self) -> int:
        return self.stoi.get("<|im_end|>", 3)

    def encode(self, text: str) -> List[int]:
        """Encode text, recognizing special tokens or falling back to char indices."""
        tokens: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            for sp in self.specials:
                if text.startswith(sp, i):
                    tokens.append(self.stoi[sp])
                    i += len(sp)
                    matched_special = True
                    break
            if not matched_special:
                ch = text[i]
                if ch in self.stoi:
                    tokens.append(self.stoi[ch])
                else:
                    # Map unknown char to pad or skip
                    tokens.append(self.pad_token_id)
                i += 1
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string."""
        return "".join([self.itos.get(idx, "") for idx in tokens])

    def save(self, filepath: str | Path) -> None:
        """Persist tokenizer vocabulary to JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "specials": self.specials}, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> CharTokenizer:
        """Load tokenizer vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for v, k in tok.stoi.items()}
        tok.specials = data["specials"]
        return tok
