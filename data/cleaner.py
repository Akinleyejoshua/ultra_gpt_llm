"""
Text Cleaning Engine and Dialogue Format Converter.
Handles NFKC normalization, artifact removal, HTML strip, and MLChat/ShareGPT formatting.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Dict, List, Optional


class TextCleaner:
    """
    Production-grade text cleaning engine:
    1. Unicode Normalization (NFKC).
    2. HTML tag and entity stripping.
    3. Citation/reference artifact removal ([1], [citation needed], [edit], etc.).
    4. Redundant whitespace consolidation and non-printable char filtering.
    """

    # Compiled regex patterns for high-throughput cleaning
    CITATION_PATTERN = re.compile(r"\[(?:\d+|citation needed|edit|note \d+)\]", re.IGNORECASE)
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    NON_PRINTABLE_PATTERN = re.compile(r"[^\x20-\x7E\n\r\t\u00A0-\uFFFF]")
    MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
    MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")

    @classmethod
    def clean(
        cls,
        text: str,
        remove_urls: bool = False,
        remove_citations: bool = True,
        strip_html: bool = True,
    ) -> str:
        """Execute full cleaning pipeline on input string."""
        if not text:
            return ""

        # Step 1: Unicode NFKC Normalization
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Unescape HTML entities & strip HTML residue
        if strip_html:
            text = html.unescape(text)
            text = cls.HTML_TAG_PATTERN.sub(" ", text)

        # Step 3: Remove citation artifacts (e.g. "[1]", "[edit]")
        if remove_citations:
            text = cls.CITATION_PATTERN.sub("", text)

        # Step 4: Optional URL removal
        if remove_urls:
            text = cls.URL_PATTERN.sub("", text)

        # Step 5: Filter non-printable control characters (preserving \n and \t)
        text = cls.NON_PRINTABLE_PATTERN.sub("", text)

        # Step 6: Collapse whitespace while preserving structural paragraph breaks
        lines = [cls.MULTI_SPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = cls.MULTI_NEWLINE_PATTERN.sub("\n\n", text)

        return text.strip()

    @classmethod
    def deduplicate_documents(cls, text: str, delimiter: str = "<|endoftext|>") -> tuple[str, int, int]:
        """
        Deduplicate document/dialogue chunks separated by delimiter while preserving first-seen order.
        Returns (deduplicated_text, original_doc_count, unique_doc_count).
        """
        chunks = text.split(delimiter)
        seen = set()
        unique_chunks: list[str] = []
        original_count = 0

        for chunk in chunks:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue
            original_count += 1
            # Normalize whitespace for fuzzy duplicate detection
            hash_key = re.sub(r"\s+", " ", chunk_stripped)
            if hash_key not in seen:
                seen.add(hash_key)
                unique_chunks.append(chunk_stripped)

        delim_spaced = f"\n\n{delimiter}\n\n"
        deduped_text = delim_spaced.join(unique_chunks)
        if deduped_text:
            deduped_text += delim_spaced
        return deduped_text, original_count, len(unique_chunks)

    @classmethod
    def deduplicate_file(cls, filepath: str | Path, delimiter: str = "<|endoftext|>") -> tuple[int, int]:
        """Deduplicate in-place a dataset file separated by delimiter."""
        from pathlib import Path
        p = Path(filepath)
        if not p.exists():
            return 0, 0
        raw = p.read_text(encoding="utf-8")
        deduped, orig_count, uniq_count = cls.deduplicate_documents(raw, delimiter)
        p.write_text(deduped, encoding="utf-8")
        return orig_count, uniq_count


class DialogueFormatter:
    """
    Format converter for multi-turn structured dialogues and thinking models:
    MLChat / ShareGPT / ChatML style with native Chain-of-Thought (<think>...</think>) support:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    <think>
    {internal_reasoning_steps}
    </think>
    {final_response}<|im_end|>
    """

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINK_START = "<think>"
    THINK_END = "</think>"

    @classmethod
    def format_turn(cls, role: str, content: str, reasoning: Optional[str] = None) -> str:
        """Format a single dialogue turn with optional thinking chain-of-thought."""
        clean_content = content.strip()
        if role == "assistant" and reasoning and reasoning.strip():
            clean_reasoning = reasoning.strip()
            body = f"{cls.THINK_START}\n{clean_reasoning}\n{cls.THINK_END}\n{clean_content}"
        else:
            body = clean_content
        return f"{cls.IM_START}{role}\n{body}{cls.IM_END}"

    @classmethod
    def format_conversation(
        cls,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Convert list of message dicts ({'role': 'user'|'assistant'|'system', 'content': '...', 'reasoning': '...'})
        into a formatted single prompt string.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", None)
            formatted.append(cls.format_turn(role, content, reasoning=reasoning))

        result = "\n".join(formatted)
        if add_generation_prompt:
            result += f"\n{cls.IM_START}assistant\n"
        return result

    @classmethod
    def extract_thinking(cls, text: str) -> Tuple[Optional[str], str]:
        """
        Separates internal reasoning (<think>...</think>) from user-facing answer.
        Returns (reasoning_text, final_answer).
        """
        pattern = re.compile(
            rf"{re.escape(cls.THINK_START)}\s*(.*?)\s*{re.escape(cls.THINK_END)}",
            re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            reasoning = match.group(1).strip()
            answer = pattern.sub("", text).strip()
            return reasoning, answer
        return None, text.strip()

    @classmethod
    def parse_conversation(cls, text: str) -> List[Dict[str, str]]:
        """Parse structured ChatML string back into a list of message dicts."""
        pattern = re.compile(
            rf"{re.escape(cls.IM_START)}(system|user|assistant)\n(.*?){re.escape(cls.IM_END)}",
            re.DOTALL,
        )
        matches = pattern.findall(text)
        parsed = []
        for role, raw_content in matches:
            if role == "assistant":
                reasoning, answer = cls.extract_thinking(raw_content)
                item = {"role": role, "content": answer}
                if reasoning:
                    item["reasoning"] = reasoning
                parsed.append(item)
            else:
                parsed.append({"role": role, "content": raw_content.strip()})
        return parsed
