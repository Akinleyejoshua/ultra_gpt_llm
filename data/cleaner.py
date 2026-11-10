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


class DialogueFormatter:
    """
    Format converter for multi-turn structured dialogues:
    MLChat / ShareGPT / ChatML style:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    @classmethod
    def format_turn(cls, role: str, content: str) -> str:
        """Format a single dialogue turn."""
        clean_content = content.strip()
        return f"{cls.IM_START}{role}\n{clean_content}{cls.IM_END}"

    @classmethod
    def format_conversation(
        cls,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Convert list of message dicts ({'role': 'user'|'assistant'|'system', 'content': '...'})
        into a formatted single prompt string.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(cls.format_turn(role, content))

        result = "\n".join(formatted)
        if add_generation_prompt:
            result += f"\n{cls.IM_START}assistant\n"
        return result

    @classmethod
    def parse_conversation(cls, text: str) -> List[Dict[str, str]]:
        """Parse structured ChatML string back into a list of message dicts."""
        pattern = re.compile(
            rf"{re.escape(cls.IM_START)}(system|user|assistant)\n(.*?){re.escape(cls.IM_END)}",
            re.DOTALL,
        )
        matches = pattern.findall(text)
        return [{"role": role, "content": content.strip()} for role, content in matches]
