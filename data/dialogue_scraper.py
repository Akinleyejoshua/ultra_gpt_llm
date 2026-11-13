"""
Comprehensive Dialogue Ingestion & Scraping Pipeline for Conversational & Thinking Models.
Supports:
1. Universal Dialogue Parser: extracts conversational turns from raw transcripts / text.
2. Reddit Multi-Turn Conversation Scraper: pulls authentic human Q&A / discussions via public JSON.
3. Interview & Transcript Scraper: crawls interview dialogues from web pages.
4. Chain-of-Thought (CoT) Thinking Dialogue Synthesizer: builds reasoning dialogues with <think>...</think> tags.
5. Greeting & Pleasantry Synthesizer: builds everyday pleasantries, polite check-ins, and natural conversational exchanges.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.parse

import requests
from bs4 import BeautifulSoup

from .cleaner import TextCleaner, DialogueFormatter

logger = logging.getLogger(__name__)


class DialogueParser:
    """
    Parses unstructured text containing dialogue or transcripts into structured ChatML turns:
    Recognizes patterns such as:
      - Human: ... / AI: ...
      - User: ... / Assistant: ...
      - Interviewer: ... / Guest: ...
      - Q: ... / A: ...
      - Person 1: ... / Person 2: ...
    """

    SPEAKER_PATTERNS = [
        # Standard AI chat formats
        (re.compile(r"^(?:User|Human|Prompt|Client|Question|Q):\s*(.+)$", re.IGNORECASE), "user"),
        (re.compile(r"^(?:Assistant|AI|Bot|Response|Answer|A):\s*(.+)$", re.IGNORECASE), "assistant"),
        # Transcript / Interview formats
        (re.compile(r"^(?:Interviewer|Host|Moderator|Speaker 1|Person A):\s*(.+)$", re.IGNORECASE), "user"),
        (re.compile(r"^(?:Interviewee|Guest|Expert|Speaker 2|Person B):\s*(.+)$", re.IGNORECASE), "assistant"),
    ]

    @classmethod
    def parse_transcript(cls, text: str) -> List[Dict[str, str]]:
        """Parse multi-speaker transcript lines into structured role/content messages."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        turns: List[Dict[str, str]] = []
        current_role: Optional[str] = None
        current_content: List[str] = []

        for line in lines:
            matched = False
            for pattern, role in cls.SPEAKER_PATTERNS:
                match = pattern.match(line)
                if match:
                    # Flush previous turn
                    if current_role and current_content:
                        turns.append({"role": current_role, "content": " ".join(current_content).strip()})
                    current_role = role
                    current_content = [match.group(1).strip()]
                    matched = True
                    break

            if not matched:
                if current_role is not None:
                    current_content.append(line)

        # Flush final turn
        if current_role and current_content:
            turns.append({"role": current_role, "content": " ".join(current_content).strip()})

        return turns


class RedditDialogueScraper:
    """
    Crawls multi-turn Q&A and conversation threads from educational/discussion subreddits
    using public, unauthenticated read-only JSON endpoints.
    """

    DEFAULT_SUBREDDITS = [
        "AskScience",
        "ExplainLikeImFive",
        "philosophy",
        "LearnMachineLearning",
    ]

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UltraGPT-Dialogue-Collector/1.0"

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.timeout = timeout

    def scrape_subreddit_dialogues(self, subreddit: str, limit: int = 15) -> List[List[Dict[str, str]]]:
        """Fetch hot posts and top comments from a subreddit and format into dialogues."""
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        dialogues: List[List[Dict[str, str]]] = []

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"Reddit API returned status {resp.status_code} for r/{subreddit}")
                return []

            data = resp.json()
            posts = data.get("data", {}).get("children", [])

            for p in posts:
                post_data = p.get("data", {})
                title = post_data.get("title", "").strip()
                selftext = post_data.get("selftext", "").strip()
                permalink = post_data.get("permalink", "")

                if not title or post_data.get("over_18", False) or post_data.get("is_video", False):
                    continue

                # Combine title and selftext as user question
                user_msg = f"{title}\n{selftext}".strip() if selftext else title
                user_msg = TextCleaner.clean(user_msg, remove_urls=True)

                # Fetch top comments for this thread
                comments = self._fetch_thread_comments(permalink)
                if not comments:
                    continue

                thread_turns: List[Dict[str, str]] = [
                    {"role": "user", "content": user_msg}
                ]
                for c in comments:
                    clean_comment = TextCleaner.clean(c, remove_urls=True)
                    if len(clean_comment) > 30:  # Filter low-effort short replies
                        thread_turns.append({"role": "assistant", "content": clean_comment})
                        break  # Capture best top answer

                if len(thread_turns) >= 2:
                    dialogues.append(thread_turns)

        except Exception as e:
            logger.warning(f"Error scraping Reddit r/{subreddit}: {e}")

        return dialogues

    def _fetch_thread_comments(self, permalink: str, max_comments: int = 2) -> List[str]:
        """Fetch top-level comments for a thread permalink."""
        url = f"https://www.reddit.com{permalink}.json?limit=3"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            if len(data) < 2:
                return []
            comment_nodes = data[1].get("data", {}).get("children", [])
            comments = []
            for c in comment_nodes:
                body = c.get("data", {}).get("body", "").strip()
                if body and body not in ("[deleted]", "[removed]"):
                    comments.append(body)
                if len(comments) >= max_comments:
                    break
            return comments
        except Exception:
            return []


class GreetingPleasantrySynthesizer:
    """
    Synthesizes natural, high-quality conversational dialogues focusing on:
    - Everyday greetings (Hello, Hi, Good morning/afternoon/evening, Hey)
    - Well-being inquiries and polite check-ins (How are you? How is your day going?)
    - Expressions of gratitude and polite closings (Thank you, You're welcome, Have a great day)
    - Friendly small talk and identity introductions (Who are you? Nice to meet you!)
    - Multi-turn pleasantry dialogues with natural turn-taking
    """

    DIALOGUE_BANK: List[List[Dict[str, str]]] = [
        # 1. Warm standard greeting
        [
            {"role": "user", "content": "Hello! How are you doing today?"},
            {
                "role": "assistant",
                "content": "Hello! I'm doing very well, thank you for asking. How are you doing today? How can I assist you?",
                "reasoning": "The user is greeting me politely and asking about my state. I should acknowledge the greeting with warmth, state that I am well, and ask how they are doing while offering my assistance."
            },
        ],
        # 2. Casual greeting
        [
            {"role": "user", "content": "Hey there! What's up?"},
            {"role": "assistant", "content": "Hey! Everything is going great on my end. How are things with you today? What are you working on?"},
        ],
        # 3. Morning greeting
        [
            {"role": "user", "content": "Good morning!"},
            {
                "role": "assistant",
                "content": "Good morning! I hope you have a productive and wonderful day ahead. What can I help you with today?",
                "reasoning": "The user provided a polite morning greeting. I should respond in kind with positive energy and ask how I can help."
            },
        ],
        # 4. Afternoon greeting
        [
            {"role": "user", "content": "Good afternoon! How is your day going?"},
            {"role": "assistant", "content": "Good afternoon! My day has been great so far, thank you. How is your afternoon going? Is there anything I can help you with?"},
        ],
        # 5. Evening greeting
        [
            {"role": "user", "content": "Good evening!"},
            {"role": "assistant", "content": "Good evening! I hope you had a pleasant day. How can I assist you tonight?"},
        ],
        # 6. Multi-turn greeting & check-in
        [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! It's great to see you. How are you doing today?"},
            {"role": "user", "content": "I'm doing well, thank you! How about you?"},
            {
                "role": "assistant",
                "content": "I'm doing wonderful, thank you for asking! I'm excited to help. What's on your mind today?",
                "reasoning": "The user confirmed they are doing well and politely reciprocated by asking how I am. I should express gratitude and invite their questions."
            },
            {"role": "user", "content": "Just wanted to say hi and check in."},
            {"role": "assistant", "content": "That's very kind of you! It's always a pleasure chatting with you. Feel free to reach out anytime if you need anything at all. Have a fantastic day!"},
        ],
        # 7. Gratitude exchange
        [
            {"role": "user", "content": "Thank you so much for your help!"},
            {
                "role": "assistant",
                "content": "You are very welcome! I'm always happy to help. If you have any other questions, feel free to ask anytime!",
                "reasoning": "The user is expressing gratitude for previous help. I should accept the thanks warmly and let them know I remain available for any future needs."
            },
        ],
        # 8. Short thanks
        [
            {"role": "user", "content": "Thanks a lot, that was very clear."},
            {"role": "assistant", "content": "Anytime! I'm glad that was helpful. Let me know if there's anything else you'd like to explore."},
        ],
        # 9. Polite farewell
        [
            {"role": "user", "content": "Goodbye, have a great day!"},
            {
                "role": "assistant",
                "content": "Goodbye! Thank you, and I hope you have a wonderful and enjoyable day as well. Take care!",
                "reasoning": "The user is concluding the conversation with a farewell. I should wish them well and say goodbye warmly."
            },
        ],
        # 10. Night farewell
        [
            {"role": "user", "content": "I'm heading to sleep now. Goodnight!"},
            {"role": "assistant", "content": "Goodnight! Sleep well and have a restful night. See you next time!"},
        ],
        # 11. Introduction & identity
        [
            {"role": "user", "content": "Who are you and what do you do?"},
            {
                "role": "assistant",
                "content": "I am UltraGPT, an AI assistant built and trained from scratch with PyTorch. I can help answer questions, write and review code, explain complex ideas, or simply enjoy a friendly conversation. How can I help you today?",
                "reasoning": "The user is asking about my identity and capabilities. I should introduce myself clearly as UltraGPT, outline my abilities, and invite their request."
            },
        ],
        # 12. Nice to meet you
        [
            {"role": "user", "content": "Nice to meet you, UltraGPT!"},
            {"role": "assistant", "content": "It is wonderful to meet you too! I'm looking forward to our conversations. How can I be of service to you today?"},
        ],
        # 13. Empathetic check-in
        [
            {"role": "user", "content": "I've had a really busy and exhausting day."},
            {
                "role": "assistant",
                "content": "I understand that feeling completely. Make sure to take some time to unwind and rest this evening. If there's anything I can do to make things easier or take off your plate, just say the word!",
                "reasoning": "The user shares that they have had an exhausting day. I should respond with empathy, encourage rest, and offer support without being pushy."
            },
        ],
        # 14. Casual weekend inquiry
        [
            {"role": "user", "content": "Happy Friday! Any plans for the weekend?"},
            {"role": "assistant", "content": "Happy Friday! As an AI, I'll be right here ready to help, but I hope you have an incredible and relaxing weekend planned! Any fun activities lined up?"},
        ],
        # 15. Quick pleasantry
        [
            {"role": "user", "content": "Hope you're having a good day!"},
            {"role": "assistant", "content": "Thank you so much! That is very thoughtful of you. I hope you're having an awesome day too. How can I help you today?"},
        ],
        # 16. Polite affirmation
        [
            {"role": "user", "content": "Sounds great, thank you!"},
            {"role": "assistant", "content": "You're most welcome! Enjoy the rest of your day, and feel free to reach out whenever you need assistance."},
        ],
    ]

    @classmethod
    def generate_dialogues(cls, count: int = 20) -> List[List[Dict[str, str]]]:
        """Generate N diverse conversational greeting and pleasantry dialogues."""
        dialogues = []
        bank_len = len(cls.DIALOGUE_BANK)
        for i in range(count):
            base_conv = cls.DIALOGUE_BANK[i % bank_len]
            # Deep copy turns
            conv_copy = []
            for turn in base_conv:
                conv_copy.append(dict(turn))
            dialogues.append(conv_copy)
        return dialogues


class ThinkingDialogueSynthesizer:
    """
    Generates rich Chain-of-Thought (CoT) dialogues equipped with <think>...</think> tags.
    Teaches the model to execute reasoning steps before formulating user-facing answers.
    """

    REASONING_TEMPLATES = [
        {
            "user": "What happens to the gradient when using a sigmoid activation function in deep networks?",
            "think": "The sigmoid function is defined as sigma(z) = 1 / (1 + exp(-z)). Its derivative is sigma'(z) = sigma(z) * (1 - sigma(z)). The maximum value of this derivative occurs at z = 0, where sigma(0) = 0.5, giving a maximum derivative of 0.25. During backpropagation across L layers, gradients are multiplied repeatedly by terms <= 0.25. Therefore, (0.25)^L decays exponentially toward zero as L grows, causing vanishing gradients.",
            "answer": "Using the sigmoid activation function in deep networks causes the vanishing gradient problem. Because the maximum derivative of the sigmoid function is only 0.25, repeatedly multiplying gradients across multiple layers causes them to diminish exponentially toward zero, preventing earlier layers from updating effectively.",
        },
        {
            "user": "Why do Transformers use scaled dot-product attention instead of regular dot-product attention?",
            "think": "In dot-product attention, we compute Q * K^T. For queries and keys with dimension d_k, assuming independent components with zero mean and unit variance, the dot product has zero mean and variance d_k. For large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by sqrt(d_k) normalizes the variance back to 1, stabilizing softmax gradients.",
            "answer": "Transformers scale the dot product by 1 / sqrt(d_k) because for large head dimensions, the dot products grow large in magnitude. This pushes the softmax function into saturated regions where gradients become vanishingly small. Scaling normalizes the variance to 1, ensuring stable gradient flow during backpropagation.",
        },
        {
            "user": "How does FlashAttention achieve faster training without changing the mathematical output?",
            "think": "Standard attention materializes the full N x N attention matrix in High-Bandwidth Memory (HBM), requiring repeated read/write memory operations. FlashAttention computes the exact same softmax and attention values by tiling input blocks into fast on-chip GPU SRAM memory and computing softmax online incrementally. It avoids writing the N x N intermediate matrix to GPU global memory, drastically reducing memory bandwidth bottlenecks.",
            "answer": "FlashAttention produces mathematically identical results to standard attention. It achieves higher speed by tiling Query, Key, and Value matrices into high-speed on-chip GPU SRAM, computing the softmax incrementally online. This eliminates the need to read and write the large N x N attention matrix to GPU High-Bandwidth Memory (HBM), overcoming memory bandwidth bottlenecks.",
        },
        {
            "user": "Explain the difference between Pre-LayerNorm and Post-LayerNorm architectures in Transformers.",
            "think": "In Post-LN (original transformer), LayerNorm is applied after the residual addition: x = LN(x + Sublayer(x)). This leads to vanishing gradients near the input layers in very deep models, requiring careful warmup schedules. In Pre-LN (GPT-2 standard), LayerNorm is applied before each sublayer: x = x + Sublayer(LN(x)). The residual stream remains an unobstructed highway for gradient flow, allowing stable training without steep learning rate warmup dependencies.",
            "answer": "In Post-LayerNorm, normalization is applied after the residual connection (x = LN(x + F(x))), which can lead to gradient instability in deep models. In Pre-LayerNorm (used by GPT-2 and modern LLMs), normalization is applied to inputs before the attention and MLP sublayers (x = x + F(LN(x))). This keeps the residual stream clear for direct gradient propagation, making training much more stable.",
        },
        {
            "user": "If I double the context length of a Transformer, how does the computational complexity scale?",
            "think": "The self-attention mechanism computes pairwise interactions between all tokens in sequence length N, which requires computing an N x N matrix (Q @ K^T). This has O(N^2) time and memory complexity with respect to sequence length. If N doubles to 2N, (2N)^2 = 4N^2. Therefore, computational cost and memory for attention quadruples.",
            "answer": "Because self-attention computes dot-products between all pairs of tokens, its time and memory complexity scales quadratically (O(N^2)). Doubling the context length quadruples (4x) the attention computation and memory requirements.",
        },
    ]

    @classmethod
    def generate_thinking_dialogues(cls, count: int = 10) -> List[List[Dict[str, str]]]:
        """Returns structured conversations equipped with <think>...</think> reasoning traces."""
        dialogues = []
        for i in range(count):
            item = cls.REASONING_TEMPLATES[i % len(cls.REASONING_TEMPLATES)]
            dialogues.append([
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["answer"], "reasoning": item["think"]},
            ])
        return dialogues


class DialogueDatasetBuilder:
    """
    Compiles multi-source dialogues into the primary training dataset:
    - Normalizes multi-turn conversations.
    - Formats into ChatML (<|im_start|>role...<|im_end|>) with thinking tokens.
    - Appends cleanly to data/dataset.txt with document boundaries (<|endoftext|>).
    """

    def __init__(self, output_path: str | Path = "data/dataset.txt"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append_conversations(self, conversations: List[List[Dict[str, str]]]) -> int:
        """Format and append a batch of conversations to the dataset."""
        total_chars = 0
        with open(self.output_path, "a", encoding="utf-8") as f:
            for conv in conversations:
                # Format into ChatML string
                formatted_chat = DialogueFormatter.format_conversation(conv)
                # Delimit conversation
                doc = f"{formatted_chat}\n\n<|endoftext|>\n\n"
                f.write(doc)
                total_chars += len(doc)
        return total_chars
