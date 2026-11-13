"""
Data Ingestion, Scraping, Cleaning, Tokenization, and Dataset Pipelines.
"""

from .cleaner import TextCleaner, DialogueFormatter
from .scraper import WebScraper
from .dialogue_scraper import (
    DialogueParser,
    RedditDialogueScraper,
    ThinkingDialogueSynthesizer,
    GreetingPleasantrySynthesizer,
    DialogueDatasetBuilder,
)
from .tokenizer import BPETokenizer, CharTokenizer
try:
    from .dataset import GPTDataset, create_dataloaders
except ImportError:
    GPTDataset = None  # type: ignore
    create_dataloaders = None  # type: ignore

__all__ = [
    "TextCleaner",
    "DialogueFormatter",
    "WebScraper",
    "DialogueParser",
    "RedditDialogueScraper",
    "ThinkingDialogueSynthesizer",
    "GreetingPleasantrySynthesizer",
    "DialogueDatasetBuilder",
    "BPETokenizer",
    "CharTokenizer",
    "GPTDataset",
    "create_dataloaders",
]
