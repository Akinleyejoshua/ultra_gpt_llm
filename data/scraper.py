"""
Lightweight Web Scraper for Dataset Ingestion.
Crawls textual content from Wikipedia, web articles, and raw text URLs,
runs through TextCleaner, and saves/appends to dataset storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
import urllib.parse

import requests
from bs4 import BeautifulSoup

from .cleaner import TextCleaner

logger = logging.getLogger(__name__)


class WebScraper:
    """
    HTTP Crawler & Text Ingestion Utility.
    Pulls clean paragraphs from Wikipedia pages or general URLs.
    """

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 UltraGPT/1.0"
    )

    def __init__(self, output_path: str | Path = "data/dataset.txt", timeout: int = 15):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.DEFAULT_USER_AGENT})

    def scrape_url(self, url: str) -> str:
        """Fetch raw HTML, extract body text, and clean it."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, navigation, footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Target Wikipedia main content or generic article body
            content_div = soup.find("div", {"id": "mw-content-text"}) or soup.find("article") or soup.body
            if not content_div:
                return ""

            paragraphs = content_div.find_all("p")
            raw_text = "\n\n".join(p.get_text() for p in paragraphs if p.get_text().strip())
            cleaned_text = TextCleaner.clean(raw_text)
            return cleaned_text
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""

    def scrape_wikipedia(self, title_or_url: str) -> str:
        """Convenience method for scraping an English Wikipedia article."""
        if title_or_url.startswith("http"):
            url = title_or_url
        else:
            encoded_title = urllib.parse.quote(title_or_url.replace(" ", "_"))
            url = f"https://en.wikipedia.org/wiki/{encoded_title}"
        return self.scrape_url(url)

    def append_to_dataset(self, text: str) -> int:
        """Append cleaned text to dataset file. Returns number of characters written."""
        if not text:
            return 0
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(text + "\n\n<|endoftext|>\n\n")
        return len(text)

    def scrape_batch(self, sources: List[str]) -> int:
        """
        Scrape a list of Wikipedia topics or URLs and append all to dataset file.
        Returns total characters appended.
        """
        total_chars = 0
        for src in sources:
            text = self.scrape_wikipedia(src) if not src.startswith("http") else self.scrape_url(src)
            chars = self.append_to_dataset(text)
            total_chars += chars
            logger.info(f"Ingested '{src}': {chars} characters.")
        return total_chars
