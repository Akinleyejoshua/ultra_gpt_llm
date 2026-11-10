"""
Lightweight Web Scraper for Dataset Ingestion.
Crawls textual content from Wikipedia (exact titles or search keywords),
general web articles, and raw text URLs, runs through TextCleaner,
and saves/appends to dataset storage.
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
    HTTP Crawler & Text Ingestion Utility:
    - Search Wikipedia articles by any keyword, phrase, or topic.
    - Scrape general web URLs.
    - Automatic DOM noise filtering and text sanitization.
    - Append to dataset.txt with <|endoftext|> boundaries.
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

    def search_wikipedia(self, query: str) -> Optional[str]:
        """
        Search Wikipedia using OpenSearch API for a keyword/phrase.
        Returns the top matching article URL, or None if no match found.
        """
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
        try:
            resp = self.session.get(api_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                # data format: [query, [titles], [descriptions], [urls]]
                if len(data) >= 4 and len(data[3]) > 0:
                    matched_url = data[3][0]
                    matched_title = data[1][0]
                    print(f"  -> Search '{query}' matched Wikipedia article: '{matched_title}' ({matched_url})")
                    return matched_url
        except Exception as e:
            logger.warning(f"Wikipedia search query failed for '{query}': {e}")
        return None

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

    def scrape_wikipedia(self, title_or_query: str) -> str:
        """
        Scrape Wikipedia by title, query phrase, or direct URL.
        If not a URL, it queries Wikipedia search to find the most relevant article.
        """
        if title_or_query.startswith("http://") or title_or_query.startswith("https://"):
            return self.scrape_url(title_or_query)

        # First try search API for best match
        matched_url = self.search_wikipedia(title_or_query)
        if matched_url:
            return self.scrape_url(matched_url)

        # Fallback: direct title encoding
        encoded_title = urllib.parse.quote(title_or_query.replace(" ", "_"))
        direct_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
        return self.scrape_url(direct_url)

    def append_to_dataset(self, text: str) -> int:
        """Append cleaned text to dataset file. Returns number of characters written."""
        if not text:
            return 0
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(text + "\n\n<|endoftext|>\n\n")
        return len(text)

    def scrape_batch(self, sources: List[str]) -> int:
        """
        Scrape a list of search queries, Wikipedia topics, or direct URLs,
        and append all cleaned text to the dataset file.
        Returns total characters appended.
        """
        total_chars = 0
        for src in sources:
            src_clean = src.strip()
            if not src_clean:
                continue
            print(f"Ingesting: '{src_clean}'...")
            text = self.scrape_wikipedia(src_clean)
            chars = self.append_to_dataset(text)
            total_chars += chars
            print(f"  [+] Wrote {chars:,} characters from '{src_clean}'")
        return total_chars
