"""
Dedicated Web Scraper CLI for UltraGPT Dataset Ingestion.
Allows users to input any search queries, Wikipedia topics, or article URLs via:
1. Command-line arguments: python scrape.py --search "Quantum computing" "Neuroscience"
2. Interactive mode: python scrape.py --interactive
3. File input: python scrape.py --file topics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from data.scraper import WebScraper


def parse_args():
    parser = argparse.ArgumentParser(description="UltraGPT Dataset Web Scraper CLI")
    parser.add_argument(
        "--search",
        nargs="+",
        help="Search query keywords or Wikipedia article topics to scrape (e.g. --search 'Quantum computing' 'Large language model')",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="Direct web URLs to scrape (e.g. --urls https://en.wikipedia.org/wiki/Deep_learning)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive prompt where you can type search terms one by one",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a text file containing topics/URLs (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.txt",
        help="Destination dataset text file (default: data/dataset.txt)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing dataset file before appending new text",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)

    if args.reset and out_path.exists():
        out_path.unlink()
        print(f"Cleared existing file: {out_path}")

    scraper = WebScraper(output_path=out_path)
    sources = []

    # 1. From CLI search arguments
    if args.search:
        sources.extend(args.search)

    # 2. From CLI direct URLs
    if args.urls:
        sources.extend(args.urls)

    # 3. From text file
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                sources.extend(lines)
            print(f"Loaded {len(lines)} topics from {file_path}")
        else:
            print(f"Error: File {file_path} not found.")

    # 4. Interactive prompt mode
    if args.interactive or (not sources and not args.file and not args.search and not args.urls):
        print("\n" + "=" * 60)
        print("UltraGPT Interactive Dataset Scraper")
        print("Type any search keyword, topic, or URL to scrape.")
        print("Type 'done' or press Enter on an empty line when finished.")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("Enter search topic or URL > ").strip()
                if not user_input or user_input.lower() in ("done", "exit", "quit"):
                    break
                sources.append(user_input)
            except (KeyboardInterrupt, EOFError):
                break

    if not sources:
        print("No topics or URLs specified. Exiting.")
        return

    print(f"\nScraping {len(sources)} item(s) to '{out_path}'...\n")
    total_chars = scraper.scrape_batch(sources)
    print("\n" + "=" * 60)
    print(f"Scraping Complete! Ingested {total_chars:,} characters.")
    print(f"Dataset location: {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
