"""
Dedicated Web Scraper CLI for UltraGPT Dataset Ingestion.
Supports:
1. Web article scraping: python scrape.py --search "Quantum computing" "Transformer architecture"
2. Reddit dialogue crawling: python scrape.py --subreddits AskScience ExplainLikeImFive
3. Chain-of-Thought reasoning synthesis: python scrape.py --thinking 20
4. Everyday greetings & pleasantries synthesis: python scrape.py --greetings 25
5. Transcript ingestion: python scrape.py --transcript interview.txt
6. Interactive mode: python scrape.py --interactive
7. Direct file input: python scrape.py --file topics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from data.scraper import WebScraper
from data.cleaner import TextCleaner
from data.dialogue_scraper import (
    DialogueParser,
    RedditDialogueScraper,
    ThinkingDialogueSynthesizer,
    GreetingPleasantrySynthesizer,
    DialogueDatasetBuilder,
)


def parse_args():
    parser = argparse.ArgumentParser(description="UltraGPT Dataset Web Scraper & Dialogue Ingestion CLI")
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
        "--subreddits",
        nargs="+",
        help="Subreddits to scrape authentic conversational Q&A dialogues from (e.g. --subreddits AskScience philosophy)",
    )
    parser.add_argument(
        "--thinking",
        type=int,
        default=None,
        help="Synthesize N Chain-of-Thought dialogues with <think>...</think> reasoning traces",
    )
    parser.add_argument(
        "--greetings",
        type=int,
        default=None,
        help="Synthesize N conversational dialogues covering everyday greetings, polite check-ins, and pleasantries",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Path to an unstructured transcript file with speaker labels to convert into ChatML",
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
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="Automatically remove duplicate documents/conversations (default: True)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_false",
        dest="dedup",
        help="Disable automatic deduplication pass",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)

    if args.reset and out_path.exists():
        out_path.unlink()
        print(f"Cleared existing file: {out_path}")

    dialogue_builder = DialogueDatasetBuilder(output_path=out_path)
    total_dialogue_chars = 0

    # 1. Scrape Reddit conversational Q&A dialogues
    if args.subreddits:
        reddit_scraper = RedditDialogueScraper()
        for sub in args.subreddits:
            print(f"Crawling multi-turn dialogues from r/{sub}...")
            dialogues = reddit_scraper.scrape_subreddit_dialogues(sub, limit=15)
            c = dialogue_builder.append_conversations(dialogues)
            total_dialogue_chars += c
            print(f"  [+] Ingested {len(dialogues)} dialogues from r/{sub} ({c:,} characters)")

    # 2. Synthesize Chain-of-Thought Thinking Dialogues (<think>...</think>)
    if args.thinking:
        print(f"Generating {args.thinking} Chain-of-Thought reasoning dialogues (<think>...</think>)...")
        thinking_convs = ThinkingDialogueSynthesizer.generate_thinking_dialogues(count=args.thinking)
        c = dialogue_builder.append_conversations(thinking_convs)
        total_dialogue_chars += c
        print(f"  [+] Ingested {len(thinking_convs)} thinking dialogues ({c:,} characters)")

    # 3. Synthesize Everyday Greetings & Pleasantries Dialogues
    if args.greetings:
        print(f"Generating {args.greetings} conversational greeting and pleasantry dialogues...")
        greeting_convs = GreetingPleasantrySynthesizer.generate_dialogues(count=args.greetings)
        c = dialogue_builder.append_conversations(greeting_convs)
        total_dialogue_chars += c
        print(f"  [+] Ingested {len(greeting_convs)} greeting & pleasantry dialogues ({c:,} characters)")

    # 4. Ingest and parse transcript file
    if args.transcript:
        t_path = Path(args.transcript)
        if t_path.exists():
            text = t_path.read_text(encoding="utf-8")
            turns = DialogueParser.parse_transcript(text)
            if turns:
                c = dialogue_builder.append_conversations([turns])
                total_dialogue_chars += c
                print(f"  [+] Converted transcript ({len(turns)} turns) to ChatML ({c:,} characters)")
        else:
            print(f"Transcript file {t_path} not found.")

    scraper = WebScraper(output_path=out_path)
    sources = []

    # 5. From CLI search arguments
    if args.search:
        sources.extend(args.search)

    # 6. From CLI direct URLs
    if args.urls:
        sources.extend(args.urls)

    # 7. From text file
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                sources.extend(lines)
            print(f"Loaded {len(lines)} topics from {file_path}")
        else:
            print(f"Error: File {file_path} not found.")

    # 8. Interactive prompt mode
    has_action = bool(
        sources
        or args.subreddits
        or args.thinking
        or args.greetings
        or args.transcript
        or args.file
    )
    if args.interactive or not has_action:
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

    if sources:
        print(f"\nScraping {len(sources)} article item(s) to '{out_path}'...\n")
        total_chars = scraper.scrape_batch(sources)
    else:
        total_chars = 0

    grand_total = total_chars + total_dialogue_chars
    print("\n" + "=" * 60)
    print(f"Scraping & Ingestion Complete! Total Ingested: {grand_total:,} characters.")
    print(f"Dataset location: {out_path.resolve()}")

    if args.dedup and out_path.exists():
        orig_docs, uniq_docs = TextCleaner.deduplicate_file(out_path)
        if orig_docs > uniq_docs:
            print(f"[Deduplication Pass] Removed {orig_docs - uniq_docs} duplicate sections. Unique sections: {uniq_docs}")
        else:
            print(f"[Dataset Verified] All {uniq_docs} sections in dataset are unique.")

    print(f"Final File Size: {out_path.stat().st_size:,} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()
