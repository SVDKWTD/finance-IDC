
from pathlib import Path
from datetime import datetime, timezone
import time

import feedparser
import pandas as pd
import requests


SEARCH_TERMS = [
    "Indian stock market",
    "NIFTY 50",
    "Reserve Bank of India inflation",
]

RSS_URL = "https://news.google.com/rss/search"


def collect_news():
    articles = []

    headers = {
        "User-Agent": "FinanceNewsResearch/1.0"
    }

    for term in SEARCH_TERMS:
        print(f"Searching news for: {term}")

        try:
            response = requests.get(
                RSS_URL,
                params={
                    "q": term,
                    "hl": "en-IN",
                    "gl": "IN",
                    "ceid": "IN:en",
                },
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                articles.append({
                    "search_term": term,
                    "title": entry.get("title", ""),
                    "published_at": entry.get("published", ""),
                    "source": entry.get("source", {}).get(
                        "title", ""
                    ),
                    "url": entry.get("link", ""),
                    "collected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

        except requests.RequestException as error:
            print(f"Request failed for '{term}': {error}")

        time.sleep(1)

    df = pd.DataFrame(articles)

    if df.empty:
        print("No articles collected. Check the connection.")
        return

    # Remove duplicate article URLs.
    df = df.drop_duplicates(subset=["url"])

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "news.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nArticles collected: {len(df)}")
    print(f"Saved to: {output_file.resolve()}")
    print(df[["title", "published_at", "source"]].head())


if __name__ == "__main__":
    collect_news()
