
from pathlib import Path
from datetime import datetime, timezone
import time

import feedparser
import pandas as pd
import requests


SEARCH_TERMS = [
    "Indian stock market",
    "NIFTY 50",
    "Sensex India",
    "Reserve Bank of India monetary policy",
    "India inflation interest rates",
    "Reliance Industries earnings",
    "Tata Consultancy Services earnings",
    "NSE BSE India market news",
]

RSS_URL = "https://news.google.com/rss/search"
REQUEST_PAUSE_SECONDS = 3

# Leave both as None for the normal recent-news search.
# For a historical test, set dates in YYYY-MM-DD format.
# Example:
# START_DATE = "2025-10-01"
# END_DATE = "2025-10-31"
START_DATE = None
END_DATE = None


def build_query(term):
    """Add optional date filters to a Google News search."""
    query = term

    if START_DATE:
        start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        query += f" after:{start.isoformat()}"

    if END_DATE:
        end = datetime.strptime(END_DATE, "%Y-%m-%d").date()
        query += f" before:{end.isoformat()}"

    return query


def collect_news():
    articles = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FinanceNewsResearch/1.0)"
    }

    for index, term in enumerate(SEARCH_TERMS):
        query = build_query(term)
        print(f"\nSearching: {query}")

        try:
            response = requests.get(
                RSS_URL,
                params={
                    "q": query,
                    "hl": "en-IN",
                    "gl": "IN",
                    "ceid": "IN:en",
                },
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            print(f"Feed entries returned: {len(feed.entries)}")

            if feed.bozo:
                print(f"Feed parsing warning: {feed.bozo_exception}")

            for entry in feed.entries:
                source_info = entry.get("source", {})
                if not isinstance(source_info, dict):
                    source_info = {}

                articles.append({
                    "search_term": term,
                    "title": entry.get("title", ""),
                    "published_at": entry.get("published", ""),
                    "source": source_info.get("title", ""),
                    "url": entry.get("link", ""),
                    "collected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

        except requests.RequestException as error:
            print(f"Request failed for '{term}': {error}")

        if index < len(SEARCH_TERMS) - 1:
            time.sleep(REQUEST_PAUSE_SECONDS)

    if not articles:
        print("\nNo articles collected. Check the connection and feed responses.")
        return

    df = pd.DataFrame(articles)

    # Remove missing URLs and duplicate URLs.
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df = df[df["url"] != ""]
    df = df.drop_duplicates(subset=["url"]).copy()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "news.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Search terms used: {len(SEARCH_TERMS)}")
    print(f"Total entries received: {len(articles)}")
    print(f"Unique articles retained: {len(df)}")
    print(f"Duplicate or empty-URL entries removed: {len(articles) - len(df)}")
    print(f"Saved to: {output_file.resolve()}")
    print("\nSample articles:")
    print(df[["title", "published_at", "source", "search_term"]].head(10))

    # Show the publication-date distribution by year.
    dates = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    valid_dates = dates.dropna()

    if not valid_dates.empty:
        print("\nArticles by publication year:")
        print(valid_dates.dt.year.value_counts().sort_index().to_string())
    else:
        print("\nNo publication dates could be parsed.")


if __name__ == "__main__":
    collect_news()
