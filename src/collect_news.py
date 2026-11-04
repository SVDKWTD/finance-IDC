from pathlib import Path
from datetime import datetime, timezone
import time

import feedparser
import pandas as pd
import requests


MARKET_SEARCH_TERMS = {
    "nifty": [
        "Indian stock market NIFTY 50",
        "NSE BSE India market news",
        "RBI monetary policy India inflation interest rates",
        "NIFTY 50 earnings corporate news",
    ],
    "gold": [
        "gold prices market news",
        "gold futures prices interest rates dollar",
        "central bank gold purchases",
    ],
    "usd_inr": [
        "USD INR exchange rate India rupee news",
        "Indian rupee dollar RBI currency market",
        "USD INR currency market news",
    ],
    "brent": [
        "Brent crude oil prices market news",
        "crude oil supply OPEC geopolitical news",
        "Brent oil futures demand inventory",
    ],
}

RSS_URL = "https://news.google.com/rss/search"
REQUEST_PAUSE_SECONDS = 3

# Leave both as None for recent-news search.
START_DATE = None
END_DATE = None


def build_query(term):
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

    searches = [
        (market, term)
        for market, terms in MARKET_SEARCH_TERMS.items()
        for term in terms
    ]

    for index, (market, term) in enumerate(searches):
        query = build_query(term)
        print(f"\nSearching [{market}]: {query}")

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
                    "market": market,
                    "search_term": term,
                    "title": entry.get("title", ""),
                    # Preserve the feed timestamp; do not fill missing times.
                    "published_at": entry.get("published", ""),
                    "source": source_info.get("title", ""),
                    "url": entry.get("link", ""),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })

        except requests.RequestException as error:
            print(f"Request failed for '{term}': {error}")

        if index < len(searches) - 1:
            time.sleep(REQUEST_PAUSE_SECONDS)

    if not articles:
        print("\nNo articles collected. Check the connection and feed responses.")
        return

    df = pd.DataFrame(articles)
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df = df[df["url"] != ""]

    # Keep an article separately for each market if it matched multiple searches.
    df = df.drop_duplicates(subset=["url", "market"]).copy()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "news.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Searches run: {len(searches)}")
    print(f"Unique article-market rows retained: {len(df)}")
    print(f"Saved to: {output_file.resolve()}")

    print("\nRows by market:")
    print(df["market"].value_counts().to_string())

    dates = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    valid_dates = dates.dropna()

    if not valid_dates.empty:
        print("\nArticles by publication year:")
        print(valid_dates.dt.year.value_counts().sort_index().to_string())
    else:
        print("\nNo publication dates could be parsed.")

    print("\nSample articles:")
    print(df[["market", "title", "published_at", "source"]].head(10))


if __name__ == "__main__":
    collect_news()