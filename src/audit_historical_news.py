from datetime import date, datetime, timezone
from pathlib import Path
import time

import feedparser
import pandas as pd
import requests


RSS_URL = "https://news.google.com/rss/search"

# Search terms are grouped by the market they relate to.
MARKET_TERMS = {
    "nifty": [
        "India stock market",
        "NIFTY 50",
        "Sensex",
    ],
    "gold": [
        "India gold prices",
        "MCX gold price India",
    ],
    "usd_inr": [
        "USD INR exchange rate",
        "Indian rupee dollar exchange rate",
    ],
    "brent": [
        "Brent crude oil price",
        "Brent crude India",
    ],
}

START_YEAR = 2021
END_YEAR = 2025
REQUEST_PAUSE_SECONDS = 3

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "raw"

ARTICLES_FILE = OUTPUT_DIR / "historical_news_2021_2025.csv"
COVERAGE_FILE = OUTPUT_DIR / "monthly_coverage_2021_2025.csv"


def month_ranges():
    """Generate start and exclusive end dates for every month."""
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = date(year, month, 1)
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            yield start, end


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Keep separate records when the same URL is relevant to different markets.
    all_articles = {}
    coverage_rows = []
    terms = [
        (market, term)
        for market, market_terms in MARKET_TERMS.items()
        for term in market_terms
    ]

    months = list(month_ranges())

    for month_index, (start, end) in enumerate(months, start=1):
        month_key = start.strftime("%Y-%m")
        print(f"\n[{month_index}/{len(months)}] Collecting {month_key}")

        received = 0
        errors = 0

        for term_index, (market, term) in enumerate(terms):
            query = f"{term} after:{start.isoformat()} before:{end.isoformat()}"

            try:
                response = session.get(
                    RSS_URL,
                    params={
                        "q": query,
                        "hl": "en-IN",
                        "gl": "IN",
                        "ceid": "IN:en",
                    },
                    timeout=30,
                )
                response.raise_for_status()

                feed = feedparser.parse(response.content)
                received += len(feed.entries)

                for entry in feed.entries:
                    url = str(entry.get("link", "")).strip()
                    title = str(entry.get("title", "")).strip()
                    published = entry.get("published", "")

                    if not url or not title:
                        continue

                    timestamp = pd.to_datetime(
                        published, errors="coerce", utc=True
                    )
                    if pd.isna(timestamp):
                        continue

                    actual_date = timestamp.date()
                    if not (start <= actual_date < end):
                        continue

                    key = (url, market)

                    if key not in all_articles:
                        source_info = entry.get("source", {})
                        if not isinstance(source_info, dict):
                            source_info = {}

                        all_articles[key] = {
                            "title": title,
                            "published_at": timestamp.isoformat(),
                            "publication_date": actual_date.isoformat(),
                            "source": source_info.get("title", ""),
                            "url": url,
                            "market": market,
                            "matched_terms": term,
                            "requested_month": month_key,
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        article = all_articles[key]
                        matched = article["matched_terms"].split(" | ")
                        if term not in matched:
                            article["matched_terms"] += f" | {term}"

            except (requests.RequestException, ValueError) as error:
                errors += 1
                print(f"  Query failed for {market} / '{term}': {error}")

            if term_index < len(terms) - 1:
                time.sleep(REQUEST_PAUSE_SECONDS)

        month_articles = sum(
            article["requested_month"] == month_key
            for article in all_articles.values()
        )

        coverage_rows.append({
            "month": month_key,
            "entries_received": received,
            "unique_valid_articles": month_articles,
            "query_errors": errors,
        })

        print(f"  Feed entries received: {received}")
        print(f"  Unique valid articles: {month_articles}")
        print(f"  Query errors: {errors}")

        # Save after each month so an interrupted run can be inspected.
        pd.DataFrame(all_articles.values()).to_csv(
            ARTICLES_FILE, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(coverage_rows).to_csv(
            COVERAGE_FILE, index=False, encoding="utf-8-sig"
        )

    coverage = pd.DataFrame(coverage_rows)
    coverage["year"] = coverage["month"].str[:4]

    print("\n========== MONTHLY COVERAGE ==========")
    print(coverage[["month", "unique_valid_articles"]].to_string(index=False))

    print("\n========== YEARLY TOTALS ==========")
    print(coverage.groupby("year")["unique_valid_articles"].sum().to_string())

    print("\n========== ARTICLES BY MARKET ==========")
    articles_df = pd.DataFrame(all_articles.values())
    if not articles_df.empty:
        print(articles_df.groupby("market").size().to_string())

    print("\n========== OUTPUT FILES ==========")
    print(f"Articles: {ARTICLES_FILE}")
    print(f"Monthly coverage: {COVERAGE_FILE}")


if __name__ == "__main__":
    main()