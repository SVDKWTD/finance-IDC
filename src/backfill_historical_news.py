
from calendar import monthrange
from datetime import date, datetime, timezone
from pathlib import Path
import time

import feedparser
import pandas as pd
import requests


RSS_URL = "https://news.google.com/rss/search"

TERMS = [
    "India stock market",
    "Sensex",
    "NIFTY 50",
    "Indian economy",
    "RBI India",
    "Indian stock market earnings",
    "India inflation interest rates",
    "Reliance Industries earnings",
    "Tata Consultancy Services earnings",
    "NSE BSE market news",
]
START_YEAR = 2021
END_YEAR = 2025
REQUEST_PAUSE_SECONDS = 3

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "raw"

ARTICLES_FILE = OUTPUT_DIR / "historical_news_backfill_2021_2025.csv"
COVERAGE_FILE = OUTPUT_DIR / "monthly_coverage_backfill_2021_2025.csv"


def month_ranges():
    """Generate start and exclusive end dates for every month."""
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = date(year, month, 1)

            if month == 12:
                end = date(year + 1, 1, 1)
            else:
                end = date(year, month + 1, 1)

            yield start, end


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.headers.update(headers)

    all_articles = {}
    coverage_rows = []

    months = list(month_ranges())

    for month_index, (start, end) in enumerate(months, start=1):
        month_key = start.strftime("%Y-%m")
        print(f"\n[{month_index}/{len(months)}] Auditing {month_key}")

        received = 0
        errors = 0

        for term_index, term in enumerate(TERMS):
            query = (
                f"{term} after:{start.isoformat()} "
                f"before:{end.isoformat()}"
            )

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

                    # Count only articles whose actual publication
                    # date falls within the requested month.
                    if pd.isna(timestamp):
                        continue

                    actual_date = timestamp.date()

                    if not (start <= actual_date < end):
                        continue

                    # Merge duplicates across queries by URL.
                    if url not in all_articles:
                        source_info = entry.get("source", {})
                        if not isinstance(source_info, dict):
                            source_info = {}

                        all_articles[url] = {
                            "title": title,
                            "published_at": timestamp.isoformat(),
                            "publication_date": actual_date.isoformat(),
                            "source": source_info.get("title", ""),
                            "url": url,
                            "matched_terms": term,
                            "requested_month": month_key,
                            "collected_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    else:
                        old_terms = all_articles[url]["matched_terms"]
                        if term not in old_terms.split(" | "):
                            all_articles[url]["matched_terms"] = (
                                old_terms + " | " + term
                            )

            except (requests.RequestException, ValueError) as error:
                errors += 1
                print(f"  Query failed for '{term}': {error}")

            if term_index < len(TERMS) - 1:
                time.sleep(REQUEST_PAUSE_SECONDS)

        # Count unique, date-validated articles for this month.
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

        # Save checkpoints after every month.
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
    yearly = coverage.groupby("year")["unique_valid_articles"].sum()
    print(yearly.to_string())

    print("\n========== OUTPUT FILES ==========")
    print(f"Articles: {ARTICLES_FILE}")
    print(f"Monthly coverage: {COVERAGE_FILE}")


if __name__ == "__main__":
    main()
