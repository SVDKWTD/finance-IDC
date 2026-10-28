from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import time

import feedparser
import pandas as pd
import requests

SEARCH_TERMS = [
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

RSS_URL = "https://news.google.com/rss/search"
REQUEST_PAUSE_SECONDS = 3
LOOKBACK_DAYS = 7
COLLECTION_START = date(2026, 1, 1)

HISTORICAL_FILE = Path("data/raw/historical_news_2021_2025.csv")
OUTPUT_FILE = Path("data/raw/recent_news_2026.csv")
REPORT_FILE = Path("data/raw/monthly_coverage_2026.csv")

COLUMNS = [
    "title",
    "published_at",
    "publication_date",
    "source",
    "url",
    "matched_terms",
    "requested_month",
    "collected_at",
]


def load_urls(path):
    if not path.exists():
        return set()
    existing = pd.read_csv(path)
    if "url" not in existing.columns:
        return set()
    return set(existing["url"].dropna().astype(str).str.strip())


def prepare_saved_frame(path):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.read_csv(path)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[COLUMNS].copy()
    frame["url"] = frame["url"].fillna("").astype(str).str.strip()
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["published_at"] = pd.to_datetime(
        frame["published_at"], errors="coerce", utc=True
    )
    frame = frame[
        (frame["url"] != "")
        & (frame["title"] != "")
        & frame["published_at"].notna()
    ].copy()
    return frame


def collect_recent_news():
    today = date.today()
    if today < COLLECTION_START:
        print("The current date is before the collection period.")
        return

    HISTORICAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    historical_urls = load_urls(HISTORICAL_FILE)
    previous = prepare_saved_frame(OUTPUT_FILE)

    # Search a rolling overlap window. The saved collection retains all earlier
    # articles; the overlap allows late-indexed articles to be discovered.
    window_start = max(
        COLLECTION_START,
        today - timedelta(days=LOOKBACK_DAYS)
    )
    window_end_exclusive = today + timedelta(days=1)
    after_date = window_start - timedelta(days=1)

    print(f"Incremental collection window: {window_start} through {today}")
    print(f"Previously saved 2026 articles: {len(previous):,}")
    print(f"Historical URLs available for deduplication: {len(historical_urls):,}")

    collected = []
    headers = {"User-Agent": "FinanceNewsResearch/1.0"}

    with requests.Session() as session:
        session.headers.update(headers)

        for term in SEARCH_TERMS:
            query = (
                f"{term} after:{after_date.isoformat()} "
                f"before:{window_end_exclusive.isoformat()}"
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
                valid_count = 0

                for entry in feed.entries:
                    title = entry.get("title", "").strip()
                    url = entry.get("link", "").strip()
                    published_text = entry.get("published", "")

                    if not title or not url or not published_text:
                        continue

                    published_at = pd.to_datetime(
                        published_text, errors="coerce", utc=True
                    )
                    if pd.isna(published_at):
                        continue

                    if not (
                        pd.Timestamp(window_start, tz="UTC")
                        <= published_at
                        < pd.Timestamp(window_end_exclusive, tz="UTC")
                    ):
                        continue

                    collected.append({
                        "title": title,
                        "published_at": published_at.isoformat(),
                        "publication_date": published_at.strftime("%Y-%m-%d"),
                        "source": entry.get("source", {}).get("title", ""),
                        "url": url,
                        "matched_terms": term,
                        "requested_month": published_at.strftime("%Y-%m"),
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    })
                    valid_count += 1

                print(f"  {term}: {valid_count} matching entries")

            except requests.RequestException as error:
                print(f"  Request failed for '{term}': {error}")

            time.sleep(REQUEST_PAUSE_SECONDS)

    new_frame = pd.DataFrame(collected, columns=COLUMNS)
    combined = pd.concat([previous, new_frame], ignore_index=True)

    if combined.empty:
        print("No saved or newly collected articles to write.")
        return

    combined["url"] = combined["url"].fillna("").astype(str).str.strip()
    combined["title"] = combined["title"].fillna("").astype(str).str.strip()
    combined["published_at"] = pd.to_datetime(
        combined["published_at"], errors="coerce", utc=True
    )
    combined = combined[
        (combined["url"] != "")
        & (combined["title"] != "")
        & combined["published_at"].notna()
        & (combined["published_at"] >= pd.Timestamp(COLLECTION_START, tz="UTC"))
        & (combined["published_at"] < pd.Timestamp(window_end_exclusive, tz="UTC"))
            | (
                (combined["url"] != "")
                & (combined["title"] != "")
                & combined["published_at"].notna()
                & (combined["published_at"] >= pd.Timestamp(COLLECTION_START, tz="UTC"))
                & combined["url"].isin(previous["url"])
            )
    ].copy()

    # Keep old saved articles even when outside the rolling search window.
    # New rows are already limited to the rolling window above.
    old_urls = set(previous["url"])
    keep_old = combined["url"].isin(old_urls)
    keep_recent = (
        (combined["published_at"] >= pd.Timestamp(window_start, tz="UTC"))
        & (combined["published_at"] < pd.Timestamp(window_end_exclusive, tz="UTC"))
    )
    combined = combined[keep_old | keep_recent].copy()
    combined = combined[~combined["url"].isin(historical_urls)].copy()

    combined["matched_terms"] = combined["matched_terms"].fillna("").astype(str)
    combined = (
        combined.groupby("url", as_index=False)
        .agg({
            "title": "first",
            "published_at": "min",
            "publication_date": "first",
            "source": "first",
            "matched_terms": lambda values: "; ".join(sorted({
                term.strip()
                for value in values
                for term in value.split(";")
                if term.strip()
            })),
            "requested_month": "first",
            "collected_at": "first",
        })
    )

    combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True)
    combined["publication_date"] = combined["published_at"].dt.strftime("%Y-%m-%d")
    combined["requested_month"] = combined["published_at"].dt.strftime("%Y-%m")
    combined = combined.sort_values("published_at")
    combined = combined[COLUMNS]
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Rebuild the full monthly report from the complete saved collection.
    monthly = (
        combined.groupby("requested_month")
        .agg(unique_articles=("url", "nunique"))
        .reset_index()
        .rename(columns={"requested_month": "month"})
    )
    monthly["search_terms"] = len(SEARCH_TERMS)
    monthly.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    print("\nIncremental collection complete.")
    print(f"New matching entries found this run: {len(new_frame):,}")
    print(f"Total unique saved 2026 articles: {len(combined):,}")
    print("\nFinal article counts by publication month:")
    print(combined.groupby("requested_month").size().to_string())
    print(f"\nNews saved to: {OUTPUT_FILE.resolve()}")
    print(f"Monthly report saved to: {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    collect_recent_news()
