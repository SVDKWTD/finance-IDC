from calendar import monthrange
from datetime import date, datetime, timezone
from pathlib import Path
import time
import feedparser
import pandas as pd
import requests

RSS_URL = "https://news.google.com/rss/search"

# Multi-Asset Search Terms Mapped to Relational Market Tags
SEARCH_QUERIES = {
    "nifty": [
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
    ],
    "gold": [
        "Gold price India MCX",
        "Gold rate India market news",
    ],
    "brent": [
        "Brent crude oil price India",
        "Crude oil news market India",
    ],
    "usd_inr": [
        "USD INR exchange rate India",
        "Indian Rupee vs Dollar news",
    ],
}

START_YEAR = 2021
END_YEAR = 2025
REQUEST_PAUSE_SECONDS = 2
MAX_ARTICLES_PER_SESSION = 25  # Uniform daily session density cap
RANDOM_SEED = 42

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "raw"

ARTICLES_FILE = OUTPUT_DIR / "historical_news_merged_2021_2025.csv"
COVERAGE_FILE = OUTPUT_DIR / "monthly_coverage_backfill_2021_2025.csv"


def month_ranges():
    """Generate start and exclusive end dates for every month from START_YEAR to END_YEAR."""
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = date(year, month, 1)
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            yield start, end


def apply_density_cap(df: pd.DataFrame, max_per_session: int = 25, seed: int = 42) -> pd.DataFrame:
    """Enforces a maximum headline cap per publication date across all asset categories."""
    if df.empty or "published_at" not in df.columns:
        return df

    df_work = df.copy()
    df_work["_pub_date"] = pd.to_datetime(df_work["published_at"], errors="coerce", utc=True).dt.date

    capped_frames = []
    for pub_date, group in df_work.groupby("_pub_date"):
        if len(group) > max_per_session:
            sampled = group.sample(n=max_per_session, random_state=seed)
            capped_frames.append(sampled)
        else:
            capped_frames.append(group)

    result = pd.concat(capped_frames, ignore_index=True).drop(columns=["_pub_date"])
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    session = requests.Session()
    session.headers.update(headers)

    all_articles = {}
    coverage_rows = []

    months = list(month_ranges())

    # Flatten query tuple: (market_tag, term)
    flat_queries = [
        (market_tag, term)
        for market_tag, terms in SEARCH_QUERIES.items()
        for term in terms
    ]

    total_queries = len(flat_queries) * len(months)
    print(f"Starting news backfill ({START_YEAR}-{END_YEAR}): {len(months)} months x {len(flat_queries)} terms ({total_queries} queries)...")

    for month_index, (start, end) in enumerate(months, start=1):
        month_key = start.strftime("%Y-%m")
        print(f"\n[{month_index}/{len(months)}] Auditing {month_key}")

        received = 0
        errors = 0

        for market_tag, term in flat_queries:
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

                    timestamp = pd.to_datetime(published, errors="coerce", utc=True)
                    if pd.isna(timestamp):
                        continue

                    actual_date = timestamp.date()
                    if not (start <= actual_date < end):
                        continue

                    # Deduplicate on composite primary key (url, market)
                    dedup_key = (url, market_tag)

                    if dedup_key not in all_articles:
                        source_info = entry.get("source", {})
                        if not isinstance(source_info, dict):
                            source_info = {}

                        all_articles[dedup_key] = {
                            "title": title,
                            "published_at": timestamp.isoformat(),
                            "publication_date": actual_date.isoformat(),
                            "source": source_info.get("title", ""),
                            "url": url,
                            "market": market_tag,
                            "matched_terms": term,
                            "requested_month": month_key,
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        old_terms = all_articles[dedup_key]["matched_terms"]
                        if term not in old_terms.split(" | "):
                            all_articles[dedup_key]["matched_terms"] = old_terms + " | " + term

            except (requests.RequestException, ValueError) as error:
                errors += 1
                print(f"  Query failed for '{term}' ({market_tag}): {error}")

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
        print(f"  Unique valid articles saved so far: {len(all_articles):,}")
        print(f"  Query errors: {errors}")

        # Intermediate checkpoint export
        df_raw = pd.DataFrame(all_articles.values())
        df_raw.to_csv(ARTICLES_FILE, index=False, encoding="utf-8-sig")
        pd.DataFrame(coverage_rows).to_csv(COVERAGE_FILE, index=False, encoding="utf-8-sig")

    print("\n========== BACKFILL COMPLETE ==========")
    df_all = pd.DataFrame(all_articles.values())
    print(f"Total raw articles collected: {len(df_all):,}")
    print("\nArticle breakdown by market tag:")
    print(df_all["market"].value_counts().to_string())

    # Apply Density Cap across all sessions (N <= 25)
    df_capped = apply_density_cap(df_all, max_per_session=MAX_ARTICLES_PER_SESSION, seed=RANDOM_SEED)
    print(f"\nArticles after session density cap (max {MAX_ARTICLES_PER_SESSION}/session): {len(df_capped):,}")

    df_capped.to_csv(ARTICLES_FILE, index=False, encoding="utf-8-sig")
    print(f"Final dataset exported to: {ARTICLES_FILE}")


if __name__ == "__main__":
    main()