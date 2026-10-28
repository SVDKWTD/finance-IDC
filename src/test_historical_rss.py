
from datetime import datetime, timezone
import time

import feedparser
import pandas as pd
import requests

RSS_URL = "https://news.google.com/rss/search"


TERMS = [
    "India stock market",
    "Sensex",
    "NIFTY",
    "Indian economy",
    "RBI India",
]


MONTHS = [
    ("2022-01-01", "2022-02-01"),
    ("2022-02-01", "2022-03-01"),
    ("2022-03-01", "2022-04-01"),
    ("2022-04-01", "2022-05-01"),
    ("2022-05-01", "2022-06-01"),
    ("2022-06-01", "2022-07-01"),
    ("2022-07-01", "2022-08-01"),
    ("2022-08-01", "2022-09-01"),
    ("2022-09-01", "2022-10-01"),
    ("2022-10-01", "2022-11-01"),
    ("2022-11-01", "2022-12-01"),
    ("2022-12-01", "2023-01-01"),
]


def main():
    rows = []

    for start, end in MONTHS:
        for term in TERMS:
            query = f"{term} after:{start} before:{end}"
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
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=30,
                )
                response.raise_for_status()

                feed = feedparser.parse(response.content)
                print("Entries returned:", len(feed.entries))

                for entry in feed.entries:
                    rows.append({
                        "search_term": term,
                        "requested_start": start,
                        "requested_end": end,
                        "title": entry.get("title", ""),
                        "published_at": entry.get("published", ""),
                        "url": entry.get("link", ""),
                        "collected_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    })

            except Exception as error:
                print("Request failed:", error)

            time.sleep(3)

    df = pd.DataFrame(rows)

    if df.empty:
        print("\nNo articles returned for these historical periods.")
        return

    df = df[df["url"].fillna("").str.strip() != ""]
    df = df.drop_duplicates(subset=["url"])

    dates = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True
    )
    df["actual_publication_date"] = dates

    print("\n========== HISTORICAL TEST SUMMARY ==========")
    print("Unique articles:", len(df))
    print("\nArticles by requested period:")
    print(df.groupby("requested_start").size().to_string())

    print("\nArticles by actual publication year:")
    print(
        dates.dropna().dt.year.value_counts()
        .sort_index().to_string()
    )

    output = "data/raw/historical_rss_test.csv"
    from pathlib import Path
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nTest results saved to {output}")


if __name__ == "__main__":
    main()
