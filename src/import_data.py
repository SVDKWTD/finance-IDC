from pathlib import Path

import pandas as pd

from db import (
    create_tables,
    upsert_market_price,
    upsert_news,
    read_market_prices,
    read_news,
    DEFAULT_DB_PATH,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"


def import_prices():
    """Import the raw NIFTY 50 CSV into SQLite."""
    file_path = RAW_DIR / "nifty50_prices.csv"
    df = pd.read_csv(file_path)

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    for _, row in df.iterrows():
        record = row.where(pd.notna(row), None).to_dict()
        upsert_market_price("^NSEI", record)

    return len(df)


def import_news():
    """Import raw news headlines into SQLite."""
    file_path = RAW_DIR / "news.csv"
    df = pd.read_csv(file_path)

    imported_rows = 0

    for _, row in df.iterrows():
        record = row.where(pd.notna(row), None).to_dict()

        if not record.get("url"):
            continue

        # Preserve the collector's market label.
        # Older rows without a label are marked unclassified.
        record["market"] = record.get("market") or "unclassified"

        upsert_news(record)
        imported_rows += 1

    return imported_rows


def main():
    create_tables()

    prices_read = import_prices()
    news_read = import_news()

    prices_saved = len(read_market_prices())
    news_saved = len(read_news())

    print("Database:", DEFAULT_DB_PATH)
    print("Price CSV rows processed:", prices_read)
    print("News CSV rows processed:", news_read)
    print("Unique market-price records in database:", prices_saved)
    print("Unique news records in database:", news_saved)
    print("Import complete.")


if __name__ == "__main__":
    main()