
from pathlib import Path
import sqlite3

import pandas as pd

from db import DEFAULT_DB_PATH, create_tables, get_connection


PROJECT_DIR = Path(__file__).resolve().parent.parent

HISTORICAL_NEWS = (
    PROJECT_DIR / "data" / "raw" / "historical_news_merged_2021_2025.csv"
)
RECENT_NEWS = PROJECT_DIR / "data" / "raw" / "recent_news_2026.csv"
PRICES_FILE = (
    PROJECT_DIR / "data" / "processed" / "nifty50_processed.csv"
)


def clean_value(value):
    """Convert pandas missing values to SQLite-compatible values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def ensure_columns(connection, table, columns):
    """Add missing columns without dropping existing data."""
    existing = {
        row["name"]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }

    for column, sql_type in columns.items():
        if column not in existing:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN "
                f"{column} {sql_type}"
            )
            print(f"Added column: {table}.{column}")


def load_news_files():
    frames = []

    for path in [HISTORICAL_NEWS, RECENT_NEWS]:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path.name}")

        # Required fields for the database
        for col in [
            "title", "published_at", "source", "url",
            "collected_at", "matched_terms", "requested_month"
        ]:
            if col not in df.columns:
                df[col] = None

        df["url"] = df["url"].fillna("").astype(str).str.strip()
        df["title"] = df["title"].fillna("").astype(str).str.strip()

        df = df[(df["url"] != "") & (df["title"] != "")].copy()
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No news CSV files were found.")

    news = pd.concat(frames, ignore_index=True)

    # Normalize timestamps and remove records with invalid timestamps.
    news["published_at"] = pd.to_datetime(
        news["published_at"], errors="coerce", utc=True
    )
    news = news[news["published_at"].notna()].copy()

    # Merge metadata for URLs appearing in more than one file.
    news["matched_terms"] = news["matched_terms"].fillna("").astype(str)

    news = (
        news.groupby("url", as_index=False)
        .agg({
            "title": "first",
            "published_at": "min",
            "source": "first",
            "collected_at": "first",
            "matched_terms": lambda values: "; ".join(
                sorted({
                    term.strip()
                    for value in values
                    for term in value.split(";")
                    if term.strip()
                })
            ),
        })
    )

    news["requested_month"] = (
        news["published_at"].dt.strftime("%Y-%m")
    )
    news["published_at"] = news["published_at"].apply(
        lambda value: value.isoformat()
    )

    print("Unique valid news articles to import:", len(news))
    return news


def load_prices_file():
    if not PRICES_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICES_FILE}")

    prices = pd.read_csv(PRICES_FILE)
    print(f"Loaded {len(prices)} price rows.")

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in prices.columns]

    if missing:
        raise ValueError(f"Price CSV is missing columns: {missing}")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date", "Close"]).copy()
    prices["Date"] = prices["Date"].dt.strftime("%Y-%m-%d")
    prices = prices.drop_duplicates(subset=["Date"], keep="last")

    return prices


def import_data():
    create_tables(DEFAULT_DB_PATH)

    news = load_news_files()
    prices = load_prices_file()

    with get_connection(DEFAULT_DB_PATH) as connection:
        # Migrate the existing schema safely.
        ensure_columns(connection, "news", {
            "matched_terms": "TEXT",
            "requested_month": "TEXT",
        })

        ensure_columns(connection, "market_prices", {
            "adj_close": "REAL",
            "daily_return": "REAL",
            "daily_return_pct": "REAL",
        })

        # Bulk insert/update news.
        news_records = []
        for _, row in news.iterrows():
            news_records.append({
                "title": row["title"],
                "published_at": row["published_at"],
                "source": clean_value(row["source"]),
                "url": row["url"],
                "collected_at": clean_value(row["collected_at"]),
                "matched_terms": clean_value(row["matched_terms"]),
                "requested_month": clean_value(row["requested_month"]),
            })

        connection.executemany("""
            INSERT INTO news (
                title, published_at, source, url, collected_at,
                matched_terms, requested_month
            )
            VALUES (
                :title, :published_at, :source, :url, :collected_at,
                :matched_terms, :requested_month
            )
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                published_at = excluded.published_at,
                source = COALESCE(excluded.source, news.source),
                matched_terms = excluded.matched_terms,
                requested_month = excluded.requested_month
        """, news_records)

        # Bulk insert/update daily NIFTY 50 prices.
        price_records = []

        for _, row in prices.iterrows():
            price_records.append({
                "ticker": "^NSEI",
                "date": row["Date"],
                "open": clean_value(row["Open"]),
                "high": clean_value(row["High"]),
                "low": clean_value(row["Low"]),
                "close": clean_value(row["Close"]),
                "volume": clean_value(row["Volume"]),
                "adj_close": clean_value(row.get("Adj_Close")),
                "daily_return": clean_value(row.get("Daily_Return")),
                "daily_return_pct": clean_value(
                    row.get("Daily_Return_Pct")
                ),
            })

        connection.executemany("""
            INSERT INTO market_prices (
                ticker, date, open, high, low, close, volume,
                adj_close, daily_return, daily_return_pct
            )
            VALUES (
                :ticker, :date, :open, :high, :low, :close, :volume,
                :adj_close, :daily_return, :daily_return_pct
            )
            ON CONFLICT(ticker, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                adj_close = excluded.adj_close,
                daily_return = excluded.daily_return,
                daily_return_pct = excluded.daily_return_pct
        """, price_records)

        # Verify the database contents before committing.
        news_count = connection.execute(
            "SELECT COUNT(*) FROM news"
        ).fetchone()[0]

        price_count = connection.execute(
            "SELECT COUNT(*) FROM market_prices WHERE ticker = '^NSEI'"
        ).fetchone()[0]

        news_dates = connection.execute("""
            SELECT MIN(published_at), MAX(published_at) FROM news
        """).fetchone()

        price_dates = connection.execute("""
            SELECT MIN(date), MAX(date)
            FROM market_prices
            WHERE ticker = '^NSEI'
        """).fetchone()

        print("\n===== SQLITE IMPORT REPORT =====")
        print("Database:", DEFAULT_DB_PATH)
        print("News records in database:", news_count)
        print("NIFTY 50 records in database:", price_count)
        print("News timestamp range:", news_dates[0], "to", news_dates[1])
        print("NIFTY 50 date range:", price_dates[0], "to", price_dates[1])


if __name__ == "__main__":
    import_data()
