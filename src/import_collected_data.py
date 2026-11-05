from pathlib import Path
import re
import pandas as pd

from db import DEFAULT_DB_PATH, create_tables, get_connection


PROJECT_DIR = Path(__file__).resolve().parent.parent

# Primary historical news file (~21.2k headlines)
HISTORICAL_NEWS = (
    PROJECT_DIR / "data" / "raw" / "historical_news_merged_2021_2025.csv"
)

# Fallback to initial file if merged version is absent
if not HISTORICAL_NEWS.exists():
    HISTORICAL_NEWS = (
        PROJECT_DIR / "data" / "raw" / "historical_news_2021_2025.csv"
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
                f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
            )
            print(f"Added column: {table}.{column}")


def normalize_title(title: str) -> str:
    """Strip punctuation and whitespace for clean title matching."""
    title = str(title).lower().strip()
    # Strip trailing source tags (e.g. " - Economic Times")
    title = re.sub(r"\s*-\s*[^-]+$", "", title)
    return re.sub(r"[^\w\s]", "", title).strip()


def load_news_files():
    frames = []

    for path in [HISTORICAL_NEWS, RECENT_NEWS]:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path.name}")

        required_columns = [
            "title",
            "published_at",
            "source",
            "url",
            "collected_at",
            "matched_terms",
            "requested_month",
            "market",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        df["url"] = df["url"].fillna("").astype(str).str.strip()
        df["title"] = df["title"].fillna("").astype(str).str.strip()

        # ISSUE 1 FIX: Default missing/blank market tags to 'nifty' instead of 'unclassified'
        df["market"] = (
            df["market"]
            .fillna("nifty")
            .astype(str)
            .str.strip()
        )
        df.loc[(df["market"] == "") | (df["market"] == "unclassified"), "market"] = "nifty"

        # Parse timestamps independently
        df["published_at"] = pd.to_datetime(
            df["published_at"],
            format="mixed",
            errors="coerce",
            utc=True,
        )

        before = len(df)
        df = df[
            (df["url"] != "")
            & (df["title"] != "")
            & df["published_at"].notna()
        ].copy()

        print(
            f"   Valid rows: {len(df)} "
            f"(removed {before - len(df)} for blank URL/title or invalid date)"
        )

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No news CSV files were found.")

    news = pd.concat(frames, ignore_index=True)
    print("Rows after combining files:", len(news))

    # ISSUE 8 FIX: De-duplicate syndicated stories on normalized title + publication date
    news["_norm_title"] = news["title"].apply(normalize_title)
    news["_pub_date"] = news["published_at"].dt.date
    
    before_title_dedup = len(news)
    news = news.drop_duplicates(subset=["_norm_title", "_pub_date", "market"], keep="first")
    print(f"De-duplicated syndicated titles: removed {before_title_dedup - len(news)} rows.")
    news.drop(columns=["_norm_title", "_pub_date"], inplace=True)

    news["matched_terms"] = news["matched_terms"].fillna("").astype(str)

    # Retain one record per URL and market, combining matched search terms.
    news = (
        news.groupby(["url", "market"], as_index=False, sort=False)
        .agg(
            title=("title", "first"),
            published_at=("published_at", "min"),
            source=("source", "first"),
            collected_at=("collected_at", "first"),
            matched_terms=(
                "matched_terms",
                lambda values: "; ".join(
                    sorted(
                        {
                            term.strip()
                            for value in values
                            for term in value.split(";")
                            if term.strip()
                        }
                    )
                ),
            ),
        )
    )

    news["requested_month"] = news["published_at"].dt.strftime("%Y-%m")
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
        ensure_columns(
            connection,
            "news",
            {
                "matched_terms": "TEXT",
                "requested_month": "TEXT",
                "market": "TEXT NOT NULL DEFAULT 'nifty'",
            },
        )

        ensure_columns(
            connection,
            "market_prices",
            {
                "adj_close": "REAL",
                "daily_return": "REAL",
                "daily_return_pct": "REAL",
            },
        )

        news_records = [
            {
                "title": row["title"],
                "published_at": row["published_at"],
                "source": clean_value(row["source"]),
                "url": row["url"],
                "collected_at": clean_value(row["collected_at"]),
                "matched_terms": clean_value(row["matched_terms"]),
                "requested_month": clean_value(row["requested_month"]),
                "market": clean_value(row["market"]),
            }
            for _, row in news.iterrows()
        ]

        connection.executemany(
            """
            INSERT INTO news (
                title, published_at, source, url, collected_at,
                matched_terms, requested_month, market
            )
            VALUES (
                :title, :published_at, :source, :url, :collected_at,
                :matched_terms, :requested_month, :market
            )
            ON CONFLICT(url, market) DO UPDATE SET
                title = excluded.title,
                published_at = excluded.published_at,
                source = COALESCE(excluded.source, news.source),
                matched_terms = excluded.matched_terms,
                requested_month = excluded.requested_month
            """,
            news_records,
        )

        price_records = [
            {
                "ticker": "^NSEI",
                "date": row["Date"],
                "open": clean_value(row["Open"]),
                "high": clean_value(row["High"]),
                "low": clean_value(row["Low"]),
                "close": clean_value(row["Close"]),
                "volume": clean_value(row["Volume"]),
                "adj_close": clean_value(row.get("Adj_Close")),
                "daily_return": clean_value(row.get("Daily_Return")),
                "daily_return_pct": clean_value(row.get("Daily_Return_Pct")),
            }
            for _, row in prices.iterrows()
        ]

        connection.executemany(
            """
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
            """,
            price_records,
        )

        news_count = connection.execute(
            "SELECT COUNT(*) FROM news"
        ).fetchone()[0]

        price_count = connection.execute(
            "SELECT COUNT(*) FROM market_prices WHERE ticker = '^NSEI'"
        ).fetchone()[0]

        news_dates = connection.execute(
            "SELECT MIN(published_at), MAX(published_at) FROM news"
        ).fetchone()

        price_dates = connection.execute(
            """
            SELECT MIN(date), MAX(date)
            FROM market_prices
            WHERE ticker = '^NSEI'
            """
        ).fetchone()

        print("\n===== SQLITE IMPORT REPORT =====")
        print("Database:", DEFAULT_DB_PATH)
        print("News records in database:", news_count)
        print("NIFTY 50 records in database:", price_count)
        print("News timestamp range:", news_dates[0], "to", news_dates[1])
        print("NIFTY 50 date range:", price_dates[0], "to", price_dates[1])


if __name__ == "__main__":
    import_data()