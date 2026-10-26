
from pathlib import Path
import sqlite3


# Find the project folder from this file's location.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "market.db"



from contextlib import contextmanager
from pathlib import Path
import sqlite3


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "market.db"


@contextmanager
def get_connection(db_path=DEFAULT_DB_PATH):
    """Open a database connection and always close it afterward."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()



def create_tables(db_path=DEFAULT_DB_PATH):
    """Create the market-price and news tables if needed."""
    with get_connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE (ticker, date)
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                published_at TEXT,
                source TEXT,
                url TEXT NOT NULL UNIQUE,
                collected_at TEXT
            )
        """)


def upsert_market_price(ticker, row, db_path=DEFAULT_DB_PATH):
    """Insert a daily price, or update it if that ticker-date exists."""
    record = {
        "ticker": ticker,
        "date": str(row["date"])[:10],
        "open": row.get("open"),
        "high": row.get("high"),
        "low": row.get("low"),
        "close": row.get("close"),
        "volume": row.get("volume"),
    }

    with get_connection(db_path) as connection:
        connection.execute("""
            INSERT INTO market_prices
                (ticker, date, open, high, low, close, volume)
            VALUES
                (:ticker, :date, :open, :high, :low, :close, :volume)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
        """, record)


def upsert_news(row, db_path=DEFAULT_DB_PATH):
    """Insert a news article, or update it if its URL already exists."""
    url = row.get("url") or row.get("link")

    if not url:
        raise ValueError("News article is missing its URL.")

    record = {
        "title": row.get("title") or "",
        "published_at": row.get("published_at") or row.get("published"),
        "source": row.get("source"),
        "url": url,
        "collected_at": row.get("collected_at"),
    }

    if not record["title"].strip():
        raise ValueError("News article is missing its title.")

    with get_connection(db_path) as connection:
        connection.execute("""
            INSERT INTO news
                (title, published_at, source, url, collected_at)
            VALUES
                (:title, :published_at, :source, :url, :collected_at)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                published_at = excluded.published_at,
                source = excluded.source,
                collected_at = excluded.collected_at
        """, record)


def read_market_prices(db_path=DEFAULT_DB_PATH):
    """Return all market-price records as dictionaries."""
    with get_connection(db_path) as connection:
        rows = connection.execute("""
            SELECT ticker, date, open, high, low, close, volume
            FROM market_prices
            ORDER BY ticker, date
        """).fetchall()

    return [dict(row) for row in rows]


def read_news(db_path=DEFAULT_DB_PATH):
    """Return all news records as dictionaries."""
    with get_connection(db_path) as connection:
        rows = connection.execute("""
            SELECT title, published_at, source, url, collected_at
            FROM news
            ORDER BY published_at
        """).fetchall()

    return [dict(row) for row in rows]


def update_news_title(url, new_title, db_path=DEFAULT_DB_PATH):
    """Update the title of a saved article using its URL."""
    with get_connection(db_path) as connection:
        cursor = connection.execute("""
            UPDATE news
            SET title = ?
            WHERE url = ?
        """, (new_title, url))

    return cursor.rowcount
