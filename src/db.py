from pathlib import Path
import sqlite3
from contextlib import contextmanager


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


def _news_schema_needs_migration(connection):
    """Check whether news has market support and the correct unique key."""
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(news)").fetchall()
    }

    if "market" not in columns:
        return True

    unique_keys = []
    for index in connection.execute("PRAGMA index_list(news)").fetchall():
        if index["unique"]:
            index_name = index["name"].replace('"', '""')
            fields = [
                row["name"]
                for row in connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                ).fetchall()
            ]
            unique_keys.append(fields)

    return ["url", "market"] not in unique_keys


def _migrate_news_table(connection):
    """Rebuild the old news table while preserving existing records."""
    old_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(news)").fetchall()
    }

    connection.execute("""
        CREATE TABLE news_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            published_at TEXT,
            source TEXT,
            url TEXT NOT NULL,
            collected_at TEXT,
            market TEXT NOT NULL DEFAULT 'unclassified',
            matched_terms TEXT,
            requested_month TEXT,
            UNIQUE (url, market)
        )
    """)

    fields = [
        "id",
        "title",
        "published_at",
        "source",
        "url",
        "collected_at",
        "market",
        "matched_terms",
        "requested_month",
    ]

    select_fields = []
    for field in fields:
        if field in old_columns:
            if field == "market":
                select_fields.append("COALESCE(market, 'unclassified')")
            else:
                select_fields.append(field)
        elif field == "market":
            select_fields.append("'unclassified'")
        else:
            select_fields.append("NULL")

    connection.execute(f"""
        INSERT INTO news_new ({", ".join(fields)})
        SELECT {", ".join(select_fields)}
        FROM news
    """)

    connection.execute("DROP TABLE news")
    connection.execute("ALTER TABLE news_new RENAME TO news")


def create_tables(db_path=DEFAULT_DB_PATH):
    """Create tables and safely migrate an existing news table."""
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
                url TEXT NOT NULL,
                collected_at TEXT,
                market TEXT NOT NULL DEFAULT 'unclassified',
                matched_terms TEXT,
                requested_month TEXT,
                UNIQUE (url, market)
            )
        """)

        if _news_schema_needs_migration(connection):
            _migrate_news_table(connection)


def upsert_market_price(ticker, row, db_path=DEFAULT_DB_PATH):
    """Insert a daily price or update the existing ticker-date record."""
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
    """Insert/update one article-market record."""
    url = row.get("url") or row.get("link")

    if not url:
        raise ValueError("News article is missing its URL.")

    title = row.get("title") or ""
    if not title.strip():
        raise ValueError("News article is missing its title.")

    market = row.get("market") or "unclassified"

    record = {
        "title": title,
        "published_at": row.get("published_at") or row.get("published"),
        "source": row.get("source"),
        "url": url,
        "collected_at": row.get("collected_at"),
        "market": market,
        "matched_terms": row.get("matched_terms"),
        "requested_month": row.get("requested_month"),
    }

    with get_connection(db_path) as connection:
        connection.execute("""
            INSERT INTO news (
                title, published_at, source, url, collected_at,
                market, matched_terms, requested_month
            )
            VALUES (
                :title, :published_at, :source, :url, :collected_at,
                :market, :matched_terms, :requested_month
            )
            ON CONFLICT(url, market) DO UPDATE SET
                title = excluded.title,
                published_at = COALESCE(excluded.published_at, news.published_at),
                source = COALESCE(excluded.source, news.source),
                collected_at = COALESCE(excluded.collected_at, news.collected_at),
                matched_terms = COALESCE(excluded.matched_terms, news.matched_terms),
                requested_month = COALESCE(
                    excluded.requested_month, news.requested_month
                )
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
    """Return all news records, including their market labels."""
    with get_connection(db_path) as connection:
        rows = connection.execute("""
            SELECT
                title, published_at, source, url, collected_at,
                market, matched_terms, requested_month
            FROM news
            ORDER BY published_at
        """).fetchall()

    return [dict(row) for row in rows]


def update_news_title(url, new_title, db_path=DEFAULT_DB_PATH):
    """Update the title of a saved article."""
    with get_connection(db_path) as connection:
        cursor = connection.execute("""
            UPDATE news
            SET title = ?
            WHERE url = ?
        """, (new_title, url))

    return cursor.rowcount