
from db import DEFAULT_DB_PATH, get_connection

with get_connection(DEFAULT_DB_PATH) as connection:
    print("DATABASE:", DEFAULT_DB_PATH)

    print("\n--- News schema ---")
    for row in connection.execute("PRAGMA table_info(news)"):
        print(dict(row))

    print("\n--- News counts by publication year ---")
    rows = connection.execute("""
        SELECT substr(published_at, 1, 4) AS year, COUNT(*) AS articles
        FROM news
        GROUP BY year
        ORDER BY year
    """).fetchall()
    for row in rows:
        print(dict(row))

    print("\n--- News timestamp extremes ---")
    rows = connection.execute("""
        SELECT published_at, title, url
        FROM news
        WHERE published_at IS NOT NULL
        ORDER BY published_at DESC
        LIMIT 10
    """).fetchall()
    for row in rows:
        print(dict(row))

    print("\n--- October 2026 records ---")
    row = connection.execute("""
        SELECT COUNT(*) AS articles
        FROM news
        WHERE published_at LIKE '2026-10%'
    """).fetchone()
    print(dict(row))

    print("\n--- Market price counts by year ---")
    rows = connection.execute("""
        SELECT substr(date, 1, 4) AS year, COUNT(*) AS trading_days
        FROM market_prices
        WHERE ticker = '^NSEI'
        GROUP BY year
        ORDER BY year
    """).fetchall()
    for row in rows:
        print(dict(row))
