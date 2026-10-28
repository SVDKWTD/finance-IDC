
import pandas as pd

from db import DEFAULT_DB_PATH, get_connection


def main():
    with get_connection(DEFAULT_DB_PATH) as connection:
        news = pd.read_sql_query("""
            SELECT published_at, source, title
            FROM news
            WHERE published_at IS NOT NULL
        """, connection)

        prices = pd.read_sql_query("""
            SELECT date, close, daily_return_pct
            FROM market_prices
            WHERE ticker = '^NSEI'
            ORDER BY date
        """, connection)

    news["published_at"] = pd.to_datetime(
        news["published_at"], errors="coerce", utc=True
    )
    news["publication_date"] = news["published_at"].dt.date

    prices["date"] = pd.to_datetime(
        prices["date"], errors="coerce"
    ).dt.date

    print("===== DATASET ALIGNMENT CHECK =====")
    print("News records:", len(news))
    print("Market observations:", len(prices))
    print("Unique news publication dates:",
          news["publication_date"].nunique())

    print("\nNews counts by year:")
    print(news.groupby(news["published_at"].dt.year).size().to_string())

    print("\nNews counts by month in 2026:")
    recent = news[news["published_at"].dt.year == 2026]
    print(
        recent.groupby(recent["published_at"].dt.strftime("%Y-%m"))
        .size()
        .to_string()
    )

    print("\nMarket date range:")
    print(prices["date"].min(), "to", prices["date"].max())

    trading_dates = set(prices["date"].dropna())
    news_dates = set(news["publication_date"].dropna())

    print("\nNews publication dates that are trading dates:",
          len(news_dates & trading_dates))
    print("News publication dates that are not trading dates:",
          len(news_dates - trading_dates))

    print("\nSample news records:")
    print(news[["published_at", "source", "title"]].head(5).to_string(
        index=False
    ))


if __name__ == "__main__":
    main()
