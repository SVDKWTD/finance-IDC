
from pathlib import Path

import pandas as pd

from db import DEFAULT_DB_PATH, get_connection


PROJECT_DIR = Path(__file__).resolve().parent.parent

ARTICLE_OUTPUT = (
    PROJECT_DIR / "data" / "processed" / "news_market_alignment.csv"
)
DAILY_OUTPUT = (
    PROJECT_DIR / "data" / "processed" / "news_market_daily.csv"
)

START_DATE = "2021-01-01"
END_DATE = "2027-01-01"  # Exclusive; includes all of 2026

MARKET_TIMEZONE = "Asia/Kolkata"
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30


def main():
    with get_connection(DEFAULT_DB_PATH) as connection:
        news = pd.read_sql_query("""
            SELECT id, title, published_at, source, url
            FROM news
            WHERE published_at IS NOT NULL
        """, connection)

        prices = pd.read_sql_query("""
            SELECT date, open, high, low, close, volume,
                   adj_close, daily_return, daily_return_pct
            FROM market_prices
            WHERE ticker = '^NSEI'
            ORDER BY date
        """, connection)

    if news.empty or prices.empty:
        raise ValueError("The news table or market-price table is empty.")

    # Parse all timestamps consistently.
    news["published_at"] = pd.to_datetime(
        news["published_at"], errors="coerce", utc=True
    )
    news = news.dropna(subset=["published_at"]).copy()

    # Limit the analysis to 2021-2026; leave SQLite untouched.
    news = news[
        (news["published_at"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (news["published_at"] < pd.Timestamp(END_DATE, tz="UTC"))
    ].copy()

    # Convert publication times to Indian Standard Time.
    news["published_at_ist"] = (
        news["published_at"].dt.tz_convert(MARKET_TIMEZONE)
    )

    prices["date"] = pd.to_datetime(
        prices["date"], errors="coerce"
    ).dt.normalize()

    prices = (
        prices.dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Construct a timezone-aware nominal market-close timestamp
    # for each actual trading date in the price dataset.
    session_dates = prices["date"]

    market_closes = (
        session_dates
        + pd.Timedelta(
            hours=MARKET_CLOSE_HOUR,
            minutes=MARKET_CLOSE_MINUTE,
        )
    ).dt.tz_localize(MARKET_TIMEZONE)

    # Assign each article to the first market close at or after
    # its publication time. If it arrives after the close, it
    # maps to the next available trading session.
    publication_times = news["published_at_ist"]

    close_ns = market_closes.astype("int64").to_numpy()
    publication_ns = publication_times.astype("int64").to_numpy()

    import numpy as np

    positions = np.searchsorted(
        close_ns, publication_ns, side="left"
    )

    has_session = positions < len(prices)

    news = news.loc[has_session].copy()
    positions = positions[has_session]

    news["feature_date"] = (
        prices.iloc[positions]["date"].to_numpy()
    )

    # Save the article-level alignment for inspection.
    ARTICLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    news["feature_date"] = pd.to_datetime(news["feature_date"])
    news["feature_date"] = news["feature_date"].dt.strftime("%Y-%m-%d")

    news.to_csv(
        ARTICLE_OUTPUT, index=False, encoding="utf-8-sig"
    )

    # Aggregate article counts and source diversity per session.
    daily_news = (
        news.groupby("feature_date")
        .agg(
            article_count=("url", "nunique"),
            unique_sources=("source", "nunique"),
        )
        .reset_index()
        .rename(columns={"feature_date": "date"})
    )

    daily_news["date"] = pd.to_datetime(daily_news["date"])

    # Merge the news features with market observations.
    daily = prices.merge(daily_news, on="date", how="left")
    daily["article_count"] = (
        daily["article_count"].fillna(0).astype(int)
    )
    daily["unique_sources"] = (
        daily["unique_sources"].fillna(0).astype(int)
    )

    # The target is the next available trading session's return.
    # This avoids using today's closing return as tomorrow's target.
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["target_next_session_return_pct"] = (
        daily["daily_return_pct"].shift(-1)
    )
    daily["target_next_session_date"] = (
        daily["date"].shift(-1)
    )

    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["target_next_session_date"] = (
        daily["target_next_session_date"].dt.strftime("%Y-%m-%d")
    )

    daily.to_csv(
        DAILY_OUTPUT, index=False, encoding="utf-8-sig"
    )

    print("===== NEWS-MARKET ALIGNMENT REPORT =====")
    print("News articles in analysis period:", len(news))
    print("Trading sessions:", len(daily))
    print("Sessions with at least one article:",
          int((daily["article_count"] > 0).sum()))
    print("Sessions without articles:",
          int((daily["article_count"] == 0).sum()))
    print("Articles per session:")
    print(daily["article_count"].describe().round(2).to_string())

    print("\nMarket date range:")
    print(daily["date"].min(), "to", daily["date"].max())

    print("\nRows without a next-session target:",
          int(daily["target_next_session_return_pct"].isna().sum()))

    print("\nArticle-level alignment saved to:")
    print(ARTICLE_OUTPUT.resolve())

    print("\nDaily dataset saved to:")
    print(DAILY_OUTPUT.resolve())

    print("\nSample daily rows:")
    print(daily.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
