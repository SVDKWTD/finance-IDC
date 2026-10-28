
from pathlib import Path

import pandas as pd

from db import read_market_prices, read_news


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def clean_prices():
    """Clean daily prices and calculate basic market indicators."""

    df = pd.DataFrame(read_market_prices())

    if df.empty:
        raise ValueError("No market prices found in SQLite.")

    # Keep NIFTY 50 for the current stage.
    df = df[df["ticker"] == "^NSEI"].copy()

    # Convert dates and numeric columns.
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")

    columns = ["open", "high", "low", "close", "volume"]

    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove invalid and duplicate records.
    df = df.dropna(subset=["Date", "close"])
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Daily percentage change expressed as a decimal.
    df["return_1d"] = df["close"].pct_change()

    # Standard deviation of returns over five trading days.
    df["volatility_5d"] = df["return_1d"].rolling(5).std()

    # Daily percentage change in trading volume.
    df["volume_change"] = df["volume"].pct_change()

    # Match the column names used by our existing scripts.
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    output_columns = [
        "Date", "Close", "High", "Low", "Open", "Volume",
        "return_1d", "volatility_5d", "volume_change"
    ]

    return df[output_columns]


def clean_news():
    """Clean news records and standardize publication dates."""

    df = pd.DataFrame(read_news())

    if df.empty:
        raise ValueError("No news records found in SQLite.")

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["title", "url", "published_at"])
    df = df.drop_duplicates(subset=["url"])
    df = df.sort_values("published_at").reset_index(drop=True)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = clean_prices()
    news = clean_news()

    price_path = OUTPUT_DIR / "nifty50_processed.csv"
    news_path = OUTPUT_DIR / "news_processed.csv"

    prices.to_csv(price_path, index=False)
    news.to_csv(news_path, index=False)

    print("Cleaning completed.")
    print("Processed price records:", len(prices))
    print("Processed news records:", len(news))
    print("Saved:", price_path)
    print("Saved:", news_path)


if __name__ == "__main__":
    main()
