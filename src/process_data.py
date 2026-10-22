
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def process_prices():
    input_file = RAW_DIR / "nifty50_prices.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing {input_file}. Run collect_prices.py first."
        )

    df = pd.read_csv(input_file)

    # Standardise the date column.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"])

    # Ensure prices are numeric.
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows without a valid closing price.
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    # Financial features.
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_5d"] = df["return_1d"].rolling(5).std()

    if "Volume" in df.columns:
        df["volume_change"] = df["Volume"].pct_change()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "nifty50_processed.csv"
    df.to_csv(output_file, index=False)

    print("Price processing complete.")
    print("Rows:", len(df))
    print("Date range:", df["Date"].min(), "to", df["Date"].max())
    print("Missing values:\n", df.isna().sum())
    print("Saved to:", output_file.resolve())


def process_news():
    input_file = RAW_DIR / "news.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing {input_file}. Run collect_news.py first."
        )

    df = pd.read_csv(input_file)

    required = ["title", "published_at", "url"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"News dataset is missing columns: {missing}")

    # Parse timestamps. Keep timezone information when available.
    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True
    )

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["published_at"])
    df = df[df["title"] != ""]
    df = df[df["url"] != ""]
    df = df.drop_duplicates(subset=["url"])
    df = df.sort_values("published_at")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "news_processed.csv"
    df.to_csv(output_file, index=False)

    print("\nNews processing complete.")
    print("Articles:", len(df))

    if not df.empty:
        print(
            "Publication range:",
            df["published_at"].min(),
            "to",
            df["published_at"].max(),
        )

    print("Saved to:", output_file.resolve())


if __name__ == "__main__":
    process_prices()
    process_news()
