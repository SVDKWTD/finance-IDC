from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

MARKET_FILE = DATA_DIR / "news_market_daily.csv"
# ISSUE 10 FIX: Read from the full-corpus daily sentiment file produced by analyze_news_sentiment.py
SENTIMENT_FILE = DATA_DIR / "news_sentiment_daily.csv"
OUTPUT_FILE = DATA_DIR / "model_dataset.csv"

MARKET_FEATURES = [
    "gold_return_pct",
    "usd_inr_return_pct",
    "brent_return_pct",
]

SENTIMENT_FEATURES = [
    "mean_sentiment",
    "median_sentiment",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
]


def main():
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Missing market file: {MARKET_FILE}")
    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(f"Missing sentiment file: {SENTIMENT_FILE}")

    market = pd.read_csv(MARKET_FILE)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    market["date"] = pd.to_datetime(
        market["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    sentiment["date"] = pd.to_datetime(
        sentiment["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    required_market = [
        "date",
        "target_next_session_return_pct",
        *MARKET_FEATURES,
    ]

    missing = set(required_market) - set(market.columns)
    if missing:
        raise ValueError(
            f"Market dataset missing columns: {sorted(missing)}"
        )

    required_sentiment = ["date", *SENTIMENT_FEATURES]
    missing = set(required_sentiment) - set(sentiment.columns)
    if missing:
        raise ValueError(
            f"Sentiment dataset missing columns: {sorted(missing)}"
        )

    if market["date"].isna().any():
        raise ValueError("Market dataset contains invalid dates.")

    if sentiment["date"].isna().any():
        raise ValueError("Sentiment dataset contains invalid dates.")

    if market["date"].duplicated().any():
        raise ValueError("Market dataset has duplicate dates.")

    if sentiment["date"].duplicated().any():
        raise ValueError("Sentiment dataset has duplicate dates.")

    # Left-merge sentiment onto market dates. Sessions with zero headlines retain NaN
    # sentiment values which downstream pipelines handle using SimpleImputer.
    combined = market.merge(
        sentiment[["date", *SENTIMENT_FEATURES]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    combined = combined.sort_values("date").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:", OUTPUT_FILE.resolve())
    print("Dataset shape:", combined.shape)
    print("Date range:", combined["date"].min(), "to", combined["date"].max())

    print("\nNew market features:")
    print(combined[MARKET_FEATURES].describe().round(4))

    print("\nMissing values:")
    print(combined[MARKET_FEATURES + SENTIMENT_FEATURES].isna().sum())

    print("\nFinal columns:")
    print(combined.columns.tolist())


if __name__ == "__main__":
    main()