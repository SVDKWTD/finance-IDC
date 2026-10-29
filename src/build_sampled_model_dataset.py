
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

MARKET_FILE = (
    PROJECT_DIR / "data" / "processed" / "news_market_daily.csv"
)

SENTIMENT_FILE = (
    PROJECT_DIR / "data" / "processed" / "news_sentiment_daily_sampled.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "model_dataset_sampled.csv"
)


def main():
    print("Loading market and sampled sentiment datasets...")

    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market file not found: {MARKET_FILE}")

    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Sampled sentiment file not found: {SENTIMENT_FILE}"
        )

    market = pd.read_csv(MARKET_FILE)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    market["date"] = pd.to_datetime(
        market["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    sentiment["date"] = pd.to_datetime(
        sentiment["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    sentiment_columns = [
        "date",
        "mean_sentiment",
        "median_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    ]

    missing = set(sentiment_columns) - set(sentiment.columns)
    if missing:
        raise ValueError(
            f"Sampled sentiment file is missing columns: {sorted(missing)}"
        )

    if market["date"].duplicated().any():
        raise ValueError("Market dataset contains duplicate dates.")

    if sentiment["date"].duplicated().any():
        raise ValueError("Sampled sentiment dataset contains duplicate dates.")

    combined = market.merge(
        sentiment[sentiment_columns],
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((combined["_merge"] == "left_only").sum())
    combined = combined.drop(columns="_merge")

    sentiment_features = [
        "mean_sentiment",
        "median_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    ]

    missing_sentiment = int(
        combined["mean_sentiment"].isna().sum()
    )

    combined = combined.sort_values("date").reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n===== SAMPLED MODEL DATASET REPORT =====")
    print(f"Market sessions: {len(combined):,}")
    print(f"Sessions without matched sentiment: {unmatched:,}")
    print(f"Rows with missing sentiment: {missing_sentiment:,}")
    print(f"Rows with missing next-session target: "
          f"{combined['target_next_session_return_pct'].isna().sum():,}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"\nSaved dataset to:\n{OUTPUT_FILE}")
    print("Dataset shape:", combined.shape)

    print("\nSample rows:")
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    main()
