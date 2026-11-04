from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

MARKET_FILE = DATA_DIR / "news_market_daily.csv"
SENTIMENT_FILE = DATA_DIR / "news_sentiment_daily_sampled.csv"
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
        raise ValueError("Sampled sentiment dataset has duplicate dates.")

    combined = market.merge(
        sentiment[["date", *SENTIMENT_FEATURES]],
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = combined.loc[combined["_merge"] != "both", "date"]
    if not unmatched.empty:
        raise ValueError(
            "No sampled sentiment found for market dates: "
            f"{unmatched.head(10).tolist()}"
            + (" ..." if len(unmatched) > 10 else "")
        )

    combined = combined.drop(columns="_merge")

    if combined[SENTIMENT_FEATURES].isna().any().any():
        missing_counts = combined[SENTIMENT_FEATURES].isna().sum()
        raise ValueError(
            "Missing sampled sentiment values after merge:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    combined = combined.sort_values("date").reset_index(drop=True)

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