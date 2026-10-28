
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MARKET_FILE = PROJECT_DIR / "data" / "processed" / "news_market_daily.csv"
SENTIMENT_FILE = PROJECT_DIR / "data" / "processed" / "news_sentiment_daily.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"


def main():
    print("Loading market and sentiment datasets...")

    market = pd.read_csv(MARKET_FILE)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    market["date"] = pd.to_datetime(market["date"]).dt.strftime("%Y-%m-%d")
    sentiment["date"] = (
        pd.to_datetime(sentiment["date"]).dt.strftime("%Y-%m-%d")
    )

    # Keep only sentiment features; market already has its own article count.
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
            f"Sentiment file is missing columns: {sorted(missing)}"
        )

    if market["date"].duplicated().any():
        raise ValueError("Market dataset contains duplicate dates.")

    if sentiment["date"].duplicated().any():
        raise ValueError("Sentiment dataset contains duplicate dates.")

    combined = market.merge(
        sentiment[sentiment_columns],
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (combined["_merge"] == "left_only").sum()
    combined = combined.drop(columns="_merge")

    # Keep the original article count from the alignment dataset.
    # Missing sentiment is not the same as neutral sentiment.
    sentiment_features = [
        "mean_sentiment",
        "median_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    ]

    missing_sentiment = combined["mean_sentiment"].isna()
    print(f"Market sessions: {len(combined):,}")
    print(f"Sessions without matched sentiment: {unmatched:,}")

    # Missing values are retained for now rather than being labelled neutral.
    combined = combined.sort_values("date").reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved combined dataset to:\n{OUTPUT_FILE}")
    print("\nDataset shape:", combined.shape)
    print("\nColumns:")
    print(combined.columns.tolist())

    print("\nSample rows:")
    print(combined.head().to_string(index=False))

    print("\nTarget availability:")
    print(
        combined["target_next_session_return_pct"]
        .describe()
        .to_string()
    )

    print(
        "\nRows with missing next-session target:",
        combined["target_next_session_return_pct"].isna().sum(),
    )

    print(
        "Rows with missing sentiment:",
        int(missing_sentiment.sum()),
    )


if __name__ == "__main__":
    main()
