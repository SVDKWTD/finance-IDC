
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

MARKET_FILE = DATA_DIR / "news_market_daily.csv"
SENTIMENT_FILE = DATA_DIR / "news_sentiment_daily.csv"
OUTPUT_FILE = DATA_DIR / "model_dataset.csv"


def main():
    print("Loading market and monthly-sampled sentiment datasets...")

    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market file not found: {MARKET_FILE}")

    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Sampled sentiment file not found: {SENTIMENT_FILE}\n"
            "Run src\\monthly_sampling_experiment.py first."
        )

    market = pd.read_csv(MARKET_FILE)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    market["date"] = pd.to_datetime(
        market["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    sentiment["date"] = pd.to_datetime(
        sentiment["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    if market["date"].isna().any():
        raise ValueError("Invalid dates found in the market dataset.")

    if sentiment["date"].isna().any():
        raise ValueError("Invalid dates found in sampled sentiment.")

    if market["date"].duplicated().any():
        raise ValueError("Market dataset contains duplicate dates.")

    if sentiment["date"].duplicated().any():
        raise ValueError("Sampled sentiment contains duplicate dates.")

    # The sampled file must supply both volume features as well as sentiment.
    required_sentiment_columns = [
        "date",
        "mean_sentiment",
        "median_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    ]

    missing = set(required_sentiment_columns) - set(sentiment.columns)
    if missing:
        raise ValueError(
            "Sampled sentiment file is missing columns: "
            f"{sorted(missing)}"
        )

    # Remove original-corpus news features before merging.
    # The model must use volume and source diversity from the sampled corpus.


    combined = market.merge(
        sentiment[required_sentiment_columns],
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
        combined[sentiment_features].isna().any(axis=1).sum()
    )

    print(f"Market sessions: {len(combined):,}")
    print(f"Sessions without matched sentiment: {unmatched:,}")
    print(f"Sessions with missing sampled features: {missing_sentiment:,}")

    if unmatched or missing_sentiment:
        raise ValueError(
            "The merged dataset has missing sampled-news features. "
            "Output was not written; inspect the source files."
        )

    combined = combined.sort_values("date").reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved main model dataset to:\n{OUTPUT_FILE}")
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


if __name__ == "__main__":
    main()
