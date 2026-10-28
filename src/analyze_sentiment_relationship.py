
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Keep rows with a known next-session return and sentiment.
    data = df.dropna(
        subset=["target_next_session_return_pct", "mean_sentiment"]
    ).copy()

    print("===== DATASET SUMMARY =====")
    print(f"Total market sessions: {len(df)}")
    print(f"Usable sessions: {len(data)}")
    print(f"Missing sentiment sessions: {df['mean_sentiment'].isna().sum()}")

    sentiment = data["mean_sentiment"]
    target = data["target_next_session_return_pct"]

    # Correlation tests
    pearson_r, pearson_p = pearsonr(sentiment, target)
    spearman_r, spearman_p = spearmanr(sentiment, target)

    print("\n===== SENTIMENT VS NEXT-SESSION RETURN =====")
    print(f"Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.4g})")
    print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4g})")

    # Compare days with positive vs negative mean sentiment.
    positive_days = data.loc[sentiment > 0, "target_next_session_return_pct"]
    negative_days = data.loc[sentiment < 0, "target_next_session_return_pct"]

    print("\n===== POSITIVE VS NEGATIVE SENTIMENT DAYS =====")
    print(f"Positive sentiment days: {len(positive_days)}")
    print(f"Negative sentiment days: {len(negative_days)}")
    print(f"Mean next-session return after positive sentiment: "
          f"{positive_days.mean():.4f}%")
    print(f"Mean next-session return after negative sentiment: "
          f"{negative_days.mean():.4f}%")

    if len(positive_days) > 1 and len(negative_days) > 1:
        statistic, p_value = ttest_ind(
            positive_days,
            negative_days,
            equal_var=False,
            nan_policy="omit",
        )
        print(f"Welch t-test p-value: {p_value:.4g}")

    # Save correlation scatter plot.
    plt.figure(figsize=(9, 6))
    plt.scatter(sentiment, target, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Mean FinBERT sentiment score")
    plt.ylabel("Next-session NIFTY 50 return (%)")
    plt.title("Daily News Sentiment vs Next-session Market Return")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "sentiment_vs_next_return.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"\nScatter plot saved to:\n{plot_path}")

    # Save a small summary table.
    summary = pd.DataFrame({
        "metric": [
            "usable_sessions",
            "pearson_correlation",
            "pearson_p_value",
            "spearman_correlation",
            "spearman_p_value",
            "positive_sentiment_mean_return_pct",
            "negative_sentiment_mean_return_pct",
        ],
        "value": [
            len(data),
            pearson_r,
            pearson_p,
            spearman_r,
            spearman_p,
            positive_days.mean(),
            negative_days.mean(),
        ],
    })

    summary.to_csv(OUTPUT_DIR / "sentiment_relationship_summary.csv", index=False)
    print("Summary saved to sentiment_relationship_summary.csv")


if __name__ == "__main__":
    main()
