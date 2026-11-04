from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "news_sentiment_scored.csv"
)

OUTPUT_ARTICLES = (
    PROJECT_DIR / "data" / "processed"
    / "news_sentiment_sampled_articles.csv"
)

OUTPUT_DAILY = (
    PROJECT_DIR / "data" / "processed"
    / "news_sentiment_daily_sampled.csv"
)

ARTICLES_PER_SAMPLE = 20
NUMBER_OF_SAMPLES = 1000
RANDOM_SEED = 42


def select_representative_sample(group, rng):
    """
    Draw repeated random samples of 20 articles and retain the sample
    whose mean sentiment is closest to the mean of all sample means.

    Sampling is without replacement within each draw. Different draws
    may contain some of the same articles.
    """
    if len(group) <= ARTICLES_PER_SAMPLE:
        selected = group.copy()
        selected["sampling_sample_mean"] = selected["sentiment_score"].mean()
        selected["sampling_target_mean"] = selected["sentiment_score"].mean()
        selected["sampling_method"] = "all_articles_under_cap"
        return selected

    sentiment_values = group["sentiment_score"].to_numpy(dtype=float)

    sample_means = np.empty(NUMBER_OF_SAMPLES, dtype=float)
    sample_positions = []

    for sample_index in range(NUMBER_OF_SAMPLES):
        positions = rng.choice(
            len(group),
            size=ARTICLES_PER_SAMPLE,
            replace=False,
        )
        sample_positions.append(positions)
        sample_means[sample_index] = sentiment_values[positions].mean()

    target_mean = float(sample_means.mean())

    # Select the draw whose mean is closest to the average sample mean.
    selected_index = int(np.argmin(np.abs(sample_means - target_mean)))
    selected_positions = sample_positions[selected_index]

    selected = group.iloc[selected_positions].copy()
    selected_sample_mean = float(sample_means[selected_index])

    selected["sampling_sample_mean"] = selected_sample_mean
    selected["sampling_target_mean"] = target_mean
    selected["sampling_method"] = "representative_random_sample"

    return selected


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"Loading scored articles from:\n{INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    required = {
        "title",
        "feature_date",
        "sentiment_score",
        "sentiment_label",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=[
            "title",
            "feature_date",
            "sentiment_score",
            "sentiment_label",
        ]
    ).copy()

    df["title"] = df["title"].astype(str).str.strip()
    df["feature_date"] = df["feature_date"].astype(str).str.strip()
    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce",
    )
    df = df.dropna(subset=["sentiment_score"])
    df = df[df["title"] != ""].copy()

    # Remove duplicate articles before sampling.
    if "url" in df.columns:
        df["url"] = df["url"].fillna("").astype(str).str.strip()
        has_url = df["url"] != ""

        with_url = df[has_url].drop_duplicates(
            subset=["url"],
            keep="last",
        )
        without_url = df[~has_url].drop_duplicates(
            subset=["title", "feature_date"],
            keep="last",
        )

        df = pd.concat(
            [with_url, without_url],
            ignore_index=True,
        )
    else:
        df = df.drop_duplicates(
            subset=["title", "feature_date"],
            keep="last",
        )

    if df.empty:
        raise ValueError("No valid scored articles remain after cleaning.")

    # Use the system date to identify the current calendar month.
    today = date.today()
    current_month = today.strftime("%Y-%m")

    print(f"Current month being sampled: {current_month}")
    print(f"Articles per random sample: {ARTICLES_PER_SAMPLE}")
    print(f"Random samples per day: {NUMBER_OF_SAMPLES}")

    original_counts = df.groupby("feature_date").size()
    sampled_groups = []

    # One deterministic random stream for reproducible reruns.
    rng = np.random.default_rng(RANDOM_SEED)

    for feature_date, group in df.groupby("feature_date", sort=True):
        # Apply the new sampling procedure only to the current month.
        if (
            feature_date.startswith(current_month)
            and len(group) > ARTICLES_PER_SAMPLE
        ):
            selected = select_representative_sample(group, rng)
        else:
            selected = group.copy()
            selected["sampling_sample_mean"] = selected[
                "sentiment_score"
            ].mean()
            selected["sampling_target_mean"] = selected[
                "sentiment_score"
            ].mean()
            selected["sampling_method"] = "not_sampled"

        sampled_groups.append(selected)

    sampled = pd.concat(sampled_groups, ignore_index=True)

    # Preserve the selected article rows and sampling metadata.
    sampled.to_csv(
        OUTPUT_ARTICLES,
        index=False,
        encoding="utf-8-sig",
    )

    # Recalculate daily features using the retained articles.
    daily = (
        sampled.groupby("feature_date", as_index=False)
        .agg(
            article_count=("title", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            median_sentiment=("sentiment_score", "median"),
            positive_fraction=(
                "sentiment_label",
                lambda x: (x == "positive").mean(),
            ),
            negative_fraction=(
                "sentiment_label",
                lambda x: (x == "negative").mean(),
            ),
            neutral_fraction=(
                "sentiment_label",
                lambda x: (x == "neutral").mean(),
            ),
            sampling_sample_mean=("sampling_sample_mean", "first"),
            sampling_target_mean=("sampling_target_mean", "first"),
            sampling_method=("sampling_method", "first"),
        )
        .rename(columns={"feature_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )

    sampled_counts = sampled.groupby("feature_date").size()
    current_month_mask = sampled["feature_date"].str.startswith(current_month)
    current_month_counts = (
        sampled.loc[current_month_mask]
        .groupby("feature_date")
        .size()
    )

    print("\n===== DAILY SENTIMENT SAMPLING REPORT =====")
    print(f"Original unique articles: {len(df):,}")
    print(f"Sampled articles: {len(sampled):,}")
    print(f"Days represented: {len(daily):,}")
    print(f"Sample size: {ARTICLES_PER_SAMPLE}")
    print(f"Random draws per sampled day: {NUMBER_OF_SAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        "Current-month days exceeding 20 before sampling:",
        int(
            (
                original_counts[
                    original_counts.index.str.startswith(current_month)
                ] > ARTICLES_PER_SAMPLE
            ).sum()
        ),
    )
    print(
        "Maximum retained articles on a current-month day:",
        int(current_month_counts.max()) if not current_month_counts.empty else 0,
    )

    if not current_month_counts.empty:
        if (current_month_counts > ARTICLES_PER_SAMPLE).any():
            raise RuntimeError(
                "Sampling validation failed: current-month daily cap exceeded."
            )

    print(f"\nSampled article file:\n{OUTPUT_ARTICLES}")
    print(f"Daily sentiment file:\n{OUTPUT_DAILY}")
    print("\nSample daily rows:")
    print(daily.head().to_string(index=False))


if __name__ == "__main__":
    main()