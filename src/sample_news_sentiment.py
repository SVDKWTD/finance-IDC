
from pathlib import Path

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

MAX_ARTICLES_PER_DAY = 250
RANDOM_SEED = 42


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
    df = df[df["title"] != ""].copy()

    # Remove duplicate articles before sampling.
    if "url" in df.columns:
        df["url"] = df["url"].fillna("").astype(str).str.strip()
        has_url = df["url"] != ""

        with_url = df[has_url].drop_duplicates(
            subset=["url"], keep="last"
        )
        without_url = df[~has_url].drop_duplicates(
            subset=["title", "feature_date"], keep="last"
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

    original_counts = df.groupby("feature_date").size()

    # Sample each day separately. This avoids pandas GroupBy.apply
    # Preserve all historical articles.
    # Sample only high-volume days in 2026.
    sampled_groups = []

    for feature_date, group in df.groupby("feature_date", sort=True):
        if feature_date >= "2026-01-01" and len(group) > MAX_ARTICLES_PER_DAY:
            sampled_group = group.sample(
                n=MAX_ARTICLES_PER_DAY,
                replace=False,
                random_state=RANDOM_SEED,
            )
        else:
            sampled_group = group.copy()

        sampled_groups.append(sampled_group)

    sampled = pd.concat(
        sampled_groups,
        ignore_index=True,
    )
 # Save article-level sample separately; original files are untouched.
    sampled.to_csv(
        OUTPUT_ARTICLES,
        index=False,
        encoding="utf-8-sig",
    )

    # Recalculate daily sentiment using sampled articles only.
    daily = (
        sampled.groupby("feature_date")
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
        )
        .reset_index()
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

    print("\n===== DAILY SENTIMENT SAMPLING REPORT =====")
    print(f"Original unique articles: {len(df):,}")
    print(f"Sampled articles: {len(sampled):,}")
    print(f"Days represented: {len(daily):,}")
    print(f"Cap per day: {MAX_ARTICLES_PER_DAY}")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        "Days exceeding cap before sampling:",
        int((original_counts > MAX_ARTICLES_PER_DAY).sum()),
    )
    print(
        "Maximum sampled articles in any day:",
        int(sampled_counts.max()),
    )
    print(
        "Days with fewer than cap:",
        int((sampled_counts < MAX_ARTICLES_PER_DAY).sum()),
    )

    # Confirm the cap was respected.
    if (sampled_counts > MAX_ARTICLES_PER_DAY).any():
        raise RuntimeError("Sampling validation failed: daily cap exceeded.")

    print(f"\nSampled article file:\n{OUTPUT_ARTICLES}")
    print(f"Daily sentiment file:\n{OUTPUT_DAILY}")
    print("\nSample daily rows:")
    print(daily.head().to_string(index=False))


if __name__ == "__main__":
    main()
