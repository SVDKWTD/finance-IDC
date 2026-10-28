from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "news_market_alignment.csv"
OUTPUT_ARTICLES = PROJECT_DIR / "data" / "processed" / "news_sentiment_scored.csv"
OUTPUT_DAILY = PROJECT_DIR / "data" / "processed" / "news_sentiment_daily.csv"

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32

SENTIMENT_COLUMNS = [
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "sentiment_score",
    "sentiment_label",
]


def article_key(row):
    """
    Stable cache key: prefer URL, otherwise use normalized title + publication time.
    The fallback supports older aligned files that may lack a URL column.
    """
    url = str(row.get("url", "") or "").strip()
    if url and url.lower() != "nan":
        return "url:" + url

    title = str(row.get("title", "") or "").strip().casefold()
    published = str(row.get("published_at", "") or "").strip()
    return f"title:{title}|published:{published}"


def load_existing_scores():
    if not OUTPUT_ARTICLES.exists():
        return {}

    try:
        old = pd.read_csv(OUTPUT_ARTICLES)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"Could not read existing sentiment cache ({error}); rebuilding it.")
        return {}

    if not set(SENTIMENT_COLUMNS).issubset(old.columns):
        print("Existing scored file lacks expected sentiment columns; rebuilding cache.")
        return {}

    scores = {}
    for _, row in old.iterrows():
        key = article_key(row)
        if not key or key.startswith("title:|"):
            continue
        values = {}
        valid = True
        for column in SENTIMENT_COLUMNS:
            value = row[column]
            if pd.isna(value):
                valid = False
                break
            values[column] = value if column == "sentiment_label" else float(value)
        if valid:
            scores[key] = values
    return scores


def build_daily_summary(scored):
    daily = (
        scored.groupby("feature_date", as_index=False)
        .agg(
            article_count=("title", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            median_sentiment=("sentiment_score", "median"),
            positive_fraction=("sentiment_label", lambda x: (x == "positive").mean()),
            negative_fraction=("sentiment_label", lambda x: (x == "negative").mean()),
            neutral_fraction=("sentiment_label", lambda x: (x == "neutral").mean()),
        )
        .rename(columns={"feature_date": "date"})
    )
    return daily


def main():
    print("Loading aligned news data...")
    df = pd.read_csv(INPUT_FILE)

    required_columns = {"title", "feature_date"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["title", "feature_date"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["feature_date"] = df["feature_date"].astype(str).str.strip()
    df = df[df["title"] != ""].reset_index(drop=True)

    # Ensure each aligned article is represented once in the scored output.
    if "url" in df.columns:
        df["url"] = df["url"].fillna("").astype(str).str.strip()
        has_url = df["url"] != ""
        with_url = df[has_url].drop_duplicates(subset=["url"], keep="last")
        without_url = df[~has_url].drop_duplicates(
            subset=["title", "feature_date"], keep="last"
        )
        df = pd.concat([with_url, without_url], ignore_index=True)
    else:
        df = df.drop_duplicates(subset=["title", "feature_date"], keep="last")
    df = df.sort_values("feature_date").reset_index(drop=True)

    cached_scores = load_existing_scores()
    keys = [article_key(row) for _, row in df.iterrows()]
    missing_indices = [
        index for index, key in enumerate(keys)
        if key not in cached_scores
    ]

    print(f"Aligned unique headlines: {len(df):,}")
    print(f"Cached headline scores available: {len(cached_scores):,}")
    print(f"Headlines needing scoring: {len(missing_indices):,}")

    if missing_indices:
        device = 0 if torch.cuda.is_available() else -1
        print("Using GPU" if device == 0 else "Using CPU")
        print(f"Loading model: {MODEL_NAME}")
        sentiment_model = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=device,
            top_k=None,
            truncation=True,
            max_length=512,
        )

        total_batches = (len(missing_indices) + BATCH_SIZE - 1) // BATCH_SIZE
        for start in range(0, len(missing_indices), BATCH_SIZE):
            batch_indices = missing_indices[start:start + BATCH_SIZE]
            batch_titles = df.loc[batch_indices, "title"].tolist()
            predictions = sentiment_model(batch_titles, batch_size=BATCH_SIZE)

            for index, prediction in zip(batch_indices, predictions):
                scores = {
                    item["label"].lower(): float(item["score"])
                    for item in prediction
                }
                positive = scores.get("positive", 0.0)
                negative = scores.get("negative", 0.0)
                neutral = scores.get("neutral", 0.0)
                cached_scores[keys[index]] = {
                    "sentiment_positive": positive,
                    "sentiment_negative": negative,
                    "sentiment_neutral": neutral,
                    "sentiment_score": positive - negative,
                    "sentiment_label": max(scores, key=scores.get),
                }

            completed = min(start + BATCH_SIZE, len(missing_indices))
            print(
                f"Scored {completed:,}/{len(missing_indices):,} new headlines "
                f"(batch {start // BATCH_SIZE + 1}/{total_batches})"
            )

            # Checkpoint after each batch, including cached results, so an
            # interruption does not force completed scoring to be repeated.
            partial = df.iloc[:].copy()
            partial_results = [cached_scores.get(key) for key in keys]
            valid_positions = [
                i for i, result in enumerate(partial_results)
                if result is not None
            ]
            checkpoint = partial.iloc[valid_positions].reset_index(drop=True)
            checkpoint_scores = pd.DataFrame(
                [partial_results[i] for i in valid_positions]
            )
            checkpoint = pd.concat([checkpoint, checkpoint_scores], axis=1)
            checkpoint.to_csv(OUTPUT_ARTICLES, index=False, encoding="utf-8-sig")

    results = [cached_scores.get(key) for key in keys]
    if any(result is None for result in results):
        missing_count = sum(result is None for result in results)
        raise RuntimeError(
            f"{missing_count} headlines still lack sentiment scores; "
            "the output was not finalized."
        )

    scored = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    )
    scored.to_csv(OUTPUT_ARTICLES, index=False, encoding="utf-8-sig")
    print(f"\nArticle-level results saved to:\n{OUTPUT_ARTICLES}")
    print(f"Total scored articles: {len(scored):,}")

    daily = build_daily_summary(scored)
    daily.to_csv(OUTPUT_DAILY, index=False, encoding="utf-8-sig")
    print(f"Daily sentiment results saved to:\n{OUTPUT_DAILY}")
    print("\nSample daily sentiment rows:")
    print(daily.head().to_string(index=False))
    print("\nSentiment label distribution:")
    print(scored["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()
