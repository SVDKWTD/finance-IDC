from pathlib import Path

import pandas as pd
import torch
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "news_market_alignment.csv"
OUTPUT_ARTICLES = PROJECT_DIR / "data" / "processed" / "news_sentiment_scored.csv"
OUTPUT_DAILY = PROJECT_DIR / "data" / "processed" / "news_sentiment_daily.csv"

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32

SCORING_VERSION = "finbert_vader_textblob_v1"

SENTIMENT_COLUMNS = [
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "finbert_score",
    "vader_score",
    "textblob_score",
    "sentiment_score",
    "sentiment_label",
    "scoring_version",
]

vader_analyzer = SentimentIntensityAnalyzer()


def article_key(row):
    """Stable cache key: URL, or normalized title + publication time."""
    url = str(row.get("url", "") or "").strip()
    if url and url.lower() != "nan":
        return "url:" + url

    title = str(row.get("title", "") or "").strip().casefold()
    published = str(row.get("published_at", "") or "").strip()
    return f"title:{title}|published:{published}"


def calculate_vader_score(text):
    """VADER compound score is in the range [-1, 1]."""
    return float(vader_analyzer.polarity_scores(text)["compound"])


def calculate_textblob_score(text):
    """TextBlob polarity is in the range [-1, 1]."""
    return float(TextBlob(text).sentiment.polarity)


def ensemble_score(finbert_score, vader_score, textblob_score):
    """Equal-weight average of three scores on the same scale."""
    return (finbert_score + vader_score + textblob_score) / 3.0


def label_from_score(score):
    """Convert the ensemble score into a sentiment label."""
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def load_existing_scores():
    if not OUTPUT_ARTICLES.exists():
        return {}

    try:
        old = pd.read_csv(OUTPUT_ARTICLES)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"Could not read existing sentiment cache ({error}); rebuilding it.")
        return {}

    required_cache_columns = set(SENTIMENT_COLUMNS)
    if not required_cache_columns.issubset(old.columns):
        print("Existing cache uses an older scoring schema; rebuilding it.")
        return {}

    old = old[old["scoring_version"] == SCORING_VERSION]

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
            if column in ("sentiment_label", "scoring_version"):
                values[column] = str(value)
            else:
                values[column] = float(value)

        if valid:
            scores[key] = values

    return scores


def build_daily_summary(scored):
    """ISSUE 5 FIX: Filter NIFTY-tagged headlines for target sentiment aggregation."""
    if "market" in scored.columns:
        nifty_scored = scored[scored["market"] == "nifty"].copy()
        if nifty_scored.empty:
            nifty_scored = scored.copy()
    else:
        nifty_scored = scored.copy()

    daily = (
        nifty_scored.groupby("feature_date", as_index=False)
        .agg(
            article_count=("title", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            median_sentiment=("sentiment_score", "median"),
            mean_finbert_sentiment=("finbert_score", "mean"),
            mean_vader_sentiment=("vader_score", "mean"),
            mean_textblob_sentiment=("textblob_score", "mean"),
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
        .rename(columns={"feature_date": "date"})
    )
    return daily


def main():
    print("Loading aligned news data...")
    df = pd.read_csv(INPUT_FILE)

    required_columns = {"title", "feature_date"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["title", "feature_date"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["feature_date"] = df["feature_date"].astype(str).str.strip()
    df = df[df["title"] != ""].reset_index(drop=True)

    if "url" in df.columns:
        df["url"] = df["url"].fillna("").astype(str).str.strip()
        has_url = df["url"] != ""

        with_url = df[has_url].drop_duplicates(subset=["url"], keep="last")
        without_url = df[~has_url].drop_duplicates(
            subset=["title", "feature_date"], keep="last"
        )
        df = pd.concat([with_url, without_url], ignore_index=True)
    else:
        df = df.drop_duplicates(
            subset=["title", "feature_date"], keep="last"
        )

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

        total_batches = (
            len(missing_indices) + BATCH_SIZE - 1
        ) // BATCH_SIZE

        for start in range(0, len(missing_indices), BATCH_SIZE):
            batch_indices = missing_indices[start:start + BATCH_SIZE]
            batch_titles = df.loc[batch_indices, "title"].tolist()

            predictions = sentiment_model(
                batch_titles,
                batch_size=BATCH_SIZE,
            )

            for index, title, prediction in zip(
                batch_indices, batch_titles, predictions
            ):
                scores = {
                    item["label"].lower(): float(item["score"])
                    for item in prediction
                }

                finbert_positive = scores.get("positive", 0.0)
                finbert_negative = scores.get("negative", 0.0)
                finbert_neutral = scores.get("neutral", 0.0)

                finbert_score = finbert_positive - finbert_negative
                vader_score = calculate_vader_score(title)
                textblob_score = calculate_textblob_score(title)

                combined_score = ensemble_score(
                    finbert_score,
                    vader_score,
                    textblob_score,
                )

                cached_scores[keys[index]] = {
                    "finbert_positive": finbert_positive,
                    "finbert_negative": finbert_negative,
                    "finbert_neutral": finbert_neutral,
                    "finbert_score": finbert_score,
                    "vader_score": vader_score,
                    "textblob_score": textblob_score,
                    "sentiment_score": combined_score,
                    "sentiment_label": label_from_score(combined_score),
                    "scoring_version": SCORING_VERSION,
                }

            completed = min(start + BATCH_SIZE, len(missing_indices))
            print(
                f"Scored {completed:,}/{len(missing_indices):,} new headlines "
                f"(batch {start // BATCH_SIZE + 1}/{total_batches})"
            )

            partial_results = [cached_scores.get(key) for key in keys]
            valid_positions = [
                i for i, result in enumerate(partial_results)
                if result is not None
            ]

            checkpoint = df.iloc[valid_positions].reset_index(drop=True)
            checkpoint_scores = pd.DataFrame(
                [partial_results[i] for i in valid_positions]
            )
            checkpoint = pd.concat(
                [checkpoint, checkpoint_scores], axis=1
            )
            checkpoint.to_csv(
                OUTPUT_ARTICLES,
                index=False,
                encoding="utf-8-sig",
            )

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
    scored.to_csv(
        OUTPUT_ARTICLES,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArticle-level results saved to:\n{OUTPUT_ARTICLES}")
    print(f"Total scored articles: {len(scored):,}")

    daily = build_daily_summary(scored)
    daily.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Daily sentiment results saved to:\n{OUTPUT_DAILY}")
    print("\nSample daily sentiment rows:")
    print(daily.head().to_string(index=False))
    print("\nEnsemble sentiment label distribution:")
    print(scored["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()