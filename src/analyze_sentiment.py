
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_FILE = Path("data/processed/news_processed.csv")
OUTPUT_FILE = Path("data/processed/news_sentiment.csv")

import re


GENERIC_PATTERNS = [
    r"^business news:.*-\s*moneycontrol\.com$",
    r"^stock market:.*-\s*moneycontrol\.com$",
    r"^nse\s*[-–:].*-\s*nse india$",
    r"^share market live,.*-\s*the economic times$",
    r"^investing\.com india\s*-\s*financial news",
    r"^indian stock market: exchanges and indexes\s*-\s*investopedia$",
]


def is_generic_headline(title):
    text = str(title).strip().lower()

    if not text:
        return True

    return any(
        re.search(pattern, text)
        for pattern in GENERIC_PATTERNS
    )


def classify_sentiment(title, score):
    if is_generic_headline(title):
        return "unclassified"

    if score >= 0.20:
        return "positive"
    if score <= -0.20:
        return "negative"

    return "neutral"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Run process_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    if "title" not in df.columns:
        raise ValueError("News dataset has no 'title' column.")

    analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiment for each headline.
    scores = df["title"].fillna("").astype(str).apply(
        analyzer.polarity_scores
    )

    score_df = pd.DataFrame(scores.tolist(), index=df.index)

    df["sentiment_positive"] = score_df["pos"]
    df["sentiment_neutral"] = score_df["neu"]
    df["sentiment_negative"] = score_df["neg"]
    df["sentiment_compound"] = score_df["compound"]

    # Simple descriptive label, not a financial recommendation.
    df["sentiment_label"] = [
    classify_sentiment(title, score)
    for title, score in zip(
        df["title"],
        df["sentiment_compound"]
    )]


    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Sentiment analysis complete.")
    print("Articles analysed:", len(df))
    print("\nSentiment distribution:")
    print(df["sentiment_label"].value_counts())
    print("\nSample results:")
    print(
        df[
            ["title", "sentiment_compound", "sentiment_label"]
        ].head(10).to_string(index=False)
    )
    print("\nSaved to:", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
