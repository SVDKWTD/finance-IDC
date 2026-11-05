from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    brier_score_loss,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

INPUT_ARTICLES = DATA / "news_sentiment_scored.csv"
MARKET_FILE = DATA / "news_market_daily.csv"
ORIGINAL_MODEL_FILE = DATA / "model_dataset.csv"

# ISSUE 6 FIX: Cap per trading session to avoid calendar-month lookahead leakage
CAP_PER_DAY = 25
RANDOM_SEED = 42
SAMPLE_YEAR = 2026

SAMPLED_ARTICLES = (
    DATA / "news_sentiment_sampled_monthly_2026.csv"
)
SAMPLED_DAILY = (
    DATA / "news_sentiment_daily_sampled_monthly_2026.csv"
)
SAMPLED_MODEL = (
    DATA / "model_dataset_sampled_monthly_2026.csv"
)

RETURN_RESULTS = DATA / "monthly_sampling_return_comparison.csv"
DIRECTION_RESULTS = DATA / "monthly_sampling_direction_comparison.csv"
PAIRED_RESULTS = DATA / "monthly_sampling_paired_predictions.csv"

MARKET_FEATURES = [
    "daily_return_pct",
    "return_lag1",
    "return_lag2",
    "return_lag5",
    "volatility_5",
    "article_count",
    "unique_sources",
]

SENTIMENT_FEATURES = MARKET_FEATURES + [
    "mean_sentiment",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
    "sentiment_lag1",
]

TARGET = "target_next_session_return_pct"


# ============================================================
# HELPERS
# ============================================================

def find_source_column(df):
    candidates = [
        "source",
        "source_name",
        "publisher",
        "news_source",
        "domain",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def extract_domain(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan

    value = str(value).strip()

    if "://" not in value:
        value = "https://" + value

    try:
        hostname = urlparse(value).hostname
        if hostname:
            return hostname.lower().removeprefix("www.")
    except ValueError:
        pass

    return np.nan


# ============================================================
# 1. SAMPLE ARTICLES BY TRADING SESSION (DAILY CAP)
# ============================================================

def sample_articles():
    if not INPUT_ARTICLES.exists():
        raise FileNotFoundError(
            f"Scored article file not found: {INPUT_ARTICLES}"
        )

    print(f"Loading scored articles:\n{INPUT_ARTICLES}")
    df = pd.read_csv(INPUT_ARTICLES)

    required = {
        "feature_date",
        "title",
        "sentiment_score",
        "sentiment_label",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing from scored articles: "
            f"{sorted(missing)}"
        )

    df["feature_date"] = pd.to_datetime(
        df["feature_date"], errors="coerce"
    )
    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"], errors="coerce"
    )
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["sentiment_label"] = (
        df["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
    )

    invalid = (
        df["feature_date"].isna()
        | df["sentiment_score"].isna()
        | (df["title"] == "")
        | (~df["sentiment_label"].isin(
            ["positive", "negative", "neutral"]
        ))
    )

    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} rows have invalid dates, titles, "
            "scores, or sentiment labels."
        )

    # ISSUE 5 FIX: Filter NIFTY headlines to avoid blending external assets
    if "market" in df.columns:
        df = df[df["market"] == "nifty"].copy()

    original_count = len(df)

    historical = df[df["feature_date"].dt.year != SAMPLE_YEAR].copy()
    recent = df[df["feature_date"].dt.year == SAMPLE_YEAR].copy()

    parts = [historical]

    # ISSUE 6 FIX: Sample per trading day/session independently
    for date_val, group in recent.groupby(recent["feature_date"].dt.date, sort=True):
        before = len(group)
        if before > CAP_PER_DAY:
            day_seed = RANDOM_SEED + int(pd.Timestamp(date_val).to_julian_date())
            selected = group.sample(
                n=CAP_PER_DAY,
                replace=False,
                random_state=day_seed % (2**32),
            )
        else:
            selected = group.copy()

        parts.append(selected)

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sort_values(
        ["feature_date", "title"]
    ).reset_index(drop=True)

    source_column = find_source_column(sampled)

    if source_column is not None:
        sampled["_source_key"] = (
            sampled[source_column].fillna("").astype(str).str.strip().str.lower()
        )
        sampled.loc[sampled["_source_key"] == "", "_source_key"] = np.nan
    elif "url" in sampled.columns:
        sampled["_source_key"] = sampled["url"].map(extract_domain)
    else:
        raise ValueError(
            "Cannot calculate sampled unique_sources: no source or URL column exists."
        )

    output_articles = sampled.drop(columns=["_source_key"]).copy()
    output_articles["feature_date"] = (
        output_articles["feature_date"].dt.strftime("%Y-%m-%d")
    )
    output_articles.to_csv(
        SAMPLED_ARTICLES,
        index=False,
        encoding="utf-8-sig",
    )

    daily = (
        sampled.groupby(sampled["feature_date"].dt.strftime("%Y-%m-%d"))
        .agg(
            article_count=("title", "size"),
            unique_sources=("_source_key", "nunique"),
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
        SAMPLED_DAILY,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== DAILY PER-SESSION SAMPLING REPORT =====")
    print(f"Scored articles before sampling: {original_count}")
    print(f"Articles after sampling: {len(sampled)}")
    print(f"Articles removed: {original_count - len(sampled)}")
    print(f"2021-2025 articles retained: {len(historical)}")
    print(f"2026 articles before sampling: {len(recent)}")
    print(f"Per-session daily cap: {CAP_PER_DAY}")

    return daily


# ============================================================
# 2. BUILD SAMPLED MODEL DATASET
# ============================================================

def build_sampled_model_dataset(sampled_daily):
    if not MARKET_FILE.exists():
        raise FileNotFoundError(
            f"Market daily file not found: {MARKET_FILE}"
        )

    market = pd.read_csv(MARKET_FILE)

    if "date" not in market.columns:
        raise ValueError("Market daily file has no 'date' column.")

    market["date"] = pd.to_datetime(
        market["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    sampled_daily = sampled_daily.copy()
    sampled_daily["date"] = pd.to_datetime(
        sampled_daily["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    market = market.drop(
        columns=["article_count", "unique_sources"],
        errors="ignore",
    )

    sentiment_columns = [
        "date",
        "article_count",
        "unique_sources",
        "mean_sentiment",
        "median_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    ]

    merged = market.merge(
        sampled_daily[sentiment_columns],
        on="date",
        how="left",
        validate="one_to_one",
    )

    for column in ["article_count", "unique_sources"]:
        merged[column] = merged[column].fillna(0)

    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(
        SAMPLED_MODEL,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved sampled model dataset to:\n{SAMPLED_MODEL}")
    return merged


# ============================================================
# 3. WALK-FORWARD EVALUATION & METRICS
# ============================================================

def evaluate_dataset(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    start = int(len(df) * 0.70)
    records = []

    for i in range(start, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[[i]].copy()

        train = train.dropna(subset=[TARGET])
        y_train = train[TARGET].astype(float)
        y_test = float(test[TARGET].iloc[0])

        predictions = {
            "Mean baseline": float(y_train.mean())
        }

        for name, features in [
            ("Market + news volume", MARKET_FEATURES),
            ("Market + news + sentiment", SENTIMENT_FEATURES),
        ]:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=10.0),
            )
            model.fit(train[features], y_train)
            predictions[name] = float(
                model.predict(test[features])[0]
            )

        y_direction = (y_train > 0).astype(int)

        record = {
            "date": test["date"].iloc[0],
            "target_next_session_date": test[
                "target_next_session_date"
            ].iloc[0],
            "actual_return_pct": y_test,
            "actual_up": int(y_test > 0),
            "mean_baseline_prediction_pct": predictions["Mean baseline"],
            "market_prediction_pct": predictions["Market + news volume"],
            "sentiment_prediction_pct": predictions[
                "Market + news + sentiment"
            ],
        }

        for name, features in [
            ("market", MARKET_FEATURES),
            ("sentiment", SENTIMENT_FEATURES),
        ]:
            if y_direction.nunique() == 2:
                classifier = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1000,
                        class_weight=None,
                    ),
                )
                classifier.fit(train[features], y_direction)
                p_up = float(
                    classifier.predict_proba(test[features])[0, 1]
                )
            else:
                p_up = float(y_direction.mean())

            record[f"{name}_p_up"] = p_up
            record[f"{name}_predicted_up"] = int(p_up >= 0.5)
            record[f"{name}_confidence"] = max(p_up, 1 - p_up)

        records.append(record)

    return pd.DataFrame(records)


def calculate_return_metrics(pred, dataset_name):
    models = [
        ("Mean baseline", "mean_baseline_prediction_pct"),
        ("Market + news volume", "market_prediction_pct"),
        ("Market + news + sentiment", "sentiment_prediction_pct"),
    ]

    rows = []

    for name, column in models:
        actual = pred["actual_return_pct"]
        estimate = pred[column]

        rows.append({
            "dataset": dataset_name,
            "model": name,
            "MAE_pct": mean_absolute_error(actual, estimate),
            "RMSE_pct": np.sqrt(
                mean_squared_error(actual, estimate)
            ),
            "R2": r2_score(actual, estimate),
        })

    return rows


def calculate_direction_metrics(pred, dataset_name):
    rows = []

    for name in ["market", "sentiment"]:
        probabilities = pred[f"{name}_p_up"]
        actual = pred["actual_up"]
        high_confidence = pred[f"{name}_confidence"] >= 0.60

        rows.append({
            "dataset": dataset_name,
            "model": name,
            "accuracy": accuracy_score(
                actual,
                pred[f"{name}_predicted_up"],
            ),
            "brier_score": brier_score_loss(
                actual,
                probabilities,
            ),
            "high_confidence_count": int(high_confidence.sum()),
            "high_confidence_accuracy": (
                accuracy_score(
                    actual[high_confidence],
                    pred.loc[
                        high_confidence,
                        f"{name}_predicted_up",
                    ],
                )
                if high_confidence.any()
                else np.nan
            ),
        })

    return rows


def main():
    if not ORIGINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Original model dataset not found: {ORIGINAL_MODEL_FILE}"
        )

    sampled_daily = sample_articles()
    build_sampled_model_dataset(sampled_daily)

    print("\nRunning original walk-forward evaluation...")
    original = evaluate_dataset(ORIGINAL_MODEL_FILE)

    print("Running sampled walk-forward evaluation...")
    sampled = evaluate_dataset(SAMPLED_MODEL)

    paired = original.merge(
        sampled,
        on=["date", "target_next_session_date"],
        suffixes=("_original", "_sampled"),
        validate="one_to_one",
        how="inner",
    )

    if paired.empty:
        raise ValueError(
            "The original and sampled datasets have no shared forecast dates."
        )

    original_paired = paired[
        [c for c in paired.columns if c.endswith("_original")]
    ].copy()

    sampled_paired = paired[
        [c for c in paired.columns if c.endswith("_sampled")]
    ].copy()

    original_paired.columns = [
        c.removesuffix("_original") for c in original_paired.columns
    ]
    sampled_paired.columns = [
        c.removesuffix("_sampled") for c in sampled_paired.columns
    ]

    returns = pd.DataFrame(
        calculate_return_metrics(original_paired, "Original")
        + calculate_return_metrics(sampled_paired, "Per-session sampled")
    )

    directions = pd.DataFrame(
        calculate_direction_metrics(original_paired, "Original")
        + calculate_direction_metrics(sampled_paired, "Per-session sampled")
    )

    returns.to_csv(RETURN_RESULTS, index=False)
    directions.to_csv(DIRECTION_RESULTS, index=False)
    paired.to_csv(PAIRED_RESULTS, index=False)

    print("\n===== RETURN COMPARISON =====")
    print(returns.round(4).to_string(index=False))

    print("\n===== DIRECTION COMPARISON =====")
    print(directions.round(4).to_string(index=False))


if __name__ == "__main__":
    main()