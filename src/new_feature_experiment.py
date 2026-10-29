
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

MODEL_FILE = DATA_DIR / "model_dataset.csv"
SCORED_NEWS_FILE = DATA_DIR / "news_sentiment_sampled_monthly_2026.csv"

INITIAL_TRAIN_FRACTION = 0.70
RIDGE_ALPHA = 10.0
RANDOM_STATE = 42

# Existing predictors: no PCA-transformed data is used.
MARKET_CORE = [
    "daily_return_pct",
    "return_lag1",
    "return_lag2",
    "return_lag5",
    "volatility_5",
]

NEWS_VOLUME = [
    "article_count",
    "unique_sources",
]

SENTIMENT = [
    "mean_sentiment",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
    "sentiment_lag1",
]

BASE_FEATURES = MARKET_CORE + NEWS_VOLUME + SENTIMENT


# =========================================================
# LOAD DATA
# =========================================================

def load_model_data():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {MODEL_FILE}")

    df = pd.read_csv(MODEL_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = [
        "date",
        "daily_return_pct",
        "article_count",
        "unique_sources",
        "mean_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
        "target_next_session_return_pct",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in model_dataset.csv: {missing}")

    # Recreate original predictors consistently.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    return df


# =========================================================
# ARTICLE-LEVEL SENTIMENT DISPERSION
# =========================================================

def add_sentiment_dispersion(df):
    """
    Try to calculate daily article-level sentiment dispersion.

    Uses the sampled article-level sentiment file, if available.
    Numeric sentiment scores are preferred. If only sentiment labels
    exist, labels are mapped to positive=1, neutral=0, negative=-1.

    If no usable article-level sentiment field exists, the experiment
    continues without dispersion and prints a warning.
    """

    if not SCORED_NEWS_FILE.exists():
        print(
            "\nWARNING: Article-level scored-news file not found. "
            "Sentiment dispersion will be skipped."
        )
        df["sentiment_dispersion"] = np.nan
        return df, False

    news = pd.read_csv(SCORED_NEWS_FILE)

    date_col = next(
        (
            c for c in [
                "date", "published_date", "published_at",
                "publication_date", "datetime"
            ]
            if c in news.columns
        ),
        None,
    )

    if date_col is None:
        print(
            "\nWARNING: No recognized date column in scored-news file. "
            "Sentiment dispersion will be skipped."
        )
        df["sentiment_dispersion"] = np.nan
        return df, False

    news["date"] = pd.to_datetime(
        news[date_col], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()

    numeric_score_col = next(
        (
            c for c in [
                "sentiment_score", "compound", "compound_score",
                "sentiment_compound", "score"
            ]
            if c in news.columns
            and pd.api.types.is_numeric_dtype(news[c])
        ),
        None,
    )

    if numeric_score_col is not None:
        news["_sentiment_value"] = pd.to_numeric(
            news[numeric_score_col], errors="coerce"
        )
        score_description = f"numeric scores from {numeric_score_col}"

    else:
        label_col = next(
            (
                c for c in [
                    "sentiment_label", "sentiment_class",
                    "sentiment", "label"
                ]
                if c in news.columns
            ),
            None,
        )

        if label_col is None:
            print(
                "\nWARNING: No recognized article sentiment score or label "
                "found. Sentiment dispersion will be skipped."
            )
            df["sentiment_dispersion"] = np.nan
            return df, False

        labels = news[label_col].astype(str).str.lower().str.strip()
        mapping = {
            "positive": 1.0,
            "pos": 1.0,
            "neutral": 0.0,
            "neu": 0.0,
            "negative": -1.0,
            "neg": -1.0,
        }

        news["_sentiment_value"] = labels.map(mapping)
        score_description = (
            f"numeric mapping of labels from {label_col}"
        )

    news = news.dropna(subset=["date", "_sentiment_value"])

    # Daily standard deviation across individual articles.
    dispersion = (
        news.groupby("date")["_sentiment_value"]
        .std()
        .rename("sentiment_dispersion")
        .reset_index()
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.merge(dispersion, on="date", how="left")

    print(
        f"\nSentiment dispersion calculated using {score_description}."
    )
    print(f"Article-level rows used: {len(news)}")
    print(
        "Note: label-based dispersion is approximate, not equivalent "
        "to dispersion of continuous sentiment scores."
    )

    return df, True


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def engineer_features(df):
    df = df.sort_values("date").reset_index(drop=True).copy()

    # A. Market momentum.
    # Product of daily gross returns gives cumulative return.
    daily_return_decimal = df["daily_return_pct"] / 100.0

    for window in [3, 5, 10]:
        gross_return = (
            1 + daily_return_decimal
        ).rolling(window).apply(np.prod, raw=True)

        df[f"momentum_{window}"] = (gross_return - 1) * 100

    # B. Longer volatility regimes.
    for window in [10, 20]:
        df[f"volatility_{window}"] = (
            df["daily_return_pct"].rolling(window).std()
        )

    # C. News-volume dynamics.
    # log1p is robust when article counts are zero.
    safe_count = df["article_count"].clip(lower=0)

    df["news_log_count"] = np.log1p(safe_count)

    df["news_log_change"] = (
        df["news_log_count"] - df["news_log_count"].shift(1)
    )

    for window in [5, 10]:
        trailing_mean = (
            df["article_count"].shift(1).rolling(window).mean()
        )
        df[f"news_volume_ratio_{window}"] = (
            df["article_count"] / trailing_mean.replace(0, np.nan)
        )

    # D. Sentiment dynamics.
    df["sentiment_change"] = (
        df["mean_sentiment"] - df["mean_sentiment"].shift(1)
    )

    df["sentiment_trend_5"] = (
        df["mean_sentiment"]
        - df["mean_sentiment"].shift(5)
    )

    df["sentiment_mean_5"] = (
        df["mean_sentiment"].rolling(5).mean()
    )

    # E. Sentiment dispersion is merged from article-level data above.
    # The model will use it only if valid observations are available.

    momentum_features = [
        "momentum_3", "momentum_5", "momentum_10"
    ]

    volatility_features = [
        "volatility_10", "volatility_20"
    ]

    news_dynamics_features = [
        "news_log_count",
        "news_log_change",
        "news_volume_ratio_5",
        "news_volume_ratio_10",
    ]

    sentiment_dynamics_features = [
        "sentiment_change",
        "sentiment_trend_5",
        "sentiment_mean_5",
    ]

    return (
        df,
        momentum_features,
        volatility_features,
        news_dynamics_features,
        sentiment_dynamics_features,
    )


# =========================================================
# WALK-FORWARD EVALUATION
# =========================================================

def evaluate_model(df, variant, features):
    """
    Expanding-window walk-forward evaluation.
    Scaling is fitted on training rows only.
    No PCA is performed.
    """

    X = df[features].to_numpy(dtype=float)
    y_return = df["target_next_session_return_pct"].to_numpy(dtype=float)
    y_direction = (y_return > 0).astype(int)
    dates = df["date"].to_numpy()

    split = int(len(df) * INITIAL_TRAIN_FRACTION)
    predictions = []

    for test_idx in range(split, len(df)):
        X_train = X[:test_idx]
        X_test = X[test_idx:test_idx + 1]

        yr_train = y_return[:test_idx]
        yd_train = y_direction[:test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train_scaled, yr_train)
        predicted_return = float(ridge.predict(X_test_scaled)[0])

        logistic = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        logistic.fit(X_train_scaled, yd_train)

        classes = list(logistic.classes_)
        if 1 in classes:
            p_up = float(
                logistic.predict_proba(X_test_scaled)[0, classes.index(1)]
            )
        else:
            p_up = 0.0

        predictions.append({
            "date": pd.Timestamp(dates[test_idx]),
            "variant": variant,
            "n_features": len(features),
            "actual_return_pct": y_return[test_idx],
            "predicted_return_pct": predicted_return,
            "actual_direction": y_direction[test_idx],
            "predicted_direction": int(p_up >= 0.5),
            "probability_up": p_up,
        })

        if (test_idx - split + 1) % 100 == 0:
            print(
                f"  {variant}: forecasted "
                f"{test_idx - split + 1} of {len(df) - split}"
            )

    return pd.DataFrame(predictions)


def evaluate_mean_baseline(df):
    y = df["target_next_session_return_pct"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    split = int(len(df) * INITIAL_TRAIN_FRACTION)

    rows = []

    for test_idx in range(split, len(df)):
        rows.append({
            "date": pd.Timestamp(dates[test_idx]),
            "variant": "Mean baseline",
            "n_features": 0,
            "actual_return_pct": y[test_idx],
            "predicted_return_pct": float(np.mean(y[:test_idx])),
            "actual_direction": int(y[test_idx] > 0),
            "predicted_direction": np.nan,
            "probability_up": np.nan,
        })

    return pd.DataFrame(rows)


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(predictions):
    return_rows = []
    direction_rows = []

    for variant, group in predictions.groupby("variant", sort=False):
        actual = group["actual_return_pct"].to_numpy(dtype=float)
        predicted = group["predicted_return_pct"].to_numpy(dtype=float)

        return_rows.append({
            "variant": variant,
            "n_features": int(group["n_features"].iloc[0]),
            "n_forecasts": len(group),
            "MAE_pct": mean_absolute_error(actual, predicted),
            "RMSE_pct": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
        })

        if group["probability_up"].isna().any():
            continue

        actual_dir = group["actual_direction"].to_numpy(dtype=int)
        predicted_dir = group["predicted_direction"].to_numpy(dtype=int)
        probabilities = group["probability_up"].to_numpy(dtype=float)

        high = (
            (probabilities >= 0.60)
            | (probabilities <= 0.40)
        )

        direction_rows.append({
            "variant": variant,
            "n_features": int(group["n_features"].iloc[0]),
            "n_forecasts": len(group),
            "accuracy": accuracy_score(actual_dir, predicted_dir),
            "brier_score": brier_score_loss(actual_dir, probabilities),
            "high_confidence_count": int(high.sum()),
            "high_confidence_accuracy": (
                accuracy_score(actual_dir[high], predicted_dir[high])
                if high.any() else np.nan
            ),
        })

    return (
        pd.DataFrame(return_rows).sort_values("MAE_pct"),
        pd.DataFrame(direction_rows).sort_values(
            ["brier_score", "accuracy"],
            ascending=[True, False],
        ),
    )


# =========================================================
# MAIN
# =========================================================

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_model_data()
    df, has_dispersion = add_sentiment_dispersion(df)

    (
        df,
        momentum_features,
        volatility_features,
        news_dynamics_features,
        sentiment_dynamics_features,
    ) = engineer_features(df)

    dispersion_features = (
        ["sentiment_dispersion"]
        if has_dispersion
        else []
    )

    feature_groups = {
        "Original features": BASE_FEATURES,
        "Original + momentum": BASE_FEATURES + momentum_features,
        "Original + volatility regimes": BASE_FEATURES + volatility_features,
        "Original + news dynamics": BASE_FEATURES + news_dynamics_features,
        "Original + sentiment dynamics": (
            BASE_FEATURES + sentiment_dynamics_features
        ),
        "Original + sentiment dispersion": (
            BASE_FEATURES + dispersion_features
        ) if dispersion_features else None,
        "Original + all new features": (
            BASE_FEATURES
            + momentum_features
            + volatility_features
            + news_dynamics_features
            + sentiment_dynamics_features
            + dispersion_features
        ),
    }

    # Remove unavailable optional variant and duplicate feature names.
    feature_groups = {
        name: list(dict.fromkeys(features))
        for name, features in feature_groups.items()
        if features is not None and features
    }

    # Drop rows missing target or any feature needed by any model.
    all_features = list(dict.fromkeys(
        feature
        for features in feature_groups.values()
        for feature in features
    ))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=all_features + ["target_next_session_return_pct"]
    ).reset_index(drop=True)

    if len(df) < 200:
        raise ValueError(
            f"Only {len(df)} rows remain after feature engineering: {len(df)}"
        )

    print("\n===== NEW FEATURE EXPERIMENT =====")
    print(f"Input: {MODEL_FILE}")
    print(f"Usable rows after feature engineering: {len(df)}")
    print(f"Initial training fraction: {INITIAL_TRAIN_FRACTION:.0%}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Article-level sentiment dispersion available: {has_dispersion}")
    print("\nFeature groups:")

    for name, features in feature_groups.items():
        print(f"  {name}: {len(features)} predictors")

    predictions_list = [evaluate_mean_baseline(df)]

    for name, features in feature_groups.items():
        print(f"\nEvaluating: {name}")
        predictions_list.append(
            evaluate_model(df, name, features)
        )

    predictions = pd.concat(predictions_list, ignore_index=True)

    # Ensure all models use the same forecast dates and targets.
    reference = (
        predictions[predictions["variant"] == "Original features"]
        [["date", "actual_return_pct"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    for name, group in predictions.groupby("variant"):
        check = (
            group[["date", "actual_return_pct"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if not check["date"].equals(reference["date"]):
            raise RuntimeError(f"Forecast dates differ for {name}")

        if not np.allclose(
            check["actual_return_pct"],
            reference["actual_return_pct"],
        ):
            raise RuntimeError(f"Actual targets differ for {name}")

    return_results, direction_results = calculate_metrics(predictions)

    return_file = DATA_DIR / "new_feature_return_results.csv"
    direction_file = DATA_DIR / "new_feature_direction_results.csv"
    predictions_file = DATA_DIR / "new_feature_predictions.csv"

    return_results.to_csv(return_file, index=False)
    direction_results.to_csv(direction_file, index=False)
    predictions.to_csv(predictions_file, index=False)

    print("\n===== RETURN METRICS (LOWER MAE/RMSE IS BETTER) =====")
    print(
        return_results.to_string(
            index=False,
            formatters={
                "MAE_pct": "{:.4f}".format,
                "RMSE_pct": "{:.4f}".format,
                "R2": "{:.4f}".format,
            },
        )
    )

    print("\n===== DIRECTION METRICS =====")
    print(
        direction_results.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "high_confidence_accuracy": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
            },
        )
    )

    print("\n===== SAVED FILES =====")
    print(return_file)
    print(direction_file)
    print(predictions_file)

    print(
        "\nIMPORTANT: The same complete-case dates are used for every model. "
        "This makes comparisons fair, but missing sentiment-dispersion "
        "observations can reduce the sample size."
    )
    print(
        "All features must be available by the forecast cutoff. "
        "If daily news is not complete by that cutoff, shift news features "
        "back one session before interpreting results as tradable."
    )


if __name__ == "__main__":
    main()
