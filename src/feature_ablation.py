
from pathlib import Path

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

INPUT_FILE = DATA / "model_dataset.csv"

RETURN_RESULTS = DATA / "feature_ablation_return_results.csv"
DIRECTION_RESULTS = DATA / "feature_ablation_direction_results.csv"
PREDICTIONS_FILE = DATA / "feature_ablation_predictions.csv"

TARGET = "target_next_session_return_pct"
INITIAL_TRAIN_FRACTION = 0.70
RIDGE_ALPHA = 10.0
HIGH_CONFIDENCE_THRESHOLD = 0.60

# Market behaviour, without news features.
MARKET_CORE = [
    "daily_return_pct",
    "return_lag1",
    "return_lag2",
    "return_lag5",
    "volatility_5",
]

# News-volume features calculated from the monthly-sampled corpus.
NEWS_VOLUME = [
    "article_count",
    "unique_sources",
]

# Sentiment features.
SENTIMENT = [
    "mean_sentiment",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
    "sentiment_lag1",
]

FULL_FEATURES = MARKET_CORE + NEWS_VOLUME + SENTIMENT


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {INPUT_FILE}\n"
            "Run src\\build_model_dataset.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = {
        "date",
        "target_next_session_date",
        TARGET,
        "daily_return_pct",
        "article_count",
        "unique_sources",
        "mean_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Model dataset is missing columns: {sorted(missing)}"
        )

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in the model dataset.")

    # These features use information available by the forecast origin.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    # Match the existing evaluation: remove rows with no target.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    return df


# ============================================================
# DEFINE ABLATION VARIANTS
# ============================================================

def make_variants():
    variants = {
        "Mean baseline": [],
        "Market core only": MARKET_CORE,
        "Market + news volume": MARKET_CORE + NEWS_VOLUME,
        "Market + sentiment": MARKET_CORE + SENTIMENT,
        "Full model": FULL_FEATURES,
        "Full model without neutral_fraction": [
            f for f in FULL_FEATURES if f != "neutral_fraction"
        ],
    }

    # Remove one feature at a time from the full model.
    # This measures each feature's incremental contribution
    # conditional on the other features being present.
    for feature in FULL_FEATURES:
        variants[f"Full minus {feature}"] = [
            f for f in FULL_FEATURES if f != feature
        ]

    # Also test removing entire groups from the full model.
    variants["Full minus news volume"] = MARKET_CORE + SENTIMENT
    variants["Full minus sentiment"] = MARKET_CORE + NEWS_VOLUME
    variants["Full minus market core"] = NEWS_VOLUME + SENTIMENT

    return variants


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def evaluate_variant(df, variant_name, features):
    start = int(len(df) * INITIAL_TRAIN_FRACTION)
    records = []

    for i in range(start, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[[i]].copy()

        train = train.dropna(subset=[TARGET])

        y_train = train[TARGET].astype(float)
        y_test = float(test[TARGET].iloc[0])
        actual_up = int(y_test > 0)

        record = {
            "variant": variant_name,
            "date": test["date"].iloc[0],
            "target_next_session_date": test[
                "target_next_session_date"
            ].iloc[0],
            "actual_return_pct": y_test,
            "actual_up": actual_up,
        }

        # Historical-mean return baseline.
        if variant_name == "Mean baseline":
            record["return_prediction_pct"] = float(y_train.mean())
            records.append(record)
            continue

        # Ridge regression for next-session return.
        regressor = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        )

        regressor.fit(train[features], y_train)
        record["return_prediction_pct"] = float(
            regressor.predict(test[features])[0]
        )

        # Logistic regression for probability of an up session.
        y_direction = (y_train > 0).astype(int)

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

        record["p_up"] = p_up
        record["predicted_up"] = int(p_up >= 0.5)
        record["confidence"] = max(p_up, 1 - p_up)

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(predictions, variant_name, features):
    actual_return = predictions["actual_return_pct"].to_numpy()
    predicted_return = predictions["return_prediction_pct"].to_numpy()

    return_row = {
        "variant": variant_name,
        "n_features": len(features),
        "features": ", ".join(features) if features else "None",
        "MAE_pct": mean_absolute_error(
            actual_return, predicted_return
        ),
        "RMSE_pct": np.sqrt(
            mean_squared_error(actual_return, predicted_return)
        ),
        "R2": r2_score(actual_return, predicted_return),
    }

    direction_row = {
        "variant": variant_name,
        "n_features": len(features),
        "accuracy": np.nan,
        "brier_score": np.nan,
        "high_confidence_count": np.nan,
        "high_confidence_accuracy": np.nan,
    }

    if "p_up" in predictions.columns:
        actual_up = predictions["actual_up"].astype(int)
        predicted_up = predictions["predicted_up"].astype(int)
        probabilities = predictions["p_up"].astype(float)
        confidence = predictions["confidence"].astype(float)

        high_confidence = confidence >= HIGH_CONFIDENCE_THRESHOLD

        direction_row.update({
            "accuracy": accuracy_score(actual_up, predicted_up),
            "brier_score": brier_score_loss(
                actual_up, probabilities
            ),
            "high_confidence_count": int(high_confidence.sum()),
            "high_confidence_accuracy": (
                accuracy_score(
                    actual_up[high_confidence],
                    predicted_up[high_confidence],
                )
                if high_confidence.any()
                else np.nan
            ),
        })

    return return_row, direction_row


# ============================================================
# MAIN
# ============================================================

def main():
    df = prepare_data()
    variants = make_variants()

    print("\n===== FEATURE ABLATION EXPERIMENT =====")
    print(f"Dataset: {INPUT_FILE}")
    print(f"Dataset rows with targets: {len(df)}")
    print(f"Initial training fraction: {INITIAL_TRAIN_FRACTION:.0%}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Variants: {len(variants)}")

    all_predictions = []
    return_rows = []
    direction_rows = []

    for name, features in variants.items():
        print(f"Evaluating: {name}")

        predictions = evaluate_variant(df, name, features)
        all_predictions.append(predictions)

        return_row, direction_row = calculate_metrics(
            predictions, name, features
        )
        return_rows.append(return_row)
        direction_rows.append(direction_row)

    all_predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )

    return_results = pd.DataFrame(return_rows).sort_values(
        "MAE_pct"
    ).reset_index(drop=True)

    direction_results = pd.DataFrame(direction_rows).sort_values(
        "brier_score",
        na_position="last",
    ).reset_index(drop=True)

    # Confirm that all variants have identical forecast dates and targets.
    reference = all_predictions_df[
        all_predictions_df["variant"] == "Full model"
    ][["date", "target_next_session_date", "actual_return_pct"]]

    for name in variants:
        current = all_predictions_df[
            all_predictions_df["variant"] == name
        ][["date", "target_next_session_date", "actual_return_pct"]]

        check = reference.merge(
            current,
            on=["date", "target_next_session_date"],
            suffixes=("_reference", "_current"),
            validate="one_to_one",
        )

        if len(check) != len(reference):
            raise ValueError(
                f"Forecast dates differ for variant: {name}"
            )

        if not np.allclose(
            check["actual_return_pct_reference"],
            check["actual_return_pct_current"],
        ):
            raise ValueError(
                f"Actual targets differ for variant: {name}"
            )

    all_predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    return_results.to_csv(RETURN_RESULTS, index=False)
    direction_results.to_csv(DIRECTION_RESULTS, index=False)

    print("\n===== RETURN METRICS (LOWER MAE/RMSE IS BETTER) =====")
    print(
        return_results[
            ["variant", "n_features", "MAE_pct", "RMSE_pct", "R2"]
        ].round(4).to_string(index=False)
    )

    print("\n===== DIRECTION METRICS =====")
    print(
        direction_results[
            [
                "variant",
                "n_features",
                "accuracy",
                "brier_score",
                "high_confidence_count",
                "high_confidence_accuracy",
            ]
        ].round(4).to_string(index=False)
    )

    full_predictions = all_predictions_df[
        all_predictions_df["variant"] == "Full model"
    ]

    print("\n===== EVALUATION PERIOD =====")
    print(f"Forecasts per variant: {len(full_predictions)}")
    print(
        f"From: {full_predictions['date'].min().date()} "
        f"to {full_predictions['date'].max().date()}"
    )

    print("\n===== SAVED FILES =====")
    print(RETURN_RESULTS)
    print(DIRECTION_RESULTS)
    print(PREDICTIONS_FILE)


if __name__ == "__main__":
    main()
