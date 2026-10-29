
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

MODEL_FILE = DATA / "model_dataset.csv"

PREDICTIONS_FILE = DATA / "walk_forward_predictions.csv"
RETURN_COMPARISON_FILE = DATA / "walk_forward_return_comparison.csv"
DIRECTION_COMPARISON_FILE = DATA / "walk_forward_direction_comparison.csv"

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


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {MODEL_FILE}\n"
            "Run src\\build_model_dataset.py first."
        )

    df = pd.read_csv(MODEL_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Predictors available at the forecast origin.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    required_columns = (
        MARKET_FEATURES + SENTIMENT_FEATURES + [TARGET]
    )
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Model dataset is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Keep the original chronological walk-forward design.
    start = int(len(df) * 0.70)
    records = []

    for i in range(start, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[[i]].copy()

        train = train.dropna(subset=[TARGET])
        y_train = train[TARGET].astype(float)
        y_test = float(test[TARGET].iloc[0])

        # Historical-mean baseline.
        predictions = {"Mean baseline": float(y_train.mean())}

        # Return prediction models.
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
            predictions[name] = float(model.predict(test[features])[0])

        # Direction probabilities use historical targets only.
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

    pred = pd.DataFrame(records)

    # Save predictions separately from the sampling experiment's outputs.
    pred.to_csv(PREDICTIONS_FILE, index=False)

    models = [
        ("Mean baseline", "mean_baseline_prediction_pct"),
        ("Market + news volume", "market_prediction_pct"),
        ("Market + news + sentiment", "sentiment_prediction_pct"),
    ]

    summary = []
    for name, column in models:
        actual = pred["actual_return_pct"]
        estimate = pred[column]

        summary.append({
            "model": name,
            "MAE_pct": mean_absolute_error(actual, estimate),
            "RMSE_pct": np.sqrt(mean_squared_error(actual, estimate)),
            "R2": r2_score(actual, estimate),
        })

    return_summary = pd.DataFrame(summary)
    return_summary.to_csv(RETURN_COMPARISON_FILE, index=False)

    direction_summary = []

    for name in ["market", "sentiment"]:
        probabilities = pred[f"{name}_p_up"]
        actual = pred["actual_up"]
        high_confidence = pred[f"{name}_confidence"] >= 0.60

        direction_summary.append({
            "model": name,
            "accuracy": accuracy_score(
                actual, pred[f"{name}_predicted_up"]
            ),
            "brier_score": brier_score_loss(actual, probabilities),
            "high_confidence_count": int(high_confidence.sum()),
            "high_confidence_accuracy": (
                accuracy_score(
                    actual[high_confidence],
                    pred.loc[high_confidence, f"{name}_predicted_up"],
                )
                if high_confidence.any()
                else np.nan
            ),
        })

    direction_summary = pd.DataFrame(direction_summary)
    direction_summary.to_csv(DIRECTION_COMPARISON_FILE, index=False)

    print("\n===== ACTIVE DATASET =====")
    print(MODEL_FILE)
    print("Features include sampled article volume and source diversity.")

    print("\n===== WALK-FORWARD RETURN COMPARISON =====")
    print(return_summary.round(4).to_string(index=False))

    print("\n===== WALK-FORWARD DIRECTION COMPARISON =====")
    print(direction_summary.round(4).to_string(index=False))

    print("\n===== EVALUATION PERIOD =====")
    print(f"Forecasts: {len(pred)}")
    print(f"From: {pred['date'].min().date()} to {pred['date'].max().date()}")

    print("\nSaved:")
    print(PREDICTIONS_FILE)
    print(RETURN_COMPARISON_FILE)
    print(DIRECTION_COMPARISON_FILE)


if __name__ == "__main__":
    main()
