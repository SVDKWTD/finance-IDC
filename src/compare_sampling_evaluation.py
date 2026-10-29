
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

DATASETS = {
    "Original": DATA / "model_dataset.csv",
    "Sampled": DATA / "model_dataset_sampled.csv",
}

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


def evaluate_dataset(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Predictors must be available at the forecast origin.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Use the same 70% chronological initial training window rule.
    start = int(len(df) * 0.70)
    records = []

    for i in range(start, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[[i]].copy()

        train = train.dropna(subset=[TARGET])
        y_train = train[TARGET].astype(float)
        y_test = float(test[TARGET].iloc[0])

        pred_mean = float(y_train.mean())
        predictions = {"Mean baseline": pred_mean}

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

        y_direction = (y_train > 0).astype(int)
        actual_direction = int(y_test > 0)

        record = {
            "date": test["date"].iloc[0],
            "target_next_session_date": test[
                "target_next_session_date"
            ].iloc[0],
            "actual_return_pct": y_test,
            "actual_up": actual_direction,
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


def summarize_returns(pred, dataset_name):
    models = [
        ("Mean baseline", "mean_baseline_prediction_pct"),
        ("Market + news volume", "market_prediction_pct"),
        ("Market + news + sentiment", "sentiment_prediction_pct"),
    ]

    results = []
    for model_name, column in models:
        actual = pred["actual_return_pct"]
        estimate = pred[column]
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "MAE_pct": mean_absolute_error(actual, estimate),
            "RMSE_pct": np.sqrt(mean_squared_error(actual, estimate)),
            "R2": r2_score(actual, estimate),
        })

    return results


def summarize_direction(pred, dataset_name):
    results = []
    for model_name in ["market", "sentiment"]:
        p = pred[f"{model_name}_p_up"]
        y = pred["actual_up"]
        mask = pred[f"{model_name}_confidence"] >= 0.60

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(
                y, pred[f"{model_name}_predicted_up"]
            ),
            "brier_score": brier_score_loss(y, p),
            "high_confidence_count": int(mask.sum()),
            "high_confidence_accuracy": (
                accuracy_score(
                    y[mask],
                    pred.loc[mask, f"{model_name}_predicted_up"],
                )
                if mask.any()
                else np.nan
            ),
        })

    return results


def main():
    all_predictions = {}
    return_results = []
    direction_results = []

    for dataset_name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        print(f"\nEvaluating {dataset_name}: {path.name}")
        pred = evaluate_dataset(path)
        all_predictions[dataset_name] = pred
        return_results.extend(summarize_returns(pred, dataset_name))
        direction_results.extend(summarize_direction(pred, dataset_name))

        print(f"Forecasts: {len(pred)}")
        print(f"Forecast dates: {pred['date'].min().date()} "
              f"to {pred['date'].max().date()}")

    original = all_predictions["Original"]
    sampled = all_predictions["Sampled"]

    # Enforce a paired comparison on identical forecast dates and outcomes.
    paired = original.merge(
        sampled,
        on=["date", "target_next_session_date"],
        suffixes=("_original", "_sampled"),
        validate="one_to_one",
    )

    if len(paired) != len(original) or len(paired) != len(sampled):
        raise ValueError(
            "The datasets do not have identical forecast dates. "
            "Comparison stopped rather than comparing different periods."
        )

    if not np.allclose(
        paired["actual_return_pct_original"],
        paired["actual_return_pct_sampled"],
        equal_nan=True,
    ):
        raise ValueError(
            "Actual targets differ between datasets. "
            "Comparison stopped."
        )

    return_summary = pd.DataFrame(return_results)
    direction_summary = pd.DataFrame(direction_results)

    return_summary.to_csv(
        DATA / "sampling_return_comparison.csv", index=False
    )
    direction_summary.to_csv(
        DATA / "sampling_direction_comparison.csv", index=False
    )

    paired.to_csv(DATA / "sampling_paired_predictions.csv", index=False)

    print("\n===== RETURN COMPARISON =====")
    print(return_summary.round(4).to_string(index=False))

    print("\n===== DIRECTION COMPARISON =====")
    print(direction_summary.round(4).to_string(index=False))

    print("\n===== PAIRED EVALUATION =====")
    print(f"Original forecasts: {len(original)}")
    print(f"Sampled forecasts: {len(sampled)}")
    print(f"Matched forecast dates: {len(paired)}")
    print(
        f"Original period: {original['date'].min().date()} "
        f"to {original['date'].max().date()}"
    )
    print(
        f"Sampled period: {sampled['date'].min().date()} "
        f"to {sampled['date'].max().date()}"
    )

    print("\nSaved separate comparison files:")
    print(DATA / "sampling_return_comparison.csv")
    print(DATA / "sampling_direction_comparison.csv")
    print(DATA / "sampling_paired_predictions.csv")


if __name__ == "__main__":
    main()
