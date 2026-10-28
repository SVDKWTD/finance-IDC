
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def make_pipeline_model(model):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Features known by the end of the current assigned trading session.
    df["previous_return_pct"] = df["daily_return_pct"]
    df["volatility_5_sessions"] = (
        df["daily_return_pct"].rolling(5, min_periods=5).std()
    )

    market_features = [
        "previous_return_pct",
        "volatility_5_sessions",
        "article_count",
        "unique_sources",
    ]
    sentiment_features = [
        "mean_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    ]
    all_features = market_features + sentiment_features

    df = df.dropna(subset=["target_next_session_return_pct"]).copy()
    df["target_up"] = (
        df["target_next_session_return_pct"] > 0
    ).astype(int)

    # Chronological holdout: latest 20% is never used for fitting.
    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train_market = train[market_features]
    X_test_market = test[market_features]
    X_train_all = train[all_features]
    X_test_all = test[all_features]

    y_train = train["target_next_session_return_pct"]
    y_test = test["target_next_session_return_pct"]
    up_train = train["target_up"]
    up_test = test["target_up"]

    print("===== CHRONOLOGICAL HOLDOUT =====")
    print(f"Training sessions: {len(train)}")
    print(f"Testing sessions:  {len(test)}")
    print(f"Training dates: {train['date'].min().date()} to "
          f"{train['date'].max().date()}")
    print(f"Testing dates:  {test['date'].min().date()} to "
          f"{test['date'].max().date()}")

    # Return forecasts: compare a mean baseline, market-only model,
    # and market-plus-sentiment model.
    return_models = [
        (
            "Mean baseline",
            make_pipeline_model(DummyRegressor(strategy="mean")),
            train[["previous_return_pct"]],
            test[["previous_return_pct"]],
        ),
        (
            "Market + news volume",
            make_pipeline_model(Ridge(alpha=10.0)),
            X_train_market,
            X_test_market,
        ),
        (
            "Market + news + sentiment",
            make_pipeline_model(Ridge(alpha=10.0)),
            X_train_all,
            X_test_all,
        ),
    ]

    return_results = []
    for name, model, xtr, xte in return_models:
        model.fit(xtr, y_train)
        pred = model.predict(xte)

        return_results.append({
            "model": name,
            "MAE_pct": mean_absolute_error(y_test, pred),
            "RMSE_pct": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

        if name == "Market + news + sentiment":
            return_predictions = pred

    pd.DataFrame(return_results).to_csv(
        OUTPUT_DIR / "forecast_model_comparison.csv", index=False
    )

    # Directional probability: estimate the probability of an upward session.
    # These are initial model probabilities, not yet calibrated probabilities.
    direction_models = [
        (
            "Majority-class baseline",
            make_pipeline_model(DummyClassifier(strategy="prior")),
            train[["previous_return_pct"]],
            test[["previous_return_pct"]],
        ),
        (
            "Market + news volume",
            make_pipeline_model(
                LogisticRegression(max_iter=2000, class_weight=None)
            ),
            X_train_market,
            X_test_market,
        ),
        (
            "Market + news + sentiment",
            make_pipeline_model(
                LogisticRegression(max_iter=2000, class_weight=None)
            ),
            X_train_all,
            X_test_all,
        ),
    ]

    direction_results = []
    for name, model, xtr, xte in direction_models:
        model.fit(xtr, up_train)
        probability = model.predict_proba(xte)[:, 1]
        predicted_up = (probability >= 0.5).astype(int)

        direction_results.append({
            "model": name,
            "accuracy": accuracy_score(up_test, predicted_up),
            "brier_score": brier_score_loss(up_test, probability),
        })

        if name == "Market + news + sentiment":
            up_probability = probability

    pd.DataFrame(direction_results).to_csv(
        OUTPUT_DIR / "direction_model_comparison.csv", index=False
    )

    # Record predictions, errors, and confidence indicators for every test day.
    forecasts = test[["date", "target_next_session_return_pct"]].copy()
    forecasts = forecasts.rename(
        columns={"target_next_session_return_pct": "actual_return_pct"}
    )
    forecasts["predicted_return_pct"] = return_predictions
    forecasts["return_error_pct"] = (
        forecasts["actual_return_pct"] - forecasts["predicted_return_pct"]
    )
    forecasts["absolute_error_pct"] = forecasts["return_error_pct"].abs()
    forecasts["actual_up"] = (forecasts["actual_return_pct"] > 0).astype(int)
    forecasts["probability_up"] = up_probability
    forecasts["predicted_direction"] = (up_probability >= 0.5).astype(int)
    forecasts["direction_correct"] = (
        forecasts["actual_up"] == forecasts["predicted_direction"]
    )
    forecasts["directional_confidence"] = np.maximum(
        up_probability, 1 - up_probability
    )
    forecasts["volatility_5_sessions"] = test[
        "volatility_5_sessions"
    ].to_numpy()
    forecasts["mean_sentiment"] = test["mean_sentiment"].to_numpy()

    forecasts.to_csv(
        OUTPUT_DIR / "forecast_test_predictions.csv", index=False
    )

    # Reliability table: compare probability estimates with observed frequencies.
    # Small samples in individual bins can make these rates unstable.
    forecasts["probability_bin"] = pd.cut(
        forecasts["probability_up"],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )

    calibration = (
        forecasts.groupby("probability_bin", observed=False)
        .agg(
            predictions=("actual_up", "size"),
            mean_predicted_probability=("probability_up", "mean"),
            observed_up_frequency=("actual_up", "mean"),
            directional_accuracy=("direction_correct", "mean"),
        )
        .reset_index()
    )
    calibration.to_csv(
        OUTPUT_DIR / "forecast_probability_reliability.csv", index=False
    )

    # Error diagnostics by volatility level.
    forecasts["volatility_group"] = pd.qcut(
        forecasts["volatility_5_sessions"].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"],
    )

    error_analysis = (
        forecasts.groupby("volatility_group", observed=False)
        .agg(
            sessions=("absolute_error_pct", "size"),
            mean_absolute_error_pct=("absolute_error_pct", "mean"),
            directional_accuracy=("direction_correct", "mean"),
            mean_confidence=("directional_confidence", "mean"),
        )
        .reset_index()
    )
    error_analysis.to_csv(
        OUTPUT_DIR / "forecast_error_by_volatility.csv", index=False
    )

    print("\n===== RETURN FORECAST COMPARISON =====")
    print(pd.DataFrame(return_results).round(4).to_string(index=False))

    print("\n===== DIRECTION PROBABILITY COMPARISON =====")
    print(pd.DataFrame(direction_results).round(4).to_string(index=False))

    print("\n===== ERROR BY VOLATILITY =====")
    print(error_analysis.round(4).to_string(index=False))

    print("\n===== HIGH-CONFIDENCE FORECASTS =====")
    high_conf = forecasts[
        forecasts["directional_confidence"] >= 0.60
    ]
    if len(high_conf):
        print(f"Forecasts with confidence >= 60%: {len(high_conf)}")
        print(f"Observed directional accuracy: "
              f"{high_conf['direction_correct'].mean():.3f}")
    else:
        print("No forecasts reached the 60% confidence threshold.")

    print("\nReports saved:")
    print("forecast_model_comparison.csv")
    print("direction_model_comparison.csv")
    print("forecast_test_predictions.csv")
    print("forecast_probability_reliability.csv")
    print("forecast_error_by_volatility.csv")


if __name__ == "__main__":
    main()
