
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
df = pd.read_csv(DATA / "model_dataset.csv", parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)

# All predictors must be available at the forecast origin.
df["return_lag1"] = df["daily_return_pct"].shift(1)
df["return_lag2"] = df["daily_return_pct"].shift(2)
df["return_lag5"] = df["daily_return_pct"].shift(5)
df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

market_features = [
    "daily_return_pct", "return_lag1", "return_lag2",
    "return_lag5", "volatility_5", "article_count",
    "unique_sources"
]
sentiment_features = market_features + [
    "mean_sentiment", "positive_fraction",
    "negative_fraction", "neutral_fraction", "sentiment_lag1"
]

target = "target_next_session_return_pct"
df = df.dropna(subset=[target]).reset_index(drop=True)

# Use the first 70% as the initial training history.
start = int(len(df) * 0.70)
records = []

for i in range(start, len(df)):
    train = df.iloc[:i].copy()
    test = df.iloc[[i]].copy()

    # Remove rows with missing targets; imputation is fitted on training data.
    train = train.dropna(subset=[target])
    y_train = train[target].astype(float)
    y_test = float(test[target].iloc[0])

    # Historical-mean baseline.
    pred_mean = float(y_train.mean())

    predictions = {"Mean baseline": pred_mean}

    for name, features in [
        ("Market + news volume", market_features),
        ("Market + news + sentiment", sentiment_features),
    ]:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0)
        )
        model.fit(train[features], y_train)
        predictions[name] = float(model.predict(test[features])[0])

    # Direction probabilities. Training labels use only historical targets.
    y_direction = (y_train > 0).astype(int)
    actual_direction = int(y_test > 0)

    record = {
        "date": test["date"].iloc[0],
        "target_next_session_date": test["target_next_session_date"].iloc[0],
        "actual_return_pct": y_test,
        "actual_up": actual_direction,
        "mean_baseline_prediction_pct": predictions["Mean baseline"],
        "market_prediction_pct": predictions["Market + news volume"],
        "sentiment_prediction_pct": predictions["Market + news + sentiment"],
    }

    for name, features in [
        ("market", market_features),
        ("sentiment", sentiment_features),
    ]:
        if y_direction.nunique() == 2:
            classifier = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight=None)
            )
            classifier.fit(train[features], y_direction)
            p_up = float(classifier.predict_proba(test[features])[0, 1])
        else:
            p_up = float(y_direction.mean())

        record[f"{name}_p_up"] = p_up
        record[f"{name}_predicted_up"] = int(p_up >= 0.5)
        record[f"{name}_confidence"] = max(p_up, 1 - p_up)

    records.append(record)

pred = pd.DataFrame(records)
pred.to_csv(DATA / "walk_forward_predictions.csv", index=False)

models = [
    ("Mean baseline", "mean_baseline_prediction_pct"),
    ("Market + news volume", "market_prediction_pct"),
    ("Market + news + sentiment", "sentiment_prediction_pct"),
]

summary = []
for name, col in models:
    actual = pred["actual_return_pct"]
    estimate = pred[col]
    summary.append({
        "model": name,
        "MAE_pct": mean_absolute_error(actual, estimate),
        "RMSE_pct": np.sqrt(mean_squared_error(actual, estimate)),
        "R2": r2_score(actual, estimate),
    })

pd.DataFrame(summary).to_csv(
    DATA / "walk_forward_return_comparison.csv", index=False
)

direction_summary = []
for name in ["market", "sentiment"]:
    p = pred[f"{name}_p_up"]
    y = pred["actual_up"]
    direction_summary.append({
        "model": name,
        "accuracy": accuracy_score(y, pred[f"{name}_predicted_up"]),
        "brier_score": brier_score_loss(y, p),
        "high_confidence_count": int((pred[f"{name}_confidence"] >= 0.60).sum()),
        "high_confidence_accuracy": (
            accuracy_score(
                y[pred[f"{name}_confidence"] >= 0.60],
                pred.loc[pred[f"{name}_confidence"] >= 0.60,
                         f"{name}_predicted_up"]
            )
            if (pred[f"{name}_confidence"] >= 0.60).any()
            else np.nan
        ),
    })

pd.DataFrame(direction_summary).to_csv(
    DATA / "walk_forward_direction_comparison.csv", index=False
)

print("\n===== WALK-FORWARD RETURN COMPARISON =====")
print(pd.DataFrame(summary).round(4).to_string(index=False))

print("\n===== WALK-FORWARD DIRECTION COMPARISON =====")
print(pd.DataFrame(direction_summary).round(4).to_string(index=False))

print("\n===== EVALUATION PERIOD =====")
print(f"Forecasts: {len(pred)}")
print(f"From: {pred['date'].min().date()} to {pred['date'].max().date()}")
print("\nSaved:")
print(DATA / "walk_forward_predictions.csv")
print(DATA / "walk_forward_return_comparison.csv")
print(DATA / "walk_forward_direction_comparison.csv")
