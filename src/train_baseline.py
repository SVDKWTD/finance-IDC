from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "baseline_predictions.csv"

TARGET = "target_next_session_return_pct"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Only lagged returns and news-count/source features.
excluded = {
    "date", TARGET,
    "open", "high", "low", "close", "adj_close",
    "volume", "daily_return", "daily_return_pct",
}

features = []
for col in df.columns:
    name = col.lower()

    if col in excluded or not pd.api.types.is_numeric_dtype(df[col]):
        continue

    is_news_feature = any(
        term in name for term in ("news", "article", "count", "source")
    )
    is_lagged_return = (
        "return" in name
        and any(term in name for term in ("lag", "prev", "prior"))
    )

    if is_news_feature or is_lagged_return:
        features.append(col)

if not features:
    raise ValueError(
        "No eligible features found. Check the column names in model_ready.csv."
    )

data = df[["date", TARGET] + features].dropna().reset_index(drop=True)

# Chronological split: do not shuffle.
split = int(len(data) * 0.80)
train = data.iloc[:split]
test = data.iloc[split:]

X_train = train[features]
y_train = train[TARGET]
X_test = test[features]
y_test = test[TARGET]

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Benchmark: predict zero return for every test session.
zero_predictions = np.zeros(len(y_test))

def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name}: MAE={mae:.4f} pp | RMSE={rmse:.4f} pp")

print(f"Features ({len(features)}): {features}")
print(f"Train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows)")
print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)")
print()

report("Zero-return benchmark", y_test, zero_predictions)
report("Lagged returns + news model", y_test, predictions)

# Directional accuracy for the model; zero predictions are not counted.
nonzero = predictions != 0
if nonzero.any():
    direction_accuracy = (
        np.sign(predictions[nonzero]) == np.sign(y_test.to_numpy()[nonzero])
    ).mean()
    print(f"Model directional accuracy (nonzero predictions): {direction_accuracy:.2%}")
else:
    print("Model directional accuracy: N/A (all predictions are zero)")

output = test[["date", TARGET]].copy()
output["zero_return_prediction"] = zero_predictions
output["model_prediction"] = predictions
output.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved predictions to: {OUTPUT_PATH}")