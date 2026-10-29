"""Prospective, frozen-spec volatility forecasting for NIFTY 50.

Run from project root:
    python src/prospective_volatility_forecast.py

Reads the existing model_dataset.csv.
Appends one forecast per data date to:
    data/processed/prospective_volatility_forecasts.csv

This is a forward evaluation log. Do not tune the model/threshold after seeing outcomes.
Forecast assumption: after the current session's close.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset.csv"
LOG = ROOT / "data" / "processed" / "prospective_volatility_forecasts.csv"
TARGET = "target_next_session_return_pct"

# Frozen specification from the completed experiment
WINDOW = 252
MIN_HISTORY = 60
VOL_PERCENTILE = 0.80
MIN_TRAIN_ROWS = 300
SEED = 42

def build_frame(raw):
    df = raw.sort_values("date").drop_duplicates("date").reset_index(drop=True).copy()
    r = pd.to_numeric(df["daily_return_pct"], errors="coerce")
    df["return_lag1"] = r.shift(1)
    df["return_lag2"] = r.shift(2)
    df["return_lag5"] = r.shift(5)
    df["return_lag10"] = r.shift(10)
    df["volatility_5"] = r.rolling(5, min_periods=5).std()
    df["volatility_20"] = r.rolling(20, min_periods=10).std()
    df["momentum_5"] = r.rolling(5, min_periods=5).sum()
    df["momentum_10"] = r.rolling(10, min_periods=10).sum()
    df["abs_return"] = r.abs()
    df["volatility_threshold_pct"] = df["abs_return"].rolling(
        WINDOW, min_periods=MIN_HISTORY
    ).quantile(VOL_PERCENTILE)

    target = pd.to_numeric(df[TARGET], errors="coerce")
    df["is_volatile_next_session"] = np.where(
        df["volatility_threshold_pct"].notna() & target.notna(),
        (target.abs() > df["volatility_threshold_pct"]).astype(float),
        np.nan,
    )
    return df

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    raw = pd.read_csv(DATA, parse_dates=["date"])
    if not {"date", "daily_return_pct", TARGET}.issubset(raw.columns):
        raise ValueError("model_dataset.csv lacks required columns.")

    df = build_frame(raw)
    features = [
        "daily_return_pct", "return_lag1", "return_lag2", "return_lag5", "return_lag10",
        "volatility_5", "volatility_20", "momentum_5", "momentum_10",
    ]
    for col in ["article_count", "unique_sources", "mean_sentiment", "median_sentiment",
                "positive_fraction", "negative_fraction", "neutral_fraction"]:
        if col in df.columns:
            features.append(col)

    # Last date with market features and a computed historical threshold.
    candidate = df.loc[df["volatility_threshold_pct"].notna()].iloc[-1]
    forecast_date = pd.Timestamp(candidate["date"])
    if pd.isna(candidate["daily_return_pct"]):
        raise ValueError("Latest row has no daily return; update the dataset after a completed market session.")

    train = df.loc[df["is_volatile_next_session"].notna()].copy()
    train = train.loc[train[features].notna().any(axis=1)]
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"Only {len(train)} labelled training rows; need at least {MIN_TRAIN_ROWS}.")
    y = train["is_volatile_next_session"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training labels contain only one class; cannot fit classifier.")

    # Training pipeline fit only on labelled past rows.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    model.fit(train[features], y)

    X_now = pd.DataFrame([{col: candidate[col] for col in features}])
    p_volatile = float(model.predict_proba(X_now)[0, list(model.named_steps["model"].classes_).index(1)])
    threshold = float(candidate["volatility_threshold_pct"])
    next_session = candidate.get("target_next_session_date", pd.NaT)
    if pd.isna(next_session):
        next_session_text = ""
    else:
        next_session_text = str(pd.Timestamp(next_session).date())

    record = {
        "forecast_generated_at_local_date": pd.Timestamp.now().date().isoformat(),
        "feature_date": forecast_date.date().isoformat(),
        "target_session_date_if_known": next_session_text,
        "probability_next_session_volatile": p_volatile,
        "historical_abs_return_threshold_pct": threshold,
        "volatility_definition": f"absolute next-session return > trailing {WINDOW}-session {VOL_PERCENTILE:.0%} quantile",
        "alert_at_0_30": int(p_volatile >= 0.30),
        "alert_at_0_20": int(p_volatile >= 0.20),
        "training_rows": len(train),
        "training_volatile_rate": float(y.mean()),
        "model_spec": "LogisticRegression(max_iter=2000, random_state=42); median imputation; StandardScaler",
        "outcome_abs_return_pct": np.nan,
        "outcome_volatile": np.nan,
        "outcome_recorded": 0,
    }

    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        old = pd.read_csv(LOG)
        if "feature_date" in old.columns and (old["feature_date"].astype(str) == record["feature_date"]).any():
            raise ValueError(
                f"A forecast for feature date {record['feature_date']} already exists in {LOG}. "
                "No row was added; this prevents duplicate forecasts."
            )
        combined = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
    else:
        combined = pd.DataFrame([record])
    combined.to_csv(LOG, index=False)

    print("=" * 66)
    print("PROSPECTIVE VOLATILITY FORECAST (FROZEN MODEL SPECIFICATION)")
    print("=" * 66)
    print(f"Feature date: {record['feature_date']}")
    print(f"Target session date: {next_session_text or 'not yet known'}")
    print(f"Probability next session is volatile: {p_volatile:.1%}")
    print(f"Historical threshold for absolute return: {threshold:.4f}%")
    print(f"Alert at 20% threshold: {'YES' if record['alert_at_0_20'] else 'NO'}")
    print(f"Alert at 30% threshold: {'YES' if record['alert_at_0_30'] else 'NO'}")
    print(f"Training rows: {len(train)}; training volatile rate: {y.mean():.1%}")
    print(f"Appended forecast log: {LOG}")
    print()
    print("Do not change the model specification after seeing this forecast.")
    print("After the target session closes, record its absolute close-to-close return and")
    print("whether it exceeded the stored historical threshold in this CSV.")
    print("This is experimental; same-date news features must be available by the forecast cutoff.")

if __name__ == "__main__":
    main()
