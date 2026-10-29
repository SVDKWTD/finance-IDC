"""Leakage-aware NIFTY 50 volatility + volatile-session direction experiment.
Run from project root: python src/volatility_forecast_experiment.py
Reads data/processed/model_dataset.csv and writes two NEW CSVs in data/processed/.
Forecast timing assumption: after the current session's close; verify news timestamps.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset.csv"
OUT_PRED = ROOT / "data" / "processed" / "volatility_experiment_predictions.csv"
OUT_SUM = ROOT / "data" / "processed" / "volatility_experiment_summary.csv"
TARGET = "target_next_session_return_pct"
INITIAL_TRAIN_FRACTION = 0.70
WINDOW = 252
MIN_HISTORY = 60
VOL_PERCENTILE = 0.80
MIN_VOLATILE_TRAIN = 30
SEED = 42

def pipeline(dummy=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if dummy:
        steps.append(("model", DummyClassifier(strategy="most_frequent", random_state=SEED)))
    else:
        steps += [("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=2000, random_state=SEED))]
    return Pipeline(steps)

def prob1(model, X):
    classes = model.named_steps["model"].classes_
    p = model.predict_proba(X)
    return p[:, list(classes).index(1)] if 1 in classes else np.zeros(len(X))

def binary_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": len(y),
        "actual_positive_rate": float(y.mean()) if len(y) else np.nan,
        "accuracy_at_0.5": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "precision_at_0.5": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "recall_at_0.5": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA}")
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    required = {"date", "daily_return_pct", TARGET}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {sorted(required - set(df.columns))}")

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

    # Threshold at date t uses known absolute returns through date t, never target t->t+1.
    df["volatility_threshold_pct"] = df["abs_return"].rolling(
        WINDOW, min_periods=MIN_HISTORY
    ).quantile(VOL_PERCENTILE)
    y_abs = pd.to_numeric(df[TARGET], errors="coerce").abs()
    df["is_volatile_next_session"] = np.where(
        df["volatility_threshold_pct"].notna() & y_abs.notna(),
        (y_abs > df["volatility_threshold_pct"]).astype(float), np.nan
    )
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df["next_session_up"] = np.where(y.notna(), (y > 0).astype(float), np.nan)

    features = [
        "daily_return_pct", "return_lag1", "return_lag2", "return_lag5", "return_lag10",
        "volatility_5", "volatility_20", "momentum_5", "momentum_10",
    ]
    for col in ["article_count", "unique_sources", "mean_sentiment", "median_sentiment",
                "positive_fraction", "negative_fraction", "neutral_fraction"]:
        if col in df.columns:
            features.append(col)

    usable = df.loc[df[TARGET].notna() & df["is_volatile_next_session"].notna()].copy().reset_index(drop=True)
    if len(usable) < 300:
        raise ValueError(f"Only {len(usable)} usable rows; check dataset/threshold settings.")
    split = int(len(usable) * INITIAL_TRAIN_FRACTION)
    preds = []

    print("=" * 68)
    print("NIFTY 50 VOLATILITY + VOLATILE-SESSION DIRECTION EXPERIMENT")
    print("=" * 68)
    print(f"Input rows: {len(df)} | Usable rows: {len(usable)}")
    print(f"Initial training: {split} | Forecasts: {len(usable)-split}")
    print(f"Volatile = absolute next-session return > rolling {VOL_PERCENTILE:.0%} percentile")
    print(f"Threshold history: {WINDOW} sessions, minimum {MIN_HISTORY}; forecast after close.")
    print("All models are refit expanding-window; no shuffled split.")

    for i in range(split, len(usable)):
        train, test = usable.iloc[:i], usable.iloc[[i]]
        Xtr, Xte = train[features], test[features]
        yv = train["is_volatile_next_session"].astype(int)
        actual_vol = int(test["is_volatile_next_session"].iloc[0])
        prevalence = float(yv.mean())

        # Baseline probability is historical training prevalence.
        vol_base = prevalence
        vol_dummy = pipeline(dummy=True)
        vol_dummy.fit(Xtr, yv)
        p_dummy = float(prob1(vol_dummy, Xte)[0])

        if yv.nunique() >= 2:
            vol_model = pipeline()
            vol_model.fit(Xtr, yv)
            p_vol = float(prob1(vol_model, Xte)[0])
        else:
            p_vol = prevalence

        # Train direction only on previously labelled volatile sessions.
        vtrain = train.loc[train["is_volatile_next_session"].astype(int) == 1]
        p_dir_base, p_dir_model = np.nan, np.nan
        direction_available = len(vtrain) >= MIN_VOLATILE_TRAIN
        if direction_available:
            yd = vtrain["next_session_up"].astype(int)
            p_dir_base = float(yd.mean())
            if yd.nunique() >= 2:
                dmodel = pipeline()
                dmodel.fit(vtrain[features], yd)
                p_dir_model = float(prob1(dmodel, Xte)[0])
            else:
                p_dir_model = p_dir_base

        row = {
            "date": test["date"].iloc[0],
            "target_next_session_date": test["target_next_session_date"].iloc[0] if "target_next_session_date" in test else pd.NaT,
            "actual_next_session_return_pct": float(test[TARGET].iloc[0]),
            "abs_actual_next_session_return_pct": abs(float(test[TARGET].iloc[0])),
            "historical_volatility_threshold_pct": float(test["volatility_threshold_pct"].iloc[0]),
            "actual_volatile": actual_vol,
            "volatility_baseline_probability": vol_base,
            "volatility_dummy_probability": p_dummy,
            "volatility_model_probability": p_vol,
            "actual_direction_up": int(test["next_session_up"].iloc[0]),
            "volatile_direction_prediction_available": direction_available,
            "volatile_direction_baseline_probability_up": p_dir_base,
            "volatile_direction_model_probability_up": p_dir_model,
        }
        preds.append(row)

    p = pd.DataFrame(preds)
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT_PRED, index=False)

    rows = []
    yv = p["actual_volatile"].astype(int).to_numpy()
    for name, col in [
        ("training_prevalence_baseline", "volatility_baseline_probability"),
        ("most_frequent_dummy", "volatility_dummy_probability"),
        ("logistic_volatility_model", "volatility_model_probability"),
    ]:
        m = binary_metrics(yv, p[col].to_numpy())
        rows.append({"task": "next_session_volatility", "model": name, **m})

    # Retrospective subgroup evaluation: actual volatile sessions only.
    d = p[(p["actual_volatile"] == 1) & p["volatile_direction_prediction_available"]].copy()
    if len(d):
        yd = d["actual_direction_up"].astype(int).to_numpy()
        for name, col in [
            ("volatile_training_prevalence_baseline", "volatile_direction_baseline_probability_up"),
            ("logistic_trained_on_past_volatile_sessions", "volatile_direction_model_probability_up"),
        ]:
            m = binary_metrics(yd, d[col].astype(float).to_numpy())
            rows.append({"task": "direction_on_actual_volatile_sessions", "model": name, **m})

    summary = pd.DataFrame(rows)
    summary.insert(0, "forecast_start", str(p["date"].min().date()))
    summary.insert(1, "forecast_end", str(p["date"].max().date()))
    summary.insert(2, "forecast_rows", len(p))
    summary.insert(3, "volatility_percentile", VOL_PERCENTILE)
    summary.insert(4, "threshold_window_sessions", WINDOW)
    summary.insert(5, "initial_train_fraction", INITIAL_TRAIN_FRACTION)
    summary.to_csv(OUT_SUM, index=False)

    print("\nVOLATILITY: lower Brier is better; ROC-AUC near 0.5 means little ranking signal.")
    print(summary[summary.task == "next_session_volatility"][
        ["model", "n", "actual_positive_rate", "accuracy_at_0.5", "brier", "roc_auc"]
    ].to_string(index=False))
    print("\nDIRECTION ON ACTUAL VOLATILE SESSIONS:")
    ds = summary[summary.task == "direction_on_actual_volatile_sessions"]
    if ds.empty:
        print("Not enough past volatile training sessions to evaluate direction.")
    else:
        print(ds[["model", "n", "actual_positive_rate", "accuracy_at_0.5", "brier", "roc_auc"]].to_string(index=False))
    print(f"\nSaved predictions: {OUT_PRED}")
    print(f"Saved summary:     {OUT_SUM}")
    print("\nCaveats:")
    print("- Same-date news features are safe only if available by the forecast cutoff.")
    print("- Volatile-day direction metrics use actual volatile outcomes retrospectively.")
    print("- Treat this as research, not a validated trading strategy.")

if __name__ == "__main__":
    main()
