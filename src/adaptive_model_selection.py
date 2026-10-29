"""Conservative recent-performance selector for the official capped dataset.

Evaluates expanding and rolling training windows on the most recent two calendar
months of genuinely out-of-sample historical predictions. Selection is based on
MAE for returns and Brier score for direction. A simple mean-return baseline is
always eligible and is the fallback unless a feature model clears a small,
predeclared improvement margin.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
SELECTION_FILE = DATA_DIR / "adaptive_model_selection.csv"
SELECTION_JSON = DATA_DIR / "adaptive_model_selection.json"
TARGET = "target_next_session_return_pct"
ROLLING_WINDOW = 252
MIN_RECENT_PREDICTIONS = 30
MIN_MODEL_IMPROVEMENT = 0.02  # challenger must lower MAE by >=2% vs baseline

MARKET_FEATURES = ["daily_return_pct", "return_lag1", "return_lag2", "return_lag5", "volatility_5", "article_count", "unique_sources"]
SENTIMENT_FEATURES = MARKET_FEATURES + ["mean_sentiment", "positive_fraction", "negative_fraction", "neutral_fraction", "sentiment_lag1"]


def make_regressor():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def make_classifier():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))


def prepare_features(df):
    df = df.sort_values("date").reset_index(drop=True).copy()
    r = pd.to_numeric(df["daily_return_pct"], errors="coerce")
    df["return_lag1"] = r.shift(1)
    df["return_lag2"] = r.shift(2)
    df["return_lag5"] = r.shift(5)
    df["volatility_5"] = r.rolling(5, min_periods=5).std()
    df["sentiment_lag1"] = pd.to_numeric(df["mean_sentiment"], errors="coerce").shift(1)
    return df


def train_slice(df, i, mode):
    start = max(0, i - ROLLING_WINDOW) if mode == "rolling" else 0
    return df.iloc[start:i].dropna(subset=[TARGET]).copy()


def select_strategy(df):
    """Return chosen strategies plus recent-window diagnostic rows."""
    df = prepare_features(df)
    known = df.dropna(subset=[TARGET]).copy().reset_index(drop=True)
    if known.empty:
        raise ValueError("No known targets available for adaptive selection.")

    latest_target_date = pd.to_datetime(known["date"].max())
    start_date = latest_target_date - pd.DateOffset(months=2)
    recent_positions = [i for i, row in known.iterrows() if pd.Timestamp(row["date"]) >= start_date]
    # The first 252 sessions are needed before rolling candidates are eligible.
    min_position = max(ROLLING_WINDOW, 300)
    recent_positions = [i for i in recent_positions if i >= min_position]

    candidates = []
    pred_records = []
    for i in recent_positions:
        test = known.iloc[[i]]
        y_true = float(test[TARGET].iloc[0])
        for mode in ("expanding", "rolling"):
            train = train_slice(known, i, mode)
            if len(train) < 300:
                continue
            y_train = train[TARGET].astype(float)
            base_pred = float(y_train.mean())
            pred_records.append({"date": test["date"].iloc[0], "mode": mode, "family": "baseline", "actual_return_pct": y_true, "predicted_return_pct": base_pred, "p_up": float((y_train > 0).mean())})
            for family, features in (("market", MARKET_FEATURES), ("sentiment", SENTIMENT_FEATURES)):
                reg = make_regressor()
                reg.fit(train[features], y_train)
                pred = float(reg.predict(test[features])[0])
                direction_train = (y_train > 0).astype(int)
                if direction_train.nunique() >= 2:
                    clf = make_classifier()
                    clf.fit(train[features], direction_train)
                    p_up = float(clf.predict_proba(test[features])[0, list(clf.named_steps["logisticregression"].classes_).index(1)])
                else:
                    p_up = float(direction_train.mean())
                pred_records.append({"date": test["date"].iloc[0], "mode": mode, "family": family, "actual_return_pct": y_true, "predicted_return_pct": pred, "p_up": p_up})

    preds = pd.DataFrame(pred_records)
    expected_n = len(set(pd.to_datetime(known.loc[recent_positions, "date"]))) if recent_positions else 0
    rows = []
    if not preds.empty:
        for (mode, family), group in preds.groupby(["mode", "family"]):
            actual = group["actual_return_pct"].astype(float)
            estimate = group["predicted_return_pct"].astype(float)
            actual_up = (actual > 0).astype(int)
            probs = group["p_up"].clip(0.0, 1.0)
            rows.append({"window_mode": mode, "model_family": family, "n_predictions": len(group), "mae_pct": mean_absolute_error(actual, estimate), "rmse_pct": mean_squared_error(actual, estimate) ** 0.5, "direction_accuracy": accuracy_score(actual_up, (probs >= 0.5).astype(int)), "direction_brier": brier_score_loss(actual_up, probs), "period_start": str(pd.to_datetime(group["date"]).min().date()), "period_end": str(pd.to_datetime(group["date"]).max().date())})
    metrics = pd.DataFrame(rows)

    expanding_baseline_count = int(((preds["family"] == "baseline") & (preds["mode"] == "expanding")).sum()) if not preds.empty else 0
    enough = expanding_baseline_count >= MIN_RECENT_PREDICTIONS
    fallback_reason = ""
    if not enough:
        selected_return = {"family": "baseline", "window_mode": "expanding"}
        selected_direction = {"family": "market", "window_mode": "expanding"}
        fallback_reason = f"Only {expanding_baseline_count} recent out-of-sample predictions; need at least {MIN_RECENT_PREDICTIONS}. Conservative defaults retained."
    else:
        baseline = metrics[(metrics.model_family == "baseline") & (metrics.window_mode == "expanding")]
        baseline_mae = float(baseline.iloc[0].mae_pct) if not baseline.empty else float("inf")
        eligible = metrics[(metrics.model_family != "baseline") & (metrics.n_predictions >= MIN_RECENT_PREDICTIONS)].copy()
        eligible = eligible[eligible.mae_pct <= baseline_mae * (1.0 - MIN_MODEL_IMPROVEMENT)]
        if eligible.empty:
            selected_return = {"family": "baseline", "window_mode": "expanding"}
            fallback_reason = "No feature model beat the expanding mean-return baseline by the required 2% MAE margin; baseline selected."
        else:
            best = eligible.sort_values(["mae_pct", "rmse_pct"], ascending=True).iloc[0]
            selected_return = {"family": str(best.model_family), "window_mode": str(best.window_mode)}

        direction_candidates = metrics[(metrics.model_family != "baseline") & (metrics.n_predictions >= MIN_RECENT_PREDICTIONS)].copy()
        if direction_candidates.empty:
            selected_direction = {"family": "market", "window_mode": "expanding"}
            fallback_reason = (fallback_reason + " " if fallback_reason else "") + "Insufficient direction candidates; expanding market classifier selected."
        else:
            best_dir = direction_candidates.sort_values(["direction_brier", "direction_accuracy"], ascending=[True, False]).iloc[0]
            selected_direction = {"family": str(best_dir.model_family), "window_mode": str(best_dir.window_mode)}

    result = {
        "selection_date": str(latest_target_date.date()),
        "evaluation_start_date": str(start_date.date()),
        "evaluation_end_date": str(latest_target_date.date()),
        "recent_prediction_count": expanding_baseline_count,
        "minimum_predictions": MIN_RECENT_PREDICTIONS,
        "rolling_window_sessions": ROLLING_WINDOW,
        "return_selection": selected_return,
        "direction_selection": selected_direction,
        "fallback_reason": fallback_reason,
        "selection_method": "recent_two_calendar_months_walk_forward_v1",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SELECTION_FILE, index=False, encoding="utf-8-sig")
    SELECTION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n===== ADAPTIVE MODEL SELECTION =====")
    print(f"Recent out-of-sample predictions: {result['recent_prediction_count']}")
    print(f"Evaluation period: {result['evaluation_start_date']} to {result['evaluation_end_date']}")
    print(f"Selected return model: {selected_return['family']} / {selected_return['window_mode']}")
    print(f"Selected direction model: {selected_direction['family']} / {selected_direction['window_mode']}")
    if fallback_reason:
        print(f"Selection note: {fallback_reason}")
    print(f"Saved adaptive metrics: {SELECTION_FILE}")
    print(f"Saved selection config: {SELECTION_JSON}")
    return result


def fit_selected_models(df, selection):
    """Fit selected return/direction configurations on their eligible history."""
    df = prepare_features(df)
    known = df.dropna(subset=[TARGET]).copy().reset_index(drop=True)
    latest = df.iloc[[-1]].copy()
    n = len(known)
    output = {}
    for task, config in (("return", selection["return_selection"]), ("direction", selection["direction_selection"])):
        family, mode = config["family"], config["window_mode"]
        start = max(0, n - ROLLING_WINDOW) if mode == "rolling" else 0
        train = known.iloc[start:].dropna(subset=[TARGET]).copy()
        y = train[TARGET].astype(float)
        features = MARKET_FEATURES if family == "market" else SENTIMENT_FEATURES
        if family == "baseline" and task == "return":
            prediction = float(y.mean())
        elif task == "return":
            model = make_regressor()
            model.fit(train[features], y)
            prediction = float(model.predict(latest[features])[0])
        else:
            labels = (y > 0).astype(int)
            if labels.nunique() < 2:
                p_up = float(labels.mean())
            else:
                model = make_classifier()
                model.fit(train[features], labels)
                classes = list(model.named_steps["logisticregression"].classes_)
                p_up = float(model.predict_proba(latest[features])[0, classes.index(1)])
            prediction = p_up
        output[task] = {"family": family, "window_mode": mode, "training_rows": len(train), "prediction": prediction}
    return output
