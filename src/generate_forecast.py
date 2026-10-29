from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

from adaptive_model_selection import select_strategy, fit_selected_models

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset_sampled_monthly_2026.csv"
OUTPUT = ROOT / "data" / "processed" / "latest_forecast.csv"
DB = ROOT / "data" / "market.db"

MARKET_FEATURES = [
    "daily_return_pct", "return_lag1", "return_lag2",
    "return_lag5", "volatility_5", "article_count",
    "unique_sources",
]
SENTIMENT_FEATURES = MARKET_FEATURES + [
    "mean_sentiment", "positive_fraction",
    "negative_fraction", "neutral_fraction", "sentiment_lag1",
]
VOLATILITY_FEATURES = [
    "daily_return_pct", "return_lag1", "return_lag2", "return_lag5",
    "return_lag10", "volatility_5", "volatility_20", "momentum_5",
    "momentum_10", "article_count", "unique_sources", "mean_sentiment",
    "median_sentiment", "positive_fraction", "negative_fraction",
    "neutral_fraction",
]
TARGET = "target_next_session_return_pct"

# Frozen volatility specification selected before prospective evaluation.
VOL_WINDOW = 252
VOL_MIN_HISTORY = 60
VOL_PERCENTILE = 0.80
VOL_MODEL_VERSION = "monthly_sampled_2026_adaptive_walkforward_v1"


def add_column_if_missing(con, table, column, sql_type):
    existing = {
        row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def add_features(df):
    df = df.sort_values("date").reset_index(drop=True).copy()
    r = pd.to_numeric(df["daily_return_pct"], errors="coerce")
    df["return_lag1"] = r.shift(1)
    df["return_lag2"] = r.shift(2)
    df["return_lag5"] = r.shift(5)
    df["return_lag10"] = r.shift(10)
    df["volatility_5"] = r.rolling(5, min_periods=5).std()
    df["volatility_20"] = r.rolling(20, min_periods=10).std()
    df["momentum_5"] = r.rolling(5, min_periods=5).sum()
    df["momentum_10"] = r.rolling(10, min_periods=10).sum()
    df["sentiment_lag1"] = pd.to_numeric(df["mean_sentiment"], errors="coerce").shift(1)
    df["abs_return"] = r.abs()
    # Threshold uses only absolute returns known through the feature date.
    df["volatility_threshold_pct"] = df["abs_return"].rolling(
        VOL_WINDOW, min_periods=VOL_MIN_HISTORY
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
        raise FileNotFoundError(f"Model dataset not found: {DATA}")
    df = pd.read_csv(DATA, parse_dates=["date"])
    required = {"date", "daily_return_pct", TARGET, "mean_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"model_dataset.csv is missing required columns: {sorted(missing)}")
    df = add_features(df)

    history = df.dropna(subset=[TARGET]).copy()
    latest = df.iloc[[-1]].copy()
    if latest["date"].isna().any():
        raise ValueError("Latest session has no valid date.")
    if latest["date"].iloc[0] != df["date"].max():
        raise ValueError("Latest session is not the final dataset row.")
    train = history.dropna(subset=[TARGET]).copy()
    if len(train) < 300:
        raise ValueError("Not enough historical rows to train safely.")

    # Adaptive selector uses only recent, completed, out-of-sample targets.
    selection = select_strategy(df)
    selected_models = fit_selected_models(df, selection)
    selected_return = selected_models["return"]
    selected_direction = selected_models["direction"]

    y = train[TARGET].astype(float)
    y_direction = (y > 0).astype(int)
    if y_direction.nunique() < 2:
        raise ValueError("Historical target contains only one direction class.")

    # Existing return regression models retained.
    market_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    market_model.fit(train[MARKET_FEATURES], y)
    sentiment_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    sentiment_model.fit(train[SENTIMENT_FEATURES], y)

    direction_market = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     LogisticRegression(max_iter=1000, random_state=42))
    direction_market.fit(train[MARKET_FEATURES], y_direction)
    direction_sentiment = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(max_iter=1000, random_state=42))
    direction_sentiment.fit(train[SENTIMENT_FEATURES], y_direction)

    forecast_origin = pd.Timestamp(latest["date"].iloc[0]).strftime("%Y-%m-%d")
    market_return = float(market_model.predict(latest[MARKET_FEATURES])[0])
    sentiment_return = float(sentiment_model.predict(latest[SENTIMENT_FEATURES])[0])
    market_p_up = float(direction_market.predict_proba(latest[MARKET_FEATURES])[0, 1])
    sentiment_p_up = float(direction_sentiment.predict_proba(latest[SENTIMENT_FEATURES])[0, 1])

    # New volatility classifier: only historically labelled outcomes are used for training.
    volatility_train = train.loc[train["is_volatile_next_session"].notna()].copy()
    volatility_available = (
        len(volatility_train) >= 300
        and volatility_train["is_volatile_next_session"].nunique() >= 2
        and pd.notna(latest["volatility_threshold_pct"].iloc[0])
    )
    volatility_probability = None
    volatility_threshold = (
        float(latest["volatility_threshold_pct"].iloc[0])
        if pd.notna(latest["volatility_threshold_pct"].iloc[0]) else None
    )
    volatility_alert_20 = None
    volatility_alert_30 = None
    volatility_train_rate = None
    if volatility_available:
        vol_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        )
        y_vol = volatility_train["is_volatile_next_session"].astype(int)
        vol_model.fit(volatility_train[VOLATILITY_FEATURES], y_vol)
        classes = list(vol_model.named_steps["logisticregression"].classes_)
        volatility_probability = float(
            vol_model.predict_proba(latest[VOLATILITY_FEATURES])[0, classes.index(1)]
        )
        volatility_alert_20 = int(volatility_probability >= 0.20)
        volatility_alert_30 = int(volatility_probability >= 0.30)
        volatility_train_rate = float(y_vol.mean())

    sentiment_available = pd.notna(latest["mean_sentiment"].iloc[0])
    warnings = []
    if not sentiment_available:
        warnings.append("Sentiment unavailable for latest session")
    if abs(sentiment_p_up - 0.5) < 0.10:
        warnings.append("Direction probability is close to 50%")
    warnings.append("Overall historical evaluation has not established that feature models beat the mean-return baseline")
    warnings.append("Direction probabilities have not been demonstrated to be calibrated")
    if not volatility_available:
        warnings.append("Volatility forecast unavailable: insufficient labelled history or threshold")
    warnings.append("Same-date news features are valid only if available by forecast cutoff")
    if selection.get("fallback_reason"):
        warnings.append("Adaptive selector: " + selection["fallback_reason"])
    warnings.append("Adaptive selection is a research heuristic; recent two-month performance may be noisy")

    forecast = {
        "dataset": "Monthly sampled 2026 (500-article monthly cap)",
        "dataset_file": DATA.name,
        "forecast_origin_date": forecast_origin,
        "target_session_date": None,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_rows": int(len(train)),
        "market_predicted_return_pct": market_return,
        "market_direction": "UP" if market_return >= 0 else "DOWN",
        "market_probability_up": market_p_up,
        "sentiment_predicted_return_pct": sentiment_return,
        "sentiment_direction": "UP" if sentiment_return >= 0 else "DOWN",
        "sentiment_probability_up": sentiment_p_up,
        "sentiment_available": int(sentiment_available),
        "selected_model_name": selected_return["family"],
        "selected_window_mode": selected_return["window_mode"],
        "selected_predicted_return_pct": float(selected_return["prediction"]),
        "selected_direction_model_name": selected_direction["family"],
        "selected_direction_window_mode": selected_direction["window_mode"],
        "selected_direction_probability_up": float(selected_direction["prediction"]),
        "adaptive_selection_date": selection["selection_date"],
        "adaptive_recent_prediction_count": int(selection["recent_prediction_count"]),
        "adaptive_selection_method": selection["selection_method"],
        "warnings": " | ".join(warnings),
        "model_version": VOL_MODEL_VERSION,
        "volatility_probability": volatility_probability,
        "volatility_threshold_pct": volatility_threshold,
        "volatility_alert_20": volatility_alert_20,
        "volatility_alert_30": volatility_alert_30,
        "volatility_training_rate": volatility_train_rate,
        "actual_abs_return_pct": None,
        "actual_volatile": None,
    }

    with sqlite3.connect(DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS forecast_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_origin_date TEXT NOT NULL,
                target_session_date TEXT,
                created_at_utc TEXT NOT NULL,
                training_rows INTEGER NOT NULL,
                market_predicted_return_pct REAL NOT NULL,
                market_direction TEXT NOT NULL,
                market_probability_up REAL NOT NULL,
                sentiment_predicted_return_pct REAL NOT NULL,
                sentiment_direction TEXT NOT NULL,
                sentiment_probability_up REAL NOT NULL,
                sentiment_available INTEGER NOT NULL,
                warnings TEXT,
                model_version TEXT NOT NULL,
                actual_return_pct REAL,
                actual_direction TEXT,
                evaluated_at_utc TEXT,
                UNIQUE(forecast_origin_date, model_version)
            )
        """)
        # Safe additive migration for existing SQLite databases.
        add_column_if_missing(con, "forecast_history", "volatility_probability", "REAL")
        add_column_if_missing(con, "forecast_history", "volatility_threshold_pct", "REAL")
        add_column_if_missing(con, "forecast_history", "volatility_alert_20", "INTEGER")
        add_column_if_missing(con, "forecast_history", "volatility_alert_30", "INTEGER")
        add_column_if_missing(con, "forecast_history", "volatility_training_rate", "REAL")
        add_column_if_missing(con, "forecast_history", "actual_abs_return_pct", "REAL")
        add_column_if_missing(con, "forecast_history", "actual_volatile", "INTEGER")
        add_column_if_missing(con, "forecast_history", "selected_model_name", "TEXT")
        add_column_if_missing(con, "forecast_history", "selected_window_mode", "TEXT")
        add_column_if_missing(con, "forecast_history", "selected_predicted_return_pct", "REAL")
        add_column_if_missing(con, "forecast_history", "selected_direction_model_name", "TEXT")
        add_column_if_missing(con, "forecast_history", "selected_direction_window_mode", "TEXT")
        add_column_if_missing(con, "forecast_history", "selected_direction_probability_up", "REAL")
        add_column_if_missing(con, "forecast_history", "adaptive_selection_date", "TEXT")
        add_column_if_missing(con, "forecast_history", "adaptive_recent_prediction_count", "INTEGER")
        add_column_if_missing(con, "forecast_history", "adaptive_selection_method", "TEXT")

        existing_record = con.execute(
            "SELECT created_at_utc, market_predicted_return_pct, market_probability_up, "
            "sentiment_predicted_return_pct, sentiment_probability_up "
            "FROM forecast_history WHERE forecast_origin_date = ? AND model_version = ?",
            (forecast_origin, VOL_MODEL_VERSION),
        ).fetchone()

        con.execute("""
            INSERT INTO forecast_history (
                forecast_origin_date, target_session_date, created_at_utc, training_rows,
                market_predicted_return_pct, market_direction, market_probability_up,
                sentiment_predicted_return_pct, sentiment_direction, sentiment_probability_up,
                sentiment_available, warnings, model_version, volatility_probability,
                volatility_threshold_pct, volatility_alert_20, volatility_alert_30,
                volatility_training_rate, actual_abs_return_pct, actual_volatile,
                selected_model_name, selected_window_mode, selected_predicted_return_pct,
                selected_direction_model_name, selected_direction_window_mode,
                selected_direction_probability_up, adaptive_selection_date,
                adaptive_recent_prediction_count, adaptive_selection_method
            ) VALUES (
                :forecast_origin_date, :target_session_date, :created_at_utc, :training_rows,
                :market_predicted_return_pct, :market_direction, :market_probability_up,
                :sentiment_predicted_return_pct, :sentiment_direction, :sentiment_probability_up,
                :sentiment_available, :warnings, :model_version, :volatility_probability,
                :volatility_threshold_pct, :volatility_alert_20, :volatility_alert_30,
                :volatility_training_rate, :actual_abs_return_pct, :actual_volatile,
                :selected_model_name, :selected_window_mode, :selected_predicted_return_pct,
                :selected_direction_model_name, :selected_direction_window_mode,
                :selected_direction_probability_up, :adaptive_selection_date,
                :adaptive_recent_prediction_count, :adaptive_selection_method
            )
            ON CONFLICT(forecast_origin_date, model_version) DO NOTHING
        """, forecast)
        saved = con.execute("""
            SELECT created_at_utc, market_predicted_return_pct, market_probability_up,
                   sentiment_predicted_return_pct, sentiment_probability_up,
                   volatility_probability, volatility_threshold_pct,
                   volatility_alert_20, volatility_alert_30, warnings
            FROM forecast_history
            WHERE forecast_origin_date = ? AND model_version = ?
        """, (forecast_origin, VOL_MODEL_VERSION)).fetchone()

    # Report the freshly computed values, not the immutable first-issued DB record.
    print("\n===== OFFICIAL NEXT-SESSION FORECAST: MONTHLY-SAMPLED DATASET =====")
    print(f"Dataset: {DATA.name} (2026 monthly cap: 500 articles/month)")
    print(f"Latest available market session: {forecast_origin}")
    print(f"Historical training rows: {len(train)}")
    print(f"Market-only forecast: {market_return:+.4f}% ({'UP' if market_return >= 0 else 'DOWN'})")
    print(f"Market-only estimated probability of UP: {market_p_up:.1%}")
    print(f"Market + sentiment forecast: {sentiment_return:+.4f}% ({'UP' if sentiment_return >= 0 else 'DOWN'})")
    print(f"Market + sentiment estimated probability of UP: {sentiment_p_up:.1%}")
    if volatility_probability is not None:
        print(f"Next-session volatility probability: {volatility_probability:.1%}")
        print(f"Volatility threshold: {volatility_threshold:.4f}% absolute return")
        print(f"Volatility alert at 20%: {'YES' if volatility_alert_20 else 'NO'}")
        print(f"Volatility alert at 30%: {'YES' if volatility_alert_30 else 'NO'}")
    else:
        print("Volatility forecast unavailable.")

    if existing_record:
        print("\nPROSPECTIVE TRACKING NOTE:")
        print("An earlier forecast for this date/model version already exists.")
        print("The original database record was preserved for honest prospective evaluation.")
        print(f"Original record created at: {saved[0]}")
        print(f"Original saved market-only forecast: {saved[1]:+.4f}% (P(UP)={saved[2]:.1%})")
        print(f"Original saved market+sentiment forecast: {saved[3]:+.4f}% (P(UP)={saved[4]:.1%})")
        print("The fresh values above are not replacing that original record.")
    print(f"Adaptive selected return: {selected_return['family']} / {selected_return['window_mode']} = {selected_return['prediction']:+.4f}%")
    print(f"Adaptive selected direction: {selected_direction['family']} / {selected_direction['window_mode']} (P(UP)={selected_direction['prediction']:.1%})")
    print(f"Adaptive selector recent OOS predictions: {selection['recent_prediction_count']}")
    print("\nWarnings:")
    for warning in warnings:
        print(f"- {warning}")
    print(f"\nForecast history database:\n{DB}")
    print("Target date is resolved from the actual exchange sessions by outcome tracking.")

    # This is the single official latest-forecast artifact. Full-corpus results
    # remain available only through the optional research comparison script.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([forecast]).to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved official capped-dataset forecast to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
