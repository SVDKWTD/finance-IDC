
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_dataset.csv"
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
TARGET = "target_next_session_return_pct"


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Features use only information available by the current session.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    history = df.dropna(subset=[TARGET]).copy()
    latest = df.iloc[[-1]].copy()

    if latest["date"].isna().any():
        raise ValueError("Latest session has no valid date.")

    latest_date = latest["date"].iloc[0]
    if latest_date != df["date"].max():
        raise ValueError("Latest session is not the final dataset row.")

    # Train only on rows whose next-session target is already known.
    train = history.dropna(subset=[TARGET]).copy()
    if len(train) < 300:
        raise ValueError("Not enough historical rows to train safely.")

    y = train[TARGET].astype(float)
    y_direction = (y > 0).astype(int)

    if y_direction.nunique() < 2:
        raise ValueError("Historical target contains only one direction class.")

    # Return model: Ridge with market and news-volume features.
    market_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    market_model.fit(train[MARKET_FEATURES], y)

    sentiment_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    sentiment_model.fit(train[SENTIMENT_FEATURES], y)

    # Direction models output estimates, not guaranteed calibrated probabilities.
    direction_market = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    direction_market.fit(train[MARKET_FEATURES], y_direction)

    direction_sentiment = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    direction_sentiment.fit(train[SENTIMENT_FEATURES], y_direction)

    # The next trading session's date is not known reliably on weekends/holidays.
    # Save the forecast origin and leave the target date blank until resolved.
    forecast_origin = latest_date.strftime("%Y-%m-%d")
    market_return = float(market_model.predict(latest[MARKET_FEATURES])[0])
    sentiment_return = float(
        sentiment_model.predict(latest[SENTIMENT_FEATURES])[0]
    )

    market_p_up = float(
        direction_market.predict_proba(latest[MARKET_FEATURES])[0, 1]
    )
    sentiment_p_up = float(
        direction_sentiment.predict_proba(latest[SENTIMENT_FEATURES])[0, 1]
    )

    # Low model confidence is flagged conservatively based on past evaluation.
    # These probabilities are not calibrated confidence estimates.
    sentiment_available = pd.notna(latest["mean_sentiment"].iloc[0])
    warnings = []
    if not sentiment_available:
        warnings.append("Sentiment unavailable for latest session")
    if abs(sentiment_p_up - 0.5) < 0.10:
        warnings.append("Direction probability is close to 50%")
    warnings.append(
        "Historical walk-forward results did not beat the mean-return baseline"
    )
    warnings.append(
        "Direction probabilities have not been demonstrated to be calibrated"
    )

    if market_return >= 0:
        market_direction = "UP"
    else:
        market_direction = "DOWN"

    if sentiment_return >= 0:
        sentiment_direction = "UP"
    else:
        sentiment_direction = "DOWN"

    forecast = {
        "forecast_origin_date": forecast_origin,
        "target_session_date": None,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_rows": int(len(train)),
        "market_predicted_return_pct": market_return,
        "market_direction": market_direction,
        "market_probability_up": market_p_up,
        "sentiment_predicted_return_pct": sentiment_return,
        "sentiment_direction": sentiment_direction,
        "sentiment_probability_up": sentiment_p_up,
        "sentiment_available": int(sentiment_available),
        "warnings": " | ".join(warnings),
        "model_version": "ridge_logistic_walkforward_v1",
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

        con.execute("""
            INSERT INTO forecast_history (
                forecast_origin_date, target_session_date, created_at_utc,
                training_rows, market_predicted_return_pct, market_direction,
                market_probability_up, sentiment_predicted_return_pct,
                sentiment_direction, sentiment_probability_up,
                sentiment_available, warnings, model_version
            )
            VALUES (
                :forecast_origin_date, :target_session_date, :created_at_utc,
                :training_rows, :market_predicted_return_pct, :market_direction,
                :market_probability_up, :sentiment_predicted_return_pct,
                :sentiment_direction, :sentiment_probability_up,
                :sentiment_available, :warnings, :model_version
            )
            ON CONFLICT(forecast_origin_date, model_version) DO NOTHING
        """, forecast)

        saved = con.execute("""
            SELECT forecast_origin_date, target_session_date,
                   market_predicted_return_pct, market_probability_up,
                   sentiment_predicted_return_pct, sentiment_probability_up,
                   warnings
            FROM forecast_history
            WHERE forecast_origin_date = ?
              AND model_version = ?
        """, (forecast_origin, forecast["model_version"])).fetchone()

    print("\n===== NEXT-SESSION FORECAST =====")
    print(f"Latest available market session: {forecast_origin}")
    print(f"Historical training rows: {len(train)}")
    print(f"Market-only forecast: {saved[2]:+.4f}% ({'UP' if saved[2] >= 0 else 'DOWN'})")
    print(f"Market-only estimated probability of UP: {saved[3]:.1%}")
    print(f"Market + sentiment forecast: {saved[4]:+.4f}% ({'UP' if saved[4] >= 0 else 'DOWN'})")
    print(f"Market + sentiment estimated probability of UP: {saved[5]:.1%}")
    print("\nWarnings:")
    for warning in warnings:
        print(f"- {warning}")
    print("\nForecast history database:")
    print(DB)
    print("\nNote: the target session date must be resolved using the actual exchange calendar.")


if __name__ == "__main__":
    main()
