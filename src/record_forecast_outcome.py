
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "market.db"
PRICES = ROOT / "data" / "processed" / "nifty50_processed.csv"
OUTPUT = ROOT / "data" / "processed" / "forecast_outcome_history.csv"
OFFICIAL_MODEL_VERSIONS = {"monthly_sampled_2026_ridge_logistic_volatility_v1", "monthly_sampled_2026_adaptive_walkforward_v1"}


def add_column_if_missing(con, table, column, sql_type):
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def load_prices():
    """Load the processed price file using its actual column names."""
    if not PRICES.exists():
        raise FileNotFoundError(f"Price file not found: {PRICES}")

    prices = pd.read_csv(PRICES)

    required = {"Date", "Daily_Return_Pct"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(
            f"Price file is missing columns: {sorted(missing)}. "
            f"Available columns: {prices.columns.tolist()}"
        )

    prices["date"] = pd.to_datetime(
        prices["Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    prices["actual_return_pct"] = pd.to_numeric(
        prices["Daily_Return_Pct"], errors="coerce"
    )

    prices = (
        prices.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    return prices


def main():
    if not DB.exists():
        raise FileNotFoundError(f"Database not found: {DB}")

    prices = load_prices()

    print(f"Loaded price rows: {len(prices):,}")
    print(f"Price dates: {prices['date'].min()} to {prices['date'].max()}")

    with sqlite3.connect(DB) as con:
        # Check that the forecast table exists before trying to read it.
        table = con.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'forecast_history'
        """).fetchone()

        if table is None:
            raise RuntimeError(
                "The forecast_history table does not exist. "
                "Run src\\generate_forecast.py first."
            )

        # Additive migration keeps existing historical forecast records intact.
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

        forecasts = pd.read_sql_query("""
            SELECT
                id,
                forecast_origin_date,
                target_session_date,
                market_predicted_return_pct,
                market_direction,
                market_probability_up,
                sentiment_predicted_return_pct,
                sentiment_direction,
                sentiment_probability_up,
                volatility_threshold_pct,
                volatility_probability,
                selected_model_name, selected_window_mode, selected_predicted_return_pct,
                selected_direction_model_name, selected_direction_window_mode,
                selected_direction_probability_up, adaptive_selection_date,
                adaptive_recent_prediction_count, adaptive_selection_method,
                model_version
            FROM forecast_history
            WHERE actual_return_pct IS NULL
            ORDER BY forecast_origin_date, id
        """, con)

        if forecasts.empty:
            print("\nNo unevaluated forecasts found.")
        else:
            updated = 0
            pending = 0

            for _, forecast in forecasts.iterrows():
                origin = str(forecast["forecast_origin_date"])[:10]
                target_value = forecast["target_session_date"]

                # Use a preassigned target date when available.
                # Otherwise, select the first available market session
                # strictly after the forecast origin.
                if pd.isna(target_value) or not str(target_value).strip():
                    later = prices.loc[prices["date"] > origin]

                    if later.empty:
                        pending += 1
                        continue

                    actual = later.iloc[0]
                    target_date = actual["date"]
                else:
                    target_date = str(target_value)[:10]
                    match = prices.loc[prices["date"] == target_date]

                    if match.empty:
                        pending += 1
                        continue

                    actual = match.iloc[0]

                actual_return = actual["actual_return_pct"]

                if pd.isna(actual_return):
                    pending += 1
                    continue

                actual_return = float(actual_return)

                if actual_return > 0:
                    actual_direction = "UP"
                elif actual_return < 0:
                    actual_direction = "DOWN"
                else:
                    actual_direction = "FLAT"

                threshold = forecast.get("volatility_threshold_pct")
                if pd.notna(threshold):
                    actual_abs_return = abs(actual_return)
                    actual_volatile = int(actual_abs_return > float(threshold))
                else:
                    actual_abs_return = None
                    actual_volatile = None

                con.execute("""
                    UPDATE forecast_history
                    SET target_session_date = ?,
                        actual_return_pct = ?,
                        actual_direction = ?,
                        evaluated_at_utc = ?,
                        actual_abs_return_pct = ?,
                        actual_volatile = ?
                    WHERE id = ?
                      AND actual_return_pct IS NULL
                """, (
                    target_date,
                    actual_return,
                    actual_direction,
                    pd.Timestamp.now(tz="UTC").isoformat(),
                    actual_abs_return,
                    actual_volatile,
                    int(forecast["id"]),
                ))

                if con.execute("SELECT changes()").fetchone()[0] > 0:
                    updated += 1

            con.commit()

            print(f"\nNew forecasts evaluated: {updated}")
            print(f"Forecasts still pending: {pending}")

        # Read the full evaluated history, including older forecasts.
        evaluated = pd.read_sql_query("""
            SELECT
                forecast_origin_date,
                target_session_date,
                market_predicted_return_pct,
                market_direction,
                market_probability_up,
                sentiment_predicted_return_pct,
                sentiment_direction,
                sentiment_probability_up,
                volatility_probability,
                volatility_threshold_pct,
                volatility_alert_20,
                volatility_alert_30,
                actual_abs_return_pct,
                actual_volatile,
                actual_return_pct,
                actual_direction,
                selected_model_name, selected_window_mode, selected_predicted_return_pct,
                selected_direction_model_name, selected_direction_window_mode,
                selected_direction_probability_up, adaptive_selection_date,
                adaptive_recent_prediction_count, adaptive_selection_method,
                model_version
            FROM forecast_history
            WHERE actual_return_pct IS NOT NULL
            ORDER BY target_session_date, forecast_origin_date
        """, con)

    # Keep all records in SQLite, but restrict the main reported metrics and CSV
    # to the official capped-dataset model. Legacy full-corpus forecasts remain
    # preserved for research and are not mixed into the official scorecard.
    all_evaluated_count = len(evaluated)
    evaluated = evaluated.loc[evaluated["model_version"].isin(OFFICIAL_MODEL_VERSIONS)].copy()
    print(f"Total evaluated forecasts in database: {all_evaluated_count}")
    print(f"Official capped-model evaluated forecasts: {len(evaluated)}")
    print(f"Legacy/other model forecasts excluded from official metrics: {all_evaluated_count - len(evaluated)}")

    if evaluated.empty:
        print(
            "\nNo capped-model outcomes can be evaluated yet. This is expected "
            "until a saved official forecast has a matching later trading session. "
            "Legacy full-corpus results are intentionally excluded from the official scorecard."
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        evaluated.to_csv(OUTPUT, index=False)
        print(f"Saved empty official outcome history to:\n{OUTPUT}")
        return

    for prefix in ["market", "sentiment"]:
        pred_col = f"{prefix}_predicted_return_pct"
        direction_col = f"{prefix}_direction"
        probability_col = f"{prefix}_probability_up"

        errors = (
            evaluated[pred_col].astype(float)
            - evaluated["actual_return_pct"].astype(float)
        )

        nonflat = evaluated["actual_direction"].isin(["UP", "DOWN"])
        direction_correct = (
            evaluated[direction_col] == evaluated["actual_direction"]
        )

        print(f"\n===== {prefix.upper()} MODEL =====")
        print(f"Return MAE: {errors.abs().mean():.4f}%")
        print(f"Return RMSE: {(errors.pow(2).mean()) ** 0.5:.4f}%")

        if nonflat.any():
            accuracy = direction_correct[nonflat].mean()
            print(
                f"Directional accuracy: {accuracy:.1%} "
                f"({int(nonflat.sum())} non-flat outcomes)"
            )

            valid = nonflat & evaluated[probability_col].notna()
            if valid.any():
                actual_up = (
                    evaluated.loc[valid, "actual_direction"] == "UP"
                ).astype(float)

                probabilities = evaluated.loc[
                    valid, probability_col
                ].astype(float)

                brier = (
                    probabilities - actual_up
                ).pow(2).mean()

                print(f"Brier score: {brier:.4f}")

    selected_rows = evaluated.dropna(subset=["selected_predicted_return_pct"]).copy()
    if not selected_rows.empty:
        selected_errors = selected_rows["selected_predicted_return_pct"].astype(float) - selected_rows["actual_return_pct"].astype(float)
        selected_nonflat = selected_rows["actual_direction"].isin(["UP", "DOWN"])
        selected_actual_up = (selected_rows["actual_direction"] == "UP").astype(float)
        selected_probs = pd.to_numeric(selected_rows["selected_direction_probability_up"], errors="coerce")
        valid_probs = selected_nonflat & selected_probs.notna()
        print("\n===== ADAPTIVELY SELECTED MODEL =====")
        print(f"Evaluated adaptive forecasts: {len(selected_rows)}")
        print(f"Return MAE: {selected_errors.abs().mean():.4f}%")
        print(f"Return RMSE: {(selected_errors.pow(2).mean()) ** 0.5:.4f}%")
        if selected_nonflat.any():
            predicted_up = selected_probs >= 0.5
            actual_up_bool = selected_rows["actual_direction"] == "UP"
            print(f"Direction accuracy: {(predicted_up[valid_probs] == actual_up_bool[valid_probs]).mean():.1%}" if valid_probs.any() else "Direction accuracy: n/a")
            if valid_probs.any():
                print(f"Direction Brier score: {((selected_probs[valid_probs] - selected_actual_up[valid_probs]) ** 2).mean():.4f}")
        print("Selected model counts:")
        print(selected_rows["selected_model_name"].value_counts(dropna=False).to_string())

    volatility_rows = evaluated.dropna(subset=["volatility_probability", "actual_volatile"])
    if not volatility_rows.empty:
        y_vol = volatility_rows["actual_volatile"].astype(int)
        p_vol = volatility_rows["volatility_probability"].astype(float)
        brier_vol = ((p_vol - y_vol) ** 2).mean()
        predicted_vol = p_vol >= 0.30
        tp = int((predicted_vol & (y_vol == 1)).sum())
        fp = int((predicted_vol & (y_vol == 0)).sum())
        fn = int((~predicted_vol & (y_vol == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        print("\n===== VOLATILITY MODEL (0.30 ALERT THRESHOLD) =====")
        print(f"Evaluated volatility forecasts: {len(volatility_rows)}")
        print(f"Volatility Brier score: {brier_vol:.4f}")
        print(f"Volatility precision: {precision:.1%}" if pd.notna(precision) else "Volatility precision: n/a")
        print(f"Volatility recall: {recall:.1%}" if pd.notna(recall) else "Volatility recall: n/a")
        print("These are live/prospective records only where the volatility probability was saved.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(OUTPUT, index=False)

    print(f"\nSaved outcome history to:\n{OUTPUT}")
    print("\nNote: small samples do not establish reliable model performance.")


if __name__ == "__main__":
    main()
