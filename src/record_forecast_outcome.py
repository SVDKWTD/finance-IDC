
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "market.db"
PRICES = ROOT / "data" / "processed" / "nifty50_processed.csv"
OUTPUT = ROOT / "data" / "processed" / "forecast_outcome_history.csv"


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

                con.execute("""
                    UPDATE forecast_history
                    SET target_session_date = ?,
                        actual_return_pct = ?,
                        actual_direction = ?,
                        evaluated_at_utc = ?
                    WHERE id = ?
                      AND actual_return_pct IS NULL
                """, (
                    target_date,
                    actual_return,
                    actual_direction,
                    pd.Timestamp.now(tz="UTC").isoformat(),
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
                actual_return_pct,
                actual_direction,
                model_version
            FROM forecast_history
            WHERE actual_return_pct IS NOT NULL
            ORDER BY target_session_date, forecast_origin_date
        """, con)

    print(f"Total evaluated forecasts: {len(evaluated)}")

    if evaluated.empty:
        print(
            "\nNo outcomes can be evaluated yet. This is normal if the "
            "price file has no trading session after the forecast date."
        )
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(OUTPUT, index=False)

    print(f"\nSaved outcome history to:\n{OUTPUT}")
    print("\nNote: small samples do not establish reliable model performance.")


if __name__ == "__main__":
    main()
