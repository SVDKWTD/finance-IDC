from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "news_market_daily.csv"
OUTPUT = ROOT / "data" / "processed" / "model_ready.csv"

ASSET_COLUMNS = [
    "gold_return_pct",
    "usd_inr_return_pct",
    "brent_return_pct",
]

TARGET = "target_next_session_return_pct"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, parse_dates=["date", "target_next_session_date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("Input rows:", len(df))
    print("Duplicate dates:", df["date"].duplicated().sum())

    # Use only external-market returns known from the previous NIFTY session.
    for col in ASSET_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required feature column missing: {col}")

        lagged_col = col.replace("_return_pct", "_prev_session_return_pct")
        df[lagged_col] = df[col].shift(1)

    # Remove contemporaneous external returns to prevent accidental leakage.
    df = df.drop(columns=ASSET_COLUMNS)

    # These columns identify the row/target; they are not model features.
    excluded = {"date", TARGET, "target_next_session_date"}
    feature_cols = [c for c in df.columns if c not in excluded]

    # Confirm target is exactly the next session's NIFTY return.
    expected_target = df["daily_return_pct"].shift(-1)
    target_diff = (df[TARGET] - expected_target).abs().dropna()
    if not target_diff.empty and target_diff.max() > 1e-8:
        raise ValueError("Target check failed: target is not next-session return.")

    # Confirm each lagged asset feature equals the prior row's raw return.
    for raw_col in ASSET_COLUMNS:
        lagged_col = raw_col.replace("_return_pct", "_prev_session_return_pct")
        expected = pd.read_csv(INPUT)[raw_col].shift(1)
        actual = df[lagged_col]
        check = (actual - expected).abs().dropna()
        if not check.empty and check.max() > 1e-8:
            raise ValueError(f"Lag check failed for {lagged_col}")

    # Keep rows with complete features and a known target.
    before = len(df)
    ready = df.dropna(subset=feature_cols + [TARGET]).copy()
    dropped = before - len(ready)

    if ready.empty:
        raise ValueError("No complete rows remain after removing missing values.")

    # Ensure output is strictly chronological.
    ready = ready.sort_values("date").reset_index(drop=True)
    ready.to_csv(OUTPUT, index=False)

    print("\n===== MODEL DATASET REPORT =====")
    print("Output:", OUTPUT)
    print("Rows retained:", len(ready))
    print("Rows dropped for missing feature/target values:", dropped)
    print("Date range:", ready["date"].min().date(), "to", ready["date"].max().date())
    print("Feature columns:", feature_cols)
    print("Target:", TARGET)
    print("Missing values in output:", int(ready.isna().sum().sum()))
    print("Target check: PASS")
    print("Lagged external-return checks: PASS")
    print("Chronological order:", ready["date"].is_monotonic_increasing)

    print("\nFirst 3 rows:")
    print(ready.head(3).to_string(index=False))
    print("\nLast 3 rows:")
    print(ready.tail(3).to_string(index=False))

if __name__ == "__main__":
    main()