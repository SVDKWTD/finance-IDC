from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market.db"
DAILY_PATH = ROOT / "data" / "processed" / "news_market_daily.csv"
ALIGNMENT_PATH = ROOT / "data" / "processed" / "news_market_alignment.csv"

ASSET_FILES = {
    "gold_return_pct": ROOT / "data" / "raw" / "gold_prices.csv",
    "usd_inr_return_pct": ROOT / "data" / "raw" / "usd_inr_prices.csv",
    "brent_return_pct": ROOT / "data" / "raw" / "brent_prices.csv",
}


def find_column(df, candidates, description):
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        f"Could not find {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def load_raw_asset_returns(path):
    if not path.exists():
        print(f"  NOT CHECKED: raw file not found: {path}")
        return None

    raw = pd.read_csv(path)

    date_col = find_column(
        raw,
        ["date", "datetime", "timestamp", "time"],
        "date",
    )
    close_col = find_column(
        raw,
        ["close", "adj_close", "adjusted_close", "price", "value"],
        "close/price",
    )

    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    raw["close_check"] = pd.to_numeric(raw[close_col], errors="coerce")

    raw = (
        raw.dropna(subset=["date", "close_check"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    raw["raw_return_pct"] = raw["close_check"].pct_change() * 100
    return raw[["date", "raw_return_pct"]]


def main():
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Daily dataset not found: {DAILY_PATH}")

    daily = pd.read_csv(DAILY_PATH)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    print("\n===== DAILY DATASET CHECK =====")
    print("Rows:", len(daily))
    print("Duplicate dates:", daily["date"].duplicated().sum())
    print("Date range:", daily["date"].min(), "to", daily["date"].max())

    expected_sessions = 1442
    print(
        "Session count:",
        "PASS" if len(daily) == expected_sessions else f"CHECK (expected {expected_sessions})",
    )

    required = [
        "date",
        "close",
        "daily_return_pct",
        "target_next_session_return_pct",
        "target_next_session_date",
    ]
    missing_columns = [col for col in required if col not in daily.columns]
    print("Missing required columns:", missing_columns)

    if missing_columns:
        raise ValueError("Daily dataset is missing required columns; cannot continue.")

    # 1. Check daily return calculation against previous row's close.
    expected_return = daily["close"].pct_change() * 100
    return_diff = (daily["daily_return_pct"] - expected_return).abs()
    return_diff = return_diff.dropna()

    print("\nDaily return calculation:")
    print("  Max absolute difference (percentage points):",
          return_diff.max() if not return_diff.empty else "not checked")
    print("  PASS" if return_diff.empty or return_diff.max() < 0.01 else "CHECK")

    # 2. Check that the target is the next row's daily return.
    expected_target = daily["daily_return_pct"].shift(-1)
    target_diff = (
        daily["target_next_session_return_pct"] - expected_target
    ).abs().dropna()

    print("\nNext-session target:")
    print("  Max absolute difference (percentage points):",
          target_diff.max() if not target_diff.empty else "not checked")
    print("  PASS" if target_diff.empty or target_diff.max() < 0.01 else "CHECK")

    expected_target_date = daily["date"].shift(-1)
    target_date_matches = (
        daily["target_next_session_date"].fillna("").astype(str)
        == expected_target_date.fillna("").astype(str)
    )
    print("  Target dates match next row:", bool(target_date_matches.all()))

    # 3. Show missing values, including their dates.
    print("\nMissing values by column:")
    print(daily.isna().sum().to_string())

    for col in [
        "daily_return_pct",
        "gold_return_pct",
        "usd_inr_return_pct",
        "brent_return_pct",
        "target_next_session_return_pct",
    ]:
        if col in daily.columns:
            missing_dates = daily.loc[daily[col].isna(), "date"].tolist()
            print(f"  {col} missing dates: {missing_dates}")

    # 4. Compare article totals with the database and alignment output.
    print("\nArticle-count checks:")

    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            db_total = pd.read_sql_query(
                "SELECT COUNT(*) AS n FROM news", conn
            )["n"].iloc[0]
        print("  Articles in SQLite:", db_total)
    else:
        db_total = None
        print("  SQLite database not found.")

    if ALIGNMENT_PATH.exists():
        alignment = pd.read_csv(ALIGNMENT_PATH)
        print("  Rows in alignment file:", len(alignment))

        if "market" in alignment.columns:
            print("  Alignment rows by market:")
            print(alignment["market"].value_counts().to_string())
    else:
        alignment = None
        print("  Alignment file not found.")

    print("  Article-count columns in daily file:")
    article_cols = [c for c in daily.columns if c.startswith("article_count_")]
    print(daily[article_cols].sum().to_string())

    # 5. Compare daily asset returns with returns calculated from raw prices.
    print("\n===== RAW ASSET RETURN CHECK =====")
    for daily_col, raw_path in ASSET_FILES.items():
        print(f"\n{daily_col}:")

        if daily_col not in daily.columns:
            print("  Column missing from daily dataset.")
            continue

        raw_returns = load_raw_asset_returns(raw_path)
        if raw_returns is None:
            continue

        merged = daily[["date", daily_col]].merge(
            raw_returns,
            on="date",
            how="left",
        )

        comparable = merged.dropna(subset=[daily_col, "raw_return_pct"])
        if comparable.empty:
            print("  No comparable non-missing returns.")
            continue

        difference = (
            comparable[daily_col] - comparable["raw_return_pct"]
        ).abs()

        print("  Comparable dates:", len(comparable))
        print("  Max absolute difference (percentage points):", difference.max())
        print("  Matching within 0.01 percentage points:",
              int((difference < 0.01).sum()), "/", len(difference))

        if difference.max() >= 0.01:
            print("  CHECK: daily returns differ from raw-file calculation.")
        else:
            print("  PASS: values match the raw-file calculation.")

    # 6. Check for zero or negative NIFTY volume. Informational only for an index.
    if "volume" in daily.columns:
        zero_volume = daily[daily["volume"] == 0]
        print("\nNIFTY zero-volume rows:", len(zero_volume))
        if not zero_volume.empty:
            print(zero_volume[["date", "volume"]].to_string(index=False))
        print("  Informational: NIFTY is an index; do not treat zero volume alone as a bad price row.")

    print("\n===== VERIFICATION COMPLETE =====")
    print(
        "Timing limitation: this script checks return calculations, not whether "
        "gold/Brent/USD-INR values were available by each NIFTY close. "
        "That requires checking the source timestamps and alignment rules."
    )


if __name__ == "__main__":
    main()