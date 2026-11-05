from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

START_DATE = "2021-01-01"
END_DATE = date.today() + timedelta(days=1)
OVERLAP_DAYS = 7

ASSETS = {
    "nifty50": "^NSEI",
    "gold": "GC=F",
    "usd_inr": "INR=X",
    "brent": "BZ=F",
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_asset(name, ticker):
    path = RAW_DIR / f"{name}_prices.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["Date"])
        existing["Date"] = pd.to_datetime(
            existing["Date"], errors="coerce"
        ).dt.tz_localize(None).dt.normalize()

        existing = existing.dropna(subset=["Date"])
        existing = existing.drop_duplicates("Date", keep="last")

        if not existing.empty:
            start = max(
                date.fromisoformat(START_DATE),
                existing["Date"].max().date()
                - timedelta(days=OVERLAP_DAYS),
            )
        else:
            start = date.fromisoformat(START_DATE)
    else:
        existing = pd.DataFrame()
        start = date.fromisoformat(START_DATE)

    print(f"\nDownloading {name}: {ticker}")

    downloaded = yf.download(
        ticker,
        start=start.isoformat(),
        end=END_DATE.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        multi_level_index=False,
    )

    if downloaded.empty:
        if existing.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        print("No new data returned; keeping existing history.")
        combined = existing
    else:
        downloaded = downloaded.reset_index()

        if isinstance(downloaded.columns, pd.MultiIndex):
            downloaded.columns = [
                col[0] for col in downloaded.columns
            ]

        downloaded["Date"] = pd.to_datetime(
            downloaded["Date"], errors="coerce"
        ).dt.tz_localize(None).dt.normalize()

        downloaded = downloaded.dropna(subset=["Date"])

        if existing.empty:
            combined = downloaded
        else:
            combined = pd.concat(
                [existing, downloaded],
                ignore_index=True,
                sort=False,
            )

        combined = combined.drop_duplicates(
            "Date", keep="last"
        )

    combined = combined.sort_values("Date").reset_index(drop=True)

    # Use only completed calendar dates.
    combined = combined[
        combined["Date"] < pd.Timestamp(date.today())
    ].copy()

    combined.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(combined):,} rows to {path}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")

    return combined


def process_nifty50_prices(df_raw: pd.DataFrame) -> pd.DataFrame:
    """ISSUE 2 FIX: Clean NIFTY prices and derive return columns expected downstream."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "nifty50_processed.csv"

    df = df_raw.dropna(subset=["Close"]).copy()
    df = df[df["Close"] > 0].copy()
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)

    # Match exact Adj Close column naming convention
    adj_col = "Adj Close" if "Adj Close" in df.columns else ("Adj_Close" if "Adj_Close" in df.columns else "Close")
    df["Adj_Close"] = df[adj_col]

    # Calculate returns and range metrics
    df["Daily_Return"] = df["Close"].diff()
    df["Daily_Return_Pct"] = df["Close"].pct_change() * 100.0
    df["Daily_Range"] = df["High"] - df["Low"]
    df["Intraday_Change_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100.0

    # Format Date column for consistent CSV export
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    cols_to_keep = [
        "Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume",
        "Daily_Return", "Daily_Return_Pct", "Daily_Range", "Intraday_Change_Pct"
    ]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    processed_df = df[available_cols].copy()

    processed_df.to_csv(processed_path, index=False)
    print(f"\n[ISSUE 2 FIX] Exported cleaned price features ({len(processed_df):,} rows) to {processed_path}")
    return processed_df


def main():
    nifty_df = None
    for name, ticker in ASSETS.items():
        downloaded = download_asset(name, ticker)
        if name == "nifty50":
            nifty_df = downloaded

    if nifty_df is not None:
        process_nifty50_prices(nifty_df)


if __name__ == "__main__":
    main()