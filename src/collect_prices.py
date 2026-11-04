
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


def main():
    for name, ticker in ASSETS.items():
        download_asset(name, ticker)


if __name__ == "__main__":
    main()
