from pathlib import Path
from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

TICKER = "^NSEI"
START_DATE = "2021-01-01"
END_DATE = date.today() + timedelta(days=1)  # Exclusive
OVERLAP_DAYS = 7

RAW_FILE = Path("data/raw/nifty50_prices.csv")
PROCESSED_FILE = Path("data/processed/nifty50_processed.csv")
SUMMARY_FILE = Path("data/processed/nifty50_eda_summary.csv")
MONTHLY_FILE = Path("data/processed/nifty50_monthly_summary.csv")
EDA_FILE = Path("data/processed/nifty50_eda.png")


def normalize_price_frame(df):
    """Normalize yfinance output to the columns used by this project."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Recent yfinance versions can return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    if "Date" not in df.columns:
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df = df.reset_index()

    df.columns = [
        str(col).strip().replace(" ", "_") for col in df.columns
    ]

    if "Date" not in df.columns:
        raise ValueError(
            "Could not find a Date column in the downloaded price data."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")

    return df


def download_prices():
    """
    Download only recent prices and merge them with the saved history.
    The first run downloads the full history; later runs use a 7-day overlap.
    """
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        existing = pd.read_csv(RAW_FILE)
        existing = normalize_price_frame(existing)

        if existing.empty:
            download_start = pd.Timestamp(START_DATE).date()
            print("Existing raw price file had no usable rows; rebuilding history.")
        else:
            last_saved_date = existing["Date"].max().date()
            download_start = max(
                date.fromisoformat(START_DATE),
                last_saved_date - timedelta(days=OVERLAP_DAYS),
            )
            print(f"Existing historical price rows: {len(existing):,}")
            print(f"Last saved trading date: {last_saved_date}")
            print(f"Refreshing recent overlap from: {download_start}")
    else:
        existing = pd.DataFrame()
        download_start = date.fromisoformat(START_DATE)
        print("No saved raw price file found; performing initial historical download.")

    print(f"Downloading {TICKER}")
    print(f"Download date range: {download_start} to {END_DATE - timedelta(days=1)}")

    downloaded = yf.download(
        TICKER,
        start=download_start.isoformat(),
        end=END_DATE.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        multi_level_index=False,
    )

    downloaded = normalize_price_frame(downloaded)

    if downloaded.empty and existing.empty:
        raise RuntimeError(
            "No market data returned and no saved history exists. "
            "Check your internet connection and Yahoo Finance availability."
        )

    if downloaded.empty:
        print("No new price rows returned. Keeping the existing history.")
        combined = existing.copy()
    else:
        print(f"Rows returned by Yahoo Finance: {len(downloaded):,}")
        if existing.empty:
            combined = downloaded.copy()
        else:
            combined = pd.concat(
                [existing, downloaded],
                ignore_index=True,
                sort=False,
            )

        # The downloaded overlap takes precedence for dates it contains.
        # Keep the downloaded version for overlapping dates: downloaded rows
        # were appended after the existing history.
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date")

    combined = combined.sort_values("Date").reset_index(drop=True)
    combined.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")

    print(f"\nTotal unique historical price rows saved: {len(combined):,}")
    print(f"Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    print(f"Raw data saved to: {RAW_FILE.resolve()}")
    print("\nFirst rows:")
    print(combined.head(3).to_string(index=False))
    print("\nLast rows:")
    print(combined.tail(3).to_string(index=False))

    return combined



def clean_and_analyze(df):
    # Recalculate derived columns over the complete locally stored history.
    clean = normalize_price_frame(df)

    price_columns = [
        col for col in [
            "Open", "High", "Low", "Close", "Adj_Close", "Volume"
        ]
        if col in clean.columns
    ]

    for col in price_columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    print("\nMissing values before cleaning:")
    print(clean[price_columns].isna().sum().to_string())

    clean = clean.dropna(subset=["Close"]).copy()

    # Keep only dates strictly before today.
    # Today's Yahoo Finance row may represent an unfinished session.
    today = pd.Timestamp(date.today())
    latest_downloaded_date = clean["Date"].max()

    clean = clean[clean["Date"] < today].copy()
    clean = clean.sort_values("Date").reset_index(drop=True)

    print(f"\nToday's date: {today.date()}")
    print(f"Latest date in downloaded data: {latest_downloaded_date.date()}")
    if not clean.empty:
        print(f"Latest completed-date row retained: {clean['Date'].max().date()}")

    if clean.empty:
        raise RuntimeError(
            "No completed price rows remain after excluding today's and future dates."
        )

    # Returns must be recalculated over the full retained chronology.
    clean["Daily_Return"] = clean["Close"].pct_change()
    clean["Daily_Return_Pct"] = clean["Daily_Return"] * 100

    if {"High", "Low"}.issubset(clean.columns):
        clean["Daily_Range"] = clean["High"] - clean["Low"]

    if {"Open", "Close"}.issubset(clean.columns):
        clean["Intraday_Change_Pct"] = (
            (clean["Close"] - clean["Open"]) / clean["Open"] * 100
        )

    clean["Year"] = clean["Date"].dt.year
    clean["Month"] = clean["Date"].dt.month
    clean["Year_Month"] = clean["Date"].dt.to_period("M").astype(str)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_FILE, index=False, encoding="utf-8-sig")

    numeric_columns = clean.select_dtypes(include="number").columns
    summary = clean[numeric_columns].describe().T
    summary["missing_values"] = clean[numeric_columns].isna().sum()
    summary.to_csv(SUMMARY_FILE, encoding="utf-8-sig")

    monthly = (
        clean.groupby("Year_Month")
        .agg(
            first_date=("Date", "min"),
            last_date=("Date", "max"),
            first_close=("Close", "first"),
            last_close=("Close", "last"),
            average_close=("Close", "mean"),
            minimum_close=("Close", "min"),
            maximum_close=("Close", "max"),
            trading_days=("Close", "count"),
        )
    )

    monthly["Monthly_Return_Pct"] = (
        (monthly["last_close"] / monthly["first_close"] - 1) * 100
    )

    if "Volume" in clean.columns:
        monthly["average_volume"] = clean.groupby("Year_Month")["Volume"].mean()

    monthly.to_csv(MONTHLY_FILE, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(clean["Date"], clean["Close"])
    axes[0].set_title("NIFTY 50 Daily Closing Price")
    axes[0].set_ylabel("Index points")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(clean["Date"], clean["Daily_Return_Pct"], linewidth=0.7)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_title("Daily Percentage Returns")
    axes[1].set_ylabel("Return (%)")
    axes[1].grid(True, alpha=0.3)

    if "Volume" in clean.columns:
        axes[2].plot(clean["Date"], clean["Volume"], linewidth=0.7)
        axes[2].set_title("Daily Volume")
        axes[2].set_ylabel("Volume")
    else:
        axes[2].hist(clean["Daily_Return_Pct"].dropna(), bins=50)
        axes[2].set_title("Distribution of Daily Returns")
        axes[2].set_xlabel("Return (%)")

    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()

    EDA_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EDA_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n===== QUICK EDA =====")
    print("Clean rows:", len(clean))
    print("First date:", clean["Date"].min().date())
    print("Last date:", clean["Date"].max().date())
    print("Duplicate dates:", clean["Date"].duplicated().sum())
    print("Missing closing prices:", clean["Close"].isna().sum())

    print("\nClosing-price statistics:")
    print(clean["Close"].describe().round(2).to_string())

    print("\nDaily-return statistics (%):")
    print(clean["Daily_Return_Pct"].describe().round(4).to_string())

    print("\nYearly row counts:")
    print(clean.groupby("Year").size().to_string())

    print("\nFiles saved:")
    for path in [PROCESSED_FILE, SUMMARY_FILE, MONTHLY_FILE, EDA_FILE]:
        print(path.resolve())

    return clean



if __name__ == "__main__":
    prices = download_prices()
    clean_and_analyze(prices)
