
from pathlib import Path

import pandas as pd
import yfinance as yf


def main():
    ticker = "^NSEI"

    print(f"Downloading historical data for {ticker}...")

    prices = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if prices.empty:
        raise RuntimeError(
            "No market data returned. Check your internet connection."
        )

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices = prices.reset_index()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "nifty50_prices.csv"
    prices.to_csv(output_file, index=False)

    print(f"Rows downloaded: {len(prices)}")
    print(f"Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
