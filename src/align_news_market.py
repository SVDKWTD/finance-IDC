from pathlib import Path
import numpy as np
import pandas as pd

from db import DEFAULT_DB_PATH, get_connection

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

ARTICLE_OUTPUT = DATA_DIR / "news_market_alignment.csv"
DAILY_OUTPUT = DATA_DIR / "news_market_daily.csv"

START_DATE = "2021-01-01"
END_DATE = "2027-01-01"

MARKET_TIMEZONE = "Asia/Kolkata"

# Market-specific information cutoffs, in IST.
MARKET_CLOSES = {
    "nifty": (15, 45),
    "gold": (0, 0),
    "usd_inr": (17, 0),
    "brent": (0, 0),
}

ASSETS = {
    "gold": "gold_prices.csv",
    "usd_inr": "usd_inr_prices.csv",
    "brent": "brent_prices.csv",
}


def load_asset(name, filename):
    path = PROJECT_DIR / "data" / "raw" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} data: {path}\n"
            "Run src/collect_prices.py first."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df.drop_duplicates("Date", keep="last").sort_values("Date")

    df[f"{name}_return_pct"] = df["Close"].pct_change() * 100
    return df[["Date", f"{name}_return_pct"]]


def make_cutoffs(dates, market):
    hour, minute = MARKET_CLOSES[market]
    local_dates = pd.DatetimeIndex(dates).normalize()
    return (
        local_dates
        + pd.Timedelta(hours=hour, minutes=minute)
    ).tz_localize(MARKET_TIMEZONE)


def main():
    with get_connection(DEFAULT_DB_PATH) as connection:
        news = pd.read_sql_query(
            """
            SELECT id, title, published_at, source, url, market
            FROM news
            WHERE published_at IS NOT NULL
            """,
            connection,
        )

        prices = pd.read_sql_query(
            """
            SELECT date, open, high, low, close, volume,
                   adj_close, daily_return, daily_return_pct
            FROM market_prices
            WHERE ticker = '^NSEI'
            ORDER BY date
            """,
            connection,
        )

    if news.empty or prices.empty:
        raise ValueError("The news table or market-price table is empty.")

    news["published_at"] = pd.to_datetime(
        news["published_at"], errors="coerce", utc=True
    )
    news = news.dropna(subset=["published_at"])

    news = news[
        (news["published_at"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (news["published_at"] < pd.Timestamp(END_DATE, tz="UTC"))
    ].copy()

    news["published_at_ist"] = news["published_at"].dt.tz_convert(
        MARKET_TIMEZONE
    )
    news["market"] = (
        news["market"].fillna("nifty").astype(str).str.strip()
    )

    prices["date"] = pd.to_datetime(
        prices["date"], errors="coerce"
    ).dt.normalize()

    prices = (
        prices.dropna(subset=["date", "close"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    nifty_closes = make_cutoffs(prices["date"], "nifty")
    nifty_close_ns = nifty_closes.asi8

    aligned_parts = []

    for market, group in news.groupby("market"):
        if market not in MARKET_CLOSES:
            continue

        market_cutoffs = make_cutoffs(prices["date"], market)
        cutoff_ns = market_cutoffs.asi8

        market_positions = np.searchsorted(
            cutoff_ns,
            group["published_at_ist"].astype("int64").to_numpy(),
            side="left",
        )

        valid = market_positions < len(prices)
        group = group.loc[valid].copy()
        market_positions = market_positions[valid]

        if group.empty:
            continue

        available_ns = cutoff_ns[market_positions]

        nifty_positions = np.searchsorted(
            nifty_close_ns, available_ns, side="left"
        )

        valid_nifty = nifty_positions < len(prices)
        group = group.loc[valid_nifty].copy()
        nifty_positions = nifty_positions[valid_nifty]

        if group.empty:
            continue

        group["market_cutoff_ist"] = market_cutoffs[
            market_positions[valid_nifty]
        ].astype(str)

        group["feature_date"] = (
            prices.iloc[nifty_positions]["date"]
            .dt.strftime("%Y-%m-%d")
            .to_numpy()
        )

        aligned_parts.append(group)

    if aligned_parts:
        aligned_news = pd.concat(aligned_parts, ignore_index=True)
    else:
        aligned_news = news.iloc[0:0].copy()
        aligned_news["market_cutoff_ist"] = pd.Series(dtype=str)
        aligned_news["feature_date"] = pd.Series(dtype=str)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    aligned_news.to_csv(
        ARTICLE_OUTPUT, index=False, encoding="utf-8-sig"
    )

    daily_news = (
        aligned_news.groupby(["feature_date", "market"])
        .agg(
            article_count=("url", "nunique"),
            unique_sources=("source", "nunique"),
        )
        .reset_index()
    )

    if not daily_news.empty:
        daily_news = daily_news.pivot(
            index="feature_date",
            columns="market",
            values=["article_count", "unique_sources"],
        )
        daily_news.columns = [
            f"{metric}_{market}"
            for metric, market in daily_news.columns
        ]
        daily_news = daily_news.reset_index().rename(
            columns={"feature_date": "date"}
        )
        daily_news["date"] = pd.to_datetime(daily_news["date"])
    else:
        daily_news = pd.DataFrame({"date": pd.to_datetime([])})

    daily = prices.merge(daily_news, on="date", how="left")

    count_columns = [
        col for col in daily.columns
        if col.startswith("article_count_")
        or col.startswith("unique_sources_")
    ]
    for column in count_columns:
        daily[column] = daily[column].fillna(0).astype(int)

    for name, filename in ASSETS.items():
        asset = load_asset(name, filename)
        daily = daily.merge(
            asset,
            left_on="date",
            right_on="Date",
            how="left",
            validate="one_to_one",
        )
        daily = daily.drop(columns="Date")
        daily[f"{name}_return_pct"] = daily[
            f"{name}_return_pct"
        ].ffill()

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["target_next_session_return_pct"] = (
        daily["daily_return_pct"].shift(-1)
    )
    daily["target_next_session_date"] = daily["date"].shift(-1)

    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["target_next_session_date"] = (
        daily["target_next_session_date"].dt.strftime("%Y-%m-%d")
    )

    daily.to_csv(DAILY_OUTPUT, index=False, encoding="utf-8-sig")

    print("===== ALIGNMENT REPORT =====")
    print("Trading sessions:", len(daily))
    print("Market date range:", daily["date"].min(), "to", daily["date"].max())
    print("Aligned articles:", len(aligned_news))
    print("Articles by market:")
    print(aligned_news["market"].value_counts().to_string())
    print("Daily dataset saved to:", DAILY_OUTPUT.resolve())


if __name__ == "__main__":
    main()