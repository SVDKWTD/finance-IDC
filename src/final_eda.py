
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = DATA / "final_eda"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "model_dataset.csv", parse_dates=["date"])
pred_path = DATA / "walk_forward_predictions.csv"
pred = pd.read_csv(pred_path, parse_dates=["date"])

sns.set_theme(style="whitegrid")

# 1. Market returns and volatility
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df["date"], df["daily_return_pct"], linewidth=0.7)
axes[0].set_title("NIFTY 50 daily returns")
axes[0].set_ylabel("Return (%)")

rolling_vol = df["daily_return_pct"].rolling(20).std()
axes[1].plot(df["date"], rolling_vol, linewidth=1)
axes[1].set_title("20-session rolling volatility")
axes[1].set_ylabel("Standard deviation (%)")
fig.tight_layout()
fig.savefig(OUT / "market_returns_volatility.png", dpi=160)
plt.close(fig)

# 2. News volume and sentiment
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df["date"], df["article_count"], linewidth=0.8)
axes[0].set_title("Daily news volume")
axes[0].set_ylabel("Article count")

axes[1].plot(df["date"], df["mean_sentiment"], linewidth=0.8)
axes[1].axhline(0, linestyle="--", linewidth=1)
axes[1].set_title("Daily mean FinBERT sentiment")
axes[1].set_ylabel("Positive minus negative probability")
fig.tight_layout()
fig.savefig(OUT / "news_volume_sentiment.png", dpi=160)
plt.close(fig)

# 3. Sentiment versus next-session return
valid = df.dropna(subset=[
    "mean_sentiment", "target_next_session_return_pct"
])
fig, ax = plt.subplots(figsize=(9, 6))
sns.regplot(
    data=valid,
    x="mean_sentiment",
    y="target_next_session_return_pct",
    scatter_kws={"alpha": 0.25, "s": 15},
    line_kws={"color": "red"},
    ax=ax
)
ax.set_title("News sentiment vs next-session NIFTY 50 return")
ax.set_xlabel("Mean FinBERT sentiment")
ax.set_ylabel("Next-session return (%)")
fig.tight_layout()
fig.savefig(OUT / "sentiment_vs_next_return.png", dpi=160)
plt.close(fig)

# 4. Actual versus walk-forward predicted returns
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    pred["date"], pred["actual_return_pct"],
    label="Actual", linewidth=0.8
)
ax.plot(
    pred["date"], pred["sentiment_prediction_pct"],
    label="Market + news + sentiment forecast", linewidth=0.8
)
ax.set_title("Walk-forward forecasts vs actual returns")
ax.set_ylabel("Return (%)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "walk_forward_actual_vs_predicted.png", dpi=160)
plt.close(fig)

# 5. Save concise dataset summary
summary = pd.DataFrame([{
    "start_date": df["date"].min().date(),
    "end_date": df["date"].max().date(),
    "sessions": len(df),
    "missing_sentiment_sessions": int(df["mean_sentiment"].isna().sum()),
    "mean_daily_return_pct": df["daily_return_pct"].mean(),
    "daily_return_std_pct": df["daily_return_pct"].std(),
    "mean_articles_per_session": df["article_count"].mean(),
    "sentiment_return_correlation": valid[
        ["mean_sentiment", "target_next_session_return_pct"]
    ].corr().iloc[0, 1],
    "walk_forward_forecasts": len(pred),
}])
summary.to_csv(OUT / "final_eda_summary.csv", index=False)

print("\n===== FINAL EDA SUMMARY =====")
print(summary.round(4).to_string(index=False))
print("\nPlots and summary saved to:")
print(OUT)
print("\nGenerated plots:")
for path in sorted(OUT.glob("*.png")):
    print(path.name)
