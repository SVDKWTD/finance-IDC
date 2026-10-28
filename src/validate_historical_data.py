
from pathlib import Path
import pandas as pd

# Input and output files
INPUT_FILE = Path("data/raw/historical_news_2021_2025.csv")
OUTPUT_FILE = Path("data/processed/news_2021_2025_clean.csv")
REPORT_FILE = Path("data/processed/news_quality_report.csv")

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"  # Exclusive: includes all of 2025

# Load the original dataset
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

required_columns = ["title", "published_at", "url"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Record data-quality issues before cleaning
df["published_at"] = pd.to_datetime(
    df["published_at"], errors="coerce", utc=True
)

df["title"] = df["title"].fillna("").astype(str).str.strip()
df["url"] = df["url"].fillna("").astype(str).str.strip()

print("\nMissing or invalid publication dates:",
      df["published_at"].isna().sum())
print("Empty titles:", (df["title"] == "").sum())
print("Empty URLs:", (df["url"] == "").sum())
print("Duplicate URLs:", df["url"].duplicated().sum())

# Keep only records with usable titles, URLs, and dates
clean = df[
    (df["title"] != "")
    & (df["url"] != "")
    & df["published_at"].notna()
].copy()

# Keep the requested five-year period
clean = clean[
    (clean["published_at"] >= pd.Timestamp(START_DATE, tz="UTC"))
    & (clean["published_at"] < pd.Timestamp(END_DATE, tz="UTC"))
].copy()

# Remove duplicate articles by URL
clean = clean.drop_duplicates(subset=["url"], keep="first")

# Derive a consistent publication date and year
clean["publication_date"] = (
    clean["published_at"].dt.strftime("%Y-%m-%d")
)
clean["year"] = clean["published_at"].dt.year

# Produce a yearly quality report
yearly = (
    clean.groupby("year")
    .size()
    .reindex(range(2021, 2026), fill_value=0)
    .rename("clean_articles")
    .reset_index()
)

yearly["target"] = 2500
yearly["shortfall"] = (
    yearly["target"] - yearly["clean_articles"]
).clip(lower=0)

# Save cleaned data and report; leave raw data unchanged
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
yearly.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

print("\nClean article count:", len(clean))
print("\nYearly counts:")
print(yearly.to_string(index=False))
print("\nMonthly counts:")
print(
    clean.groupby(clean["published_at"].dt.strftime("%Y-%m"))
    .size()
    .to_string()
)
print(f"\nClean data saved to: {OUTPUT_FILE.resolve()}")
print(f"Quality report saved to: {REPORT_FILE.resolve()}")
