# Finance + News Analysis

## Project Overview

This project is being developed for IDC409, Topic 3: Web Scraping I — Finance + News.

The goal is to collect Indian financial market data and news, process the data, explore patterns, and evaluate machine-learning models for predicting next-day market direction and returns.

The project uses free data sources and open-source Python libraries.

## Current Data Sources

- **Market prices:** Yahoo Finance via `yfinance`, initially using the NIFTY 50 index (`^NSEI`).
- **Financial news:** Google News RSS.
- **Sentiment analysis:** VADER sentiment analysis applied to news headlines.

Historical news coverage may be incomplete. The availability and time coverage of each source will be documented as the pipeline develops.

## Project Structure

```text
finance-IDC-Project/
├── src/
│   ├── collect_prices.py
│   ├── collect_news.py
│   ├── process_data.py
│   └── analyze_sentiment.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
├── environment.yml
├── README.md
└── .gitignore
```

The structure will be extended with SQLite database helpers, a unified collection pipeline, cleaning and feature-engineering scripts, and analysis notebooks.

## Environment Setup

Activate the existing Conda environment:

```powershell
conda activate finance-news
```

Install dependencies from the environment specification when creating a new environment:

```powershell
conda env create -f environment.yml
```

Run the current data pipeline from the project root:

```powershell
python src/collect_prices.py
python src/collect_news.py
python src/process_data.py
python src/analyze_sentiment.py
```

## Planned Work

1. Add SQLite storage for collected data.
2. Integrate collection while avoiding unnecessary duplicate downloads.
3. Clean and export data for analysis.
4. Perform exploratory data analysis (EDA).
5. Engineer market and news features.
6. Evaluate next-day direction classification and return regression.
7. Compare models against naive baselines using chronological train/test splits.

## Data and Reproducibility

Raw and processed datasets are local project outputs and should not be committed to Git unless a small, appropriate sample is explicitly selected. Secrets must never be committed.

The project will document its data sources, limitations, processing decisions, and model evaluation methods as development continues.
