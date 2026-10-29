# NIFTY 50 Forecasting Using Financial News Sentiment

## 1. Project Overview

This project investigates whether financial-news sentiment and news activity improve the prediction of next-session NIFTY 50 returns and market direction beyond historical market features and simple baseline models.

It combines market-price collection, financial-news scraping, sentiment analysis using FinBERT, feature engineering, forecasting, and chronological model evaluation.

The project also includes a **monthly news-sampling experiment** to investigate whether limiting the number of news articles processed during 2026 affects forecasting performance.

This is an exploratory research project, not a validated trading system or investment recommendation.

## 2. Research Objectives

The project aims to:

- Collect historical and recent NIFTY 50 market data.
- Collect financial-news headlines relevant to the Indian stock market.
- Quantify news sentiment using a pretrained financial-language model.
- Combine market variables, news volume, source diversity, and sentiment features.
- Predict next-session market returns and UP/DOWN direction.
- Compare market-only models against models incorporating news features.
- Evaluate the effect of monthly news sampling while retaining the full-corpus model as the baseline.
- Track prospective forecasts and evaluate them when the corresponding market outcomes become available.

## 3. Technology Stack

- **Python** — data collection, processing, modelling, and orchestration
- **pandas and NumPy** — data manipulation and numerical computation
- **yfinance** — market-price collection
- **Google News RSS** — financial-news collection
- **Hugging Face Transformers / ProsusAI FinBERT** — financial sentiment analysis
- **scikit-learn** — regression, classification, and evaluation
- **SQLite** — persistent storage of market data, news records, and forecast history
- **Matplotlib and Seaborn** — data visualization
- **Jupyter Notebook** — exploratory and comparative analysis
- **Git and GitHub** — version control

## 4. Repository Structure

```text
finance-IDC-Project/
├── README.md
├── context.md
├── decisions.md
├── INTEGRATION_NOTES.md
├── run_pipeline.py
├── nifty50_news_sampling_analysis.ipynb
├── src/
│   ├── collect_prices.py
│   ├── collect_recent_news.py
│   ├── import_collected_data.py
│   ├── align_news_market.py
│   ├── analyze_news_sentiment.py
│   ├── build_model_dataset.py
│   ├── generate_forecast.py
│   ├── record_forecast_outcome.py
│   ├── evaluate_walk_forward.py
│   └── monthly_sampling_experiment.py
└── data/
    ├── raw/
    ├── processed/
    └── market.db
```

The structure above highlights the principal files. The `src/` directory may contain additional data-quality, feature-ablation, statistical-testing, and experimental modelling scripts.

Generated data and the SQLite database are stored locally and are excluded from version control where configured in `.gitignore`.

## 5. Main Forecasting Workflow

The primary workflow is orchestrated by `run_pipeline.py`.

Run it from the project root with the `finance-IDC` Conda environment activated:

```bash
python run_pipeline.py
```

The pipeline executes the following stages in order.

### Stage 1 — Collect market prices

**Script:** `src/collect_prices.py`

Downloads and updates NIFTY 50 price data, including OHLC prices, adjusted closing prices, and trading volume. The processing workflow excludes the incomplete current market session from the completed-session dataset.

### Stage 2 — Collect recent financial news

**Script:** `src/collect_recent_news.py`

Collects recent financial-news headlines using Google News RSS, maintains previously collected records, and deduplicates against historical URLs where available.

### Stage 3 — Import collected data

**Script:** `src/import_collected_data.py`

Imports collected news and market records into the local SQLite database, `data/market.db`.

### Stage 4 — Align news with market sessions

**Script:** `src/align_news_market.py`

Aligns news publication timestamps with market-session dates and constructs daily news-market features. The output includes news volume, source diversity, market variables, and next-session return targets.

Timestamp alignment is an approximation of information availability and must be checked against the intended forecast cutoff to prevent look-ahead leakage.

### Stage 5 — Score news sentiment

**Script:** `src/analyze_news_sentiment.py`

Uses the pretrained `ProsusAI/finbert` model to score financial headlines. Cached headline scores are reused where possible, and newly encountered headlines are scored incrementally.

The script generates article-level sentiment scores and daily sentiment aggregates, including:

- Mean sentiment
- Median sentiment
- Positive-news fraction
- Negative-news fraction
- Neutral-news fraction

### Stage 6 — Build the main model dataset

**Script:** `src/build_model_dataset.py`

Combines market features and daily full-corpus sentiment features into:

`data/processed/model_dataset.csv`

This is the **primary dataset used by the main forecasting pipeline**.

It contains market variables, news counts, source counts, sentiment aggregates, and the next-session return target.

### Stage 7 — Generate forecasts

**Script:** `src/generate_forecast.py`

Fits the configured models and produces next-session return and direction estimates. Forecast records are persisted in SQLite for subsequent evaluation.

### Stage 8 — Evaluate available forecast outcomes

**Script:** `src/record_forecast_outcome.py`

Checks whether previously recorded forecasts have corresponding completed market outcomes and evaluates those that can now be resolved.

Prospective forecast results should be interpreted cautiously when the number of evaluated outcomes is small.

## 6. Main Dataset and Data Flow

The main model dataset is:

`data/processed/model_dataset.csv`

The daily sentiment features used to build it come from:

`data/processed/news_sentiment_daily.csv`

Other important intermediate files include:

| File | Purpose |
|---|---|
| `data/raw/nifty50_prices.csv` | Raw market-price data |
| `data/raw/recent_news_2026.csv` | Recently collected news |
| `data/processed/news_market_alignment.csv` | Article-level news-market alignment |
| `data/processed/news_market_daily.csv` | Daily market and news features |
| `data/processed/news_sentiment_scored.csv` | Article-level sentiment scores |
| `data/processed/news_sentiment_daily.csv` | Full-corpus daily sentiment aggregates |
| `data/processed/model_dataset.csv` | Main full-corpus modelling dataset |
| `data/market.db` | SQLite database for stored data and forecast history |

The precise collection period and row counts can change as new data is collected.

## 7. Monthly News-Sampling Experiment

**Script:** `src/monthly_sampling_experiment.py`

The sampling experiment evaluates the effect of reducing news volume in 2026 without replacing the primary dataset.

### Sampling strategy

- Retain historical articles from 2021–2025.
- For 2026, retain at most 500 articles per calendar month.
- Where a month contains more than 500 articles, select a reproducible random sample using a fixed seed strategy.
- Retain months with fewer than 500 articles in full.
- Recalculate daily sentiment aggregates from the sampled article corpus.
- Rebuild a separate sampled model dataset.
- Evaluate the original and sampled datasets using the same chronological walk-forward procedure and compare predictions on shared dates.

This experiment changes the news-derived features while keeping the market observations and target construction consistent.

### Important separation of datasets

The experiment produces a separate sampled dataset:

`data/processed/model_dataset_sampled_monthly_2026.csv`

The main pipeline continues to use:

`data/processed/model_dataset.csv`

**The sampled dataset is used for comparison, not as a replacement for the full-corpus baseline.**

### Sampling experiment outputs

| File | Purpose |
|---|---|
| `data/processed/news_sentiment_sampled_monthly_2026.csv` | Article-level sampled news corpus |
| `data/processed/news_sentiment_daily_sampled_monthly_2026.csv` | Daily sentiment calculated from sampled articles |
| `data/processed/model_dataset_sampled_monthly_2026.csv` | Sampled modelling dataset |
| `data/processed/monthly_sampling_return_comparison.csv` | Return-prediction comparison metrics |
| `data/processed/monthly_sampling_direction_comparison.csv` | Direction-prediction comparison metrics |
| `data/processed/monthly_sampling_paired_predictions.csv` | Original and sampled predictions matched by date |

Run the experiment independently:

```bash
python src/monthly_sampling_experiment.py
```

This command generates the sampled data and runs the comparison. It is separate from `run_pipeline.py`, so running the main forecasting pipeline does not automatically replace or regenerate the sampling experiment's outputs.

## 8. Comparative Analysis Notebook

**Notebook:** `nifty50_news_sampling_analysis.ipynb`

The notebook is intended to analyze the generated datasets and experimental outputs. It includes:

- Dataset integrity checks
- Market-price and return exploration
- News-volume and source-diversity analysis
- Full-corpus versus sampled sentiment comparisons
- Walk-forward performance comparisons
- Paired prediction and error analysis
- Missing-value and correlation checks
- A consolidated summary of findings

Place the notebook in the repository root. After running the pipeline and sampling experiment, open it in Jupyter:

```bash
jupyter notebook nifty50_news_sampling_analysis.ipynb
```

Run the notebook cells in order. Its input paths must point to the files under `data/processed/`.

The notebook analyzes existing outputs; it does not replace the pipeline or automatically retrain the models unless additional modelling code is explicitly added.

## 9. Forecast Targets and Models

### Return prediction

The regression target is the percentage close-to-close return for the next available trading session:

`target_next_session_return_pct`

The evaluation compares configured market-only and news-augmented models against a simple mean-return baseline.

### Direction prediction

A separate classification model estimates whether the next-session return is positive or negative. Directional accuracy and Brier score are used to evaluate predictions.

### Walk-forward evaluation

The project uses time-ordered walk-forward evaluation to simulate forecasting on successive future dates. This is preferable to randomly splitting time-series observations because it preserves chronological ordering.

Performance should be compared on the same forecast dates and under the same evaluation procedure. Return MAE and RMSE, directional accuracy, and Brier score measure different aspects of performance and should be interpreted together.

## 10. Current Findings and Limitations

In the recorded monthly-sampling experiment, both the full-corpus and sampled evaluations produced 432 forecasts on shared dates spanning 27 January 2025 to 27 October 2026.

The observed results showed that:

- The mean-return baseline had lower return MAE than the evaluated news-augmented models.
- The monthly-sampled news models performed slightly better than their corresponding full-corpus variants on several reported metrics.
- Directional performance differed between model variants, and higher directional accuracy did not necessarily correspond to a better Brier score.
- The available prospective forecast history was too small to establish reliable real-world predictive performance.

These results are exploratory. Small metric differences do not establish statistically significant or persistent improvement. Further validation should examine statistical uncertainty, data leakage, information availability at forecast time, robustness across time periods, and performance against appropriate baselines.

## 11. Setup and Reproducibility

The project was developed on Windows using Anaconda Prompt and a Conda environment named `finance-IDC`.

Activate the environment:

```bash
conda activate finance-IDC
```

Navigate to the project root:

```bat
cd /d C:\Users\surya\Documents\finance-IDC-Project
```

Run the main workflow:

```bash
python run_pipeline.py
```

Run the separate sampling experiment:

```bash
python src/monthly_sampling_experiment.py
```

Open the analysis notebook:

```bash
jupyter notebook nifty50_news_sampling_analysis.ipynb
```

Install the project's dependencies using the dependency file provided in the repository, if available. Otherwise, consult the imports in the scripts and install the required packages in the active Conda environment.

The first run may require downloading the pretrained FinBERT model. Network access may also be needed for market-price and news collection.

For reproducible experiments, preserve the sampling seed, model configuration, input data snapshot, and evaluation period.

## 12. Data, Privacy, and Version Control

The `data/` directory contains generated data, intermediate CSV files, local databases, and experiment outputs. These files may be large and can change after every collection or processing run.

The repository is configured to keep generated data out of version control. Do not commit API credentials, tokens, `.env` files, local database files, or virtual environments.

To reproduce the analysis, first generate the required data locally, then run the sampling experiment and notebook.

## 13. Research Disclaimer

This project is intended for academic and exploratory analysis. Forecasts are estimates, not investment advice. Historical walk-forward results do not guarantee future performance, and no trading strategy or reliable predictive edge has been established by the results reported here.
