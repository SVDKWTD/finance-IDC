

# Finance-IDC: Financial News Sentiment and NIFTY 50 Forecasting

An experimental data pipeline for studying whether financial-news sentiment and news volume provide useful information for forecasting the next trading session of the NIFTY 50.

> **Research status:** The pipeline runs end to end, but successful execution does not establish predictive value. Current evaluation has not shown reliable improvement over a simple mean-return baseline. Treat forecasts as experimental outputs, not investment advice.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Question and Objectives](#2-research-question-and-objectives)
3. [Prediction Tasks](#3-prediction-tasks)
4. [System Architecture](#4-system-architecture)
5. [Data Sources](#5-data-sources)
6. [Repository Structure](#6-repository-structure)
7. [Environment Setup](#7-environment-setup)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Pipeline Stages](#9-pipeline-stages)
10. [News-Market Alignment](#10-news-market-alignment)
11. [Sentiment Analysis](#11-sentiment-analysis)
12. [Feature Engineering and Targets](#12-feature-engineering-and-targets)
13. [Forecasting and Model Selection](#13-forecasting-and-model-selection)
14. [Forecast Outcome Tracking](#14-forecast-outcome-tracking)
15. [Evaluation and Interpretation](#15-evaluation-and-interpretation)
16. [Generated Data and Outputs](#16-generated-data-and-outputs)
17. [Exploratory Data Analysis](#17-exploratory-data-analysis)
18. [Reproducibility and Data Integrity](#18-reproducibility-and-data-integrity)
19. [Limitations](#19-limitations)
20. [Future Work](#20-future-work)
21. [Disclaimer](#21-disclaimer)

---

## 1. Project Overview

Financial markets respond to a mixture of price movements, corporate announcements, macroeconomic developments, geopolitical events, investor expectations, and other information. Some of this information is published in financial news before it is fully reflected in market prices.

This project builds a Python-based workflow to investigate whether financial-news sentiment and news volume contain useful information for predicting the next trading session of the NIFTY 50.

The project combines:

- Historical and recent NIFTY 50 market data
- Financial-news headlines collected from news feeds
- Timestamp-based news-to-market alignment
- Finance-domain sentiment analysis using FinBERT
- Market and sentiment feature engineering
- Regression and classification models
- Chronological evaluation
- Persistent forecast records and outcome tracking

The project is designed as an experimental research workflow. It is intended to make data collection, preprocessing, modelling, and evaluation traceable and repeatable—not to provide a ready-made trading strategy.

## 2. Research Question and Objectives

### Main research question

**Does financial-news sentiment provide useful predictive information for forecasting the next NIFTY 50 trading session beyond information already available from market data?**

The project does not assume that sentiment must improve a forecast. A finding that sentiment adds no measurable predictive value is a valid research result.

### Objectives

1. Collect NIFTY 50 market data and financial-news headlines.
2. Store collected information in a local database and structured files.
3. Associate news with the market session at which it could have become available.
4. Score headlines using a finance-domain language model.
5. Construct market, news-volume, and sentiment features.
6. Compare market-only models with models that also use news information.
7. Evaluate predictions chronologically rather than randomly mixing past and future observations.
8. Record forecasts before their outcomes are known and evaluate them when the target session becomes available.

## 3. Prediction Tasks

The workflow considers three short-horizon prediction tasks.

| Task | Formulation | Output |
|---|---|---|
| Next-session return | Regression | Predicted percentage return |
| Next-session direction | Binary classification | Probability that the next return is positive |
| Next-session high movement | Binary classification | Probability of an unusually large absolute return |

The volatility event is defined relative to a historical threshold. It is not a claim that the market will necessarily move in a particular direction.

## 4. System Architecture

The routine pipeline is orchestrated by `run_pipeline.py`.

```text
External market data ──┐
                       ├──> Import and store
Financial-news feeds ──┘          │
                                  ▼
                         News-market alignment
                                  │
                                  ▼
                         FinBERT sentiment scoring
                                  │
                                  ▼
                       Daily feature construction
                                  │
                                  ▼
                         Model dataset generation
                                  │
                                  ▼
                         Forecast generation
                                  │
                                  ▼
                       Forecast outcome tracking
```

The workflow separates data acquisition, processing, modelling, and evaluation so that intermediate outputs can be inspected independently.

## 5. Data Sources

### Market data

The project uses NIFTY 50 historical prices and external market series, including:

- NIFTY 50
- Gold
- USD/INR
- Brent crude oil

Market data is used to construct returns, lagged returns, rolling volatility, and related features.

### Financial news

Financial-news headlines are collected from news feeds, including Google News RSS search results. The collected archive is not a complete census of financial journalism. Its coverage depends on the queries, source availability, feed behavior, and collection dates.

### Sentiment model

Headlines are scored using **FinBERT**, a language model trained for financial text. The model produces positive, negative, and neutral sentiment scores or labels.

The sentiment analysis is headline-based. It does not establish the sentiment of the full article, the author’s intent, or the eventual market impact.

## 6. Repository Structure

The following shows the main project organization. Generated files may vary depending on which pipeline stages have been run.

```text
finance-IDC-Project/
│
├── data/
│   ├── raw/
│   │   ├── recent_news_2026.csv
│   │   └── nifty50_prices.csv
│   │
│   ├── processed/
│   │   ├── nifty50_processed.csv
│   │   ├── news_market_alignment.csv
│   │   ├── news_market_daily.csv
│   │   ├── news_sentiment_scored.csv
│   │   ├── news_sentiment_daily.csv
│   │   ├── news_sentiment_training.csv
│   │   ├── sampling_audit.csv
│   │   ├── model_dataset.csv
│   │   ├── adaptive_model_selection.csv
│   │   ├── adaptive_model_selection.json
│   │   ├── latest_forecast.csv
│   │   └── forecast_outcome_history.csv
│   │
│   └── market.db
│
├── notebooks/
│   └── Finance_IDC_Project_Walkthrough.ipynb
│
├── src/
│   ├── collect_recent_news.py
│   ├── collect_prices.py
│   ├── import_collected_data.py
│   ├── align_news_market.py
│   ├── analyze_news_sentiment.py
│   ├── build_training_dataset.py
│   ├── generate_forecast.py
│   ├── adaptive_model_selection.py
│   ├── record_forecast_outcome.py
│   └── ...
│
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Important files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Runs the routine pipeline stages in sequence |
| `src/collect_prices.py` | Refreshes market-price data |
| `src/collect_recent_news.py` | Collects recent financial-news headlines |
| `src/import_collected_data.py` | Imports collected data into the local database |
| `src/align_news_market.py` | Aligns headlines with market sessions |
| `src/analyze_news_sentiment.py` | Scores headlines and aggregates sentiment |
| `src/build_training_dataset.py` | Builds the dataset used for modelling |
| `src/generate_forecast.py` | Generates and stores the latest forecast |
| `src/adaptive_model_selection.py` | Selects a forecasting strategy |
| `src/record_forecast_outcome.py` | Evaluates forecasts once outcomes are available |
| `notebooks/Finance_IDC_Project_Walkthrough.ipynb` | Guided project walkthrough and analysis |

## 7. Environment Setup

### Requirements

- Python 3.10 or later
- Conda or another Python environment manager
- Internet access for refreshing data
- Sufficient memory for FinBERT inference
- CUDA-compatible GPU is optional

### Clone the repository

```bash
git clone https://github.com/SVDKWTD/finance-IDC.git
cd finance-IDC
```

### Create and activate the environment

```bash
conda create -n finance-IDC python=3.11
conda activate finance-IDC
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Depending on the installed PyTorch build and hardware, PyTorch may need a separate installation. Use the official PyTorch installation instructions for the appropriate CPU or CUDA configuration.

## 8. Running the Pipeline

Run the complete routine workflow from the repository root:

```bash
python run_pipeline.py
```

The pipeline executes its configured stages sequentially and stops if a required stage fails.

### Run using existing collected data

If the pipeline supports the skip-refresh option in the current version:

```bash
python run_pipeline.py --skip-refresh
```

This is useful when the required local data already exists and a fresh collection is not needed.

### Run selected stages

If the current version supports stage selection:

```bash
python run_pipeline.py --only align,sentiment,dataset,forecast,outcomes
```

Use the options implemented in the checked-out `run_pipeline.py`; command-line options may change as the project evolves.

### Explore the notebook

Open `Finance_IDC_Project_Walkthrough.ipynb` in Jupyter Notebook or JupyterLab. Select the `finance-IDC` environment and execute the notebook from top to bottom.

The notebook is intended for explanation and analysis. The routine pipeline remains the source of the generated production outputs.

## 9. Pipeline Stages

### 9.1 Market-price collection

The market collection stage retrieves NIFTY 50 and external market-price data and updates the local data store.

The pipeline uses price data to calculate daily returns and derived market features. Since market data providers may revise or backfill historical values, the exact downloaded data should be treated as a versioned input to an experiment.

### 9.2 Financial-news collection

The news collection stage retrieves recent headlines matching configured financial search terms.

Collected records may include fields such as headline, publication timestamp, source, URL, and market category. Duplicate or repeated headlines may occur across feeds and collection runs; the import and processing stages handle the project’s configured deduplication logic.

The collector’s coverage is query-dependent. It should not be interpreted as a complete representation of all financial news.

### 9.3 Data import and storage

Collected data is imported into the project’s SQLite database and/or processed files. The database provides persistent storage for news records, prices, forecasts, and forecast outcomes.

### 9.4 News-market alignment

News timestamps are converted to Indian Standard Time and associated with a market session using configured market cutoffs.

This step is important because a headline published after a market’s forecast cutoff cannot be used to explain or predict that same cutoff’s next-session outcome.

### 9.5 Sentiment scoring

FinBERT assigns sentiment scores to headlines. Article-level scores are then aggregated into daily features, such as average sentiment and the proportion of positive, negative, and neutral headlines.

The sentiment model provides a numerical representation of headline language. It does not directly measure investor positioning, market expectations, or causal impact.

### 9.6 Dataset construction

Market and news features are combined into a modelling dataset. The dataset contains historical predictors and target values for the next trading session.

The data-building stage also handles missing values and prepares the rows required by the forecasting scripts.

### 9.7 Forecast generation

The forecasting stage fits the configured models and generates estimates for the latest eligible forecast origin. It stores the result so the forecast can later be evaluated against the realized market outcome.

### 9.8 Outcome tracking

The outcome-tracking stage checks whether the target trading session has occurred and whether its market data is available. When it is, the stored forecast can be compared with the realized result.

Forecasts from different model versions should remain distinguishable during evaluation.

## 10. News-Market Alignment

News alignment is one of the most consequential parts of the project. A model can appear predictive if it uses information that was not available at the time the forecast would have been made.

The alignment workflow uses publication timestamps and market-specific cutoffs to determine the earliest eligible NIFTY session for each headline.

### Core principle

**A feature may be used only if it would have been available at the forecast origin.**

For a NIFTY forecast made at 15:45 IST, news published after 15:45 cannot be treated as information available to that forecast. Such news must be assigned to a later eligible session.

### Non-trading days

News published on weekends or market holidays is assigned to the next eligible trading session under the alignment rules.

### External-market features

External market data must follow the same availability principle. A same-date daily return is valid only if its underlying price was available by the forecast cutoff.

For example, a USD/INR close recorded at 17:00 IST is not available to a NIFTY forecast made at 15:45 IST on that same date. The implementation must use an earlier available value or otherwise lag the feature.

The timestamp convention of each data provider should be checked rather than inferred from the date label alone.

## 11. Sentiment Analysis

FinBERT is used to score financial-news headlines. The project aggregates the headline-level outputs into daily sentiment features.

Typical sentiment features include:

- Mean sentiment score
- Positive headline fraction
- Negative headline fraction
- Neutral headline fraction
- News article count
- Number of unique sources

News volume and sentiment are different signals. A day with many headlines is not necessarily positive or negative; it may simply reflect an unusually active news cycle.

### Headline-only limitation

A headline may omit context, qualifications, or the eventual details of an event. The resulting sentiment score is therefore a noisy proxy for the information conveyed by the full article.

## 12. Feature Engineering and Targets

### Market features

The modelling workflow uses market-derived predictors such as:

- Daily return
- Lagged returns
- Rolling volatility
- Momentum or recent-return summaries
- External market returns

These features represent information from recent price behavior and related markets.

### News features

News-derived predictors include:

- Article count
- Unique source count
- Mean sentiment
- Positive, negative, and neutral fractions
- Lagged sentiment

### Target variables

The main regression target is the next-session NIFTY 50 return:

\[
r_{t+1} = 100 \times \left(\frac{P_{t+1}}{P_t} - 1\right)
\]

where \(P_t\) is the NIFTY 50 close for the forecast-origin session and \(P_{t+1}\) is the next eligible session’s close.

The direction target indicates whether the next-session return is positive. The volatility target indicates whether the absolute next-session return exceeds a threshold derived from historical observations.

Targets describe future outcomes; they must never be included among the predictors for the same forecast.

## 13. Forecasting and Model Selection

The project compares market-only information with market-plus-news information.

### Model groups

- **Market-only models:** use market-derived features.
- **Market-plus-sentiment models:** add news volume and sentiment features.
- **Baseline models:** provide a simple reference point against which more complex models can be judged.

The project also includes an adaptive model-selection component. Its purpose is to choose a strategy based on historical evaluation rather than assuming one model is always best.

### Why chronological evaluation matters

Random train-test splits can mix future observations into training data and past observations into testing data. That is inappropriate for forecasting.

Evaluation should preserve time order: models are trained using past observations and tested on later observations. Any model selection, feature selection, preprocessing, or threshold estimation must also be restricted to information available at the relevant training point.

### Forecast records

A forecast is stored with its origin date and model version. This makes it possible to distinguish a prediction made before an outcome was known from a retrospective prediction generated after the fact.

## 14. Forecast Outcome Tracking

The outcome tracker checks whether a forecast’s target session has become available.

When the realized outcome is available, the system can record:

- Actual next-session return
- Actual market direction
- Actual absolute return
- Volatility-event outcome
- Evaluation status

This creates a prospective record of forecasts and outcomes. It is more informative than repeatedly fitting a model and reporting only its in-sample performance.

## 15. Evaluation and Interpretation

Forecasting results should be compared against simple baselines and reported with the evaluation period and number of observations.

For regression, useful metrics include:

- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- \(R^2\)

For direction classification, useful metrics include:

- Accuracy
- Brier score
- Confusion matrix
- Calibration analysis, when enough observations are available

For volatility classification, the evaluation should include metrics appropriate for imbalanced outcomes and should compare against a baseline that accounts for event frequency.

### Current interpretation

The project’s existing evaluation has not established that the market-plus-sentiment model reliably improves on the mean-return baseline. Any small differences in direction accuracy should be interpreted alongside probability scores, sample size, and chronological evaluation design.

A pipeline that completes successfully is evidence that the software workflow runs. It is not evidence that the model has a profitable or statistically reliable predictive edge.

## 16. Generated Data and Outputs

The exact set of generated files depends on which stages have been run. Common outputs include:

| Output | Description |
|---|---|
| `nifty50_processed.csv` | Processed NIFTY 50 market data |
| `news_market_alignment.csv` | Headline-level market-session assignments |
| `news_market_daily.csv` | Daily news counts and market-level aggregates |
| `news_sentiment_scored.csv` | Headline-level sentiment scores |
| `news_sentiment_daily.csv` | Daily aggregated sentiment features |
| `news_sentiment_training.csv` | Sentiment data prepared for modelling |
| `sampling_audit.csv` | Audit of the news-sampling process |
| `model_dataset.csv` | Combined feature and target dataset |
| `adaptive_model_selection.csv` | Model-selection records |
| `adaptive_model_selection.json` | Serialized model-selection information |
| `latest_forecast.csv` | Most recent generated forecast |
| `forecast_outcome_history.csv` | Forecasts and their realized outcomes |

The SQLite database stores persistent project records. Avoid editing generated outputs manually unless the purpose is a controlled experiment; regenerate them from the upstream stages instead.

## 17. Exploratory Data Analysis

The next analysis stage should characterize the data before attempting to improve the models.

Recommended EDA questions include:

1. How many market sessions and news records are present?
2. How does news volume vary by day, month, and source?
3. How are positive, negative, and neutral headlines distributed?
4. Does sentiment vary with news volume?
5. How do market returns and volatility vary over time?
6. Are missing values concentrated in particular periods or features?
7. Do feature distributions change between historical and recent data?
8. Are there unusual dates, duplicate records, or extreme values?
9. Does the monthly sampling procedure change the distribution of headlines or sentiment?
10. Are same-date features demonstrably available by the forecast cutoff?

EDA should first describe the data and its limitations. It should not be used to select features or thresholds using the full dataset and then report those choices as if they were made before the test period.

## 18. Reproducibility and Data Integrity

For a reproducible run:

- Use the intended Conda environment.
- Run scripts from the repository root.
- Keep raw inputs separate from generated outputs.
- Record the collection date and data range.
- Preserve forecast origin dates and model versions.
- Use chronological evaluation.
- Regenerate downstream datasets after changing upstream alignment or feature logic.
- Inspect warnings and failed stages rather than relying only on the final success message.

External data sources can change, and collection feeds may return different records on later runs. Exact reproducibility may therefore require preserving the raw data used for each experiment.

## 19. Limitations

This project is an experimental forecasting system, not a production trading system.

1. **News coverage:** Search-feed collection does not capture every relevant article.
2. **Headline-only sentiment:** Headlines can omit important context.
3. **Sampling effects:** Sampling can alter the distribution of publishers, topics, and event types.
4. **Historical/live differences:** The historical training corpus and the live forecast input may not have identical collection or sampling procedures.
5. **Feature availability:** Daily market data must be checked against the forecast cutoff to prevent look-ahead leakage.
6. **Small prospective sample:** A limited number of stored forecasts cannot establish reliable performance.
7. **Probability calibration:** Predicted probabilities have not been demonstrated to be well calibrated.
8. **Model-selection noise:** Adaptive selection can overfit historical evaluation results.
9. **Trading costs:** Transaction costs, slippage, execution, position sizing, and portfolio risk are not fully modelled.
10. **No causal identification:** Predictive association does not show that news sentiment causes market movement.

## 20. Future Work

Potential next steps include:

- Conducting a structured exploratory data analysis of the raw and processed datasets.
- Auditing feature availability at each forecast cutoff.
- Comparing market-only and market-plus-sentiment models on identical chronological folds.
- Evaluating feature ablations to identify which news features, if any, add information.
- Measuring probability calibration and uncertainty.
- Expanding prospective forecast tracking before drawing conclusions.
- Testing whether results are robust across time periods and alternative baselines.
- Improving news coverage and preserving source metadata.
- Evaluating transaction costs only after a credible forecasting signal has been established.

## 21. Disclaimer

This project is for academic and experimental purposes only. It does not provide financial advice, and its forecasts should not be used as the sole basis for investment or trading decisions.

