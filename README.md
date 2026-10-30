# Finance-IDC: News-Informed NIFTY 50 Forecasting

A data-driven forecasting pipeline that combines historical NIFTY 50 market data with financial news sentiment to investigate whether news-derived information can improve next-session market predictions.

The project integrates financial data collection, news processing, sentiment analysis using FinBERT, time-series feature engineering, machine learning, and walk-forward evaluation into a reproducible Python pipeline.

---

## Overview

Financial markets respond to a combination of historical price movements, economic developments, corporate announcements, investor expectations, and other external events.

This project explores the relationship between financial news and NIFTY 50 market behaviour by combining two primary data sources:

- **Market data:** Historical daily OHLCV data for the NIFTY 50 index.
- **Financial news:** Financial headlines collected through Google News RSS and scored using a pretrained FinBERT model.

The resulting dataset is used to develop and evaluate models for three forecasting tasks:

| Task | Objective | Model |
|---|---|---|
| Return prediction | Predict the next trading session's percentage return | Ridge Regression |
| Direction prediction | Estimate the probability of an UP or DOWN session | Logistic Regression |
| Volatility prediction | Estimate the probability of unusually high next-session volatility | Logistic Regression |

The project also investigates whether sentiment-derived features provide predictive information beyond historical market variables.

## Pipeline Architecture

```text
         DATA COLLECTION
         ┌───────────────┐
         │ Google News   │
         │ RSS           │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │ Financial     │
         │ News Archive  │
         └───────┬───────┘
                 │
                 │
         ┌───────▼───────┐       ┌────────────────┐
         │ News Cleaning │       │ Yahoo Finance  │
         │ & Alignment   │       │ NIFTY 50 Data  │
         └───────┬───────┘       └───────┬────────┘
                 │                       │
                 └───────────┬───────────┘
                             │
                    ┌────────▼────────┐
                    │ SQLite Database │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ FinBERT         │
                    │ Sentiment       │
                    │ Analysis        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Feature         │
                    │ Engineering     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Training Dataset│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Walk-Forward    │
                    │ Model Selection │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Next-Session    │
                    │ Forecast        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Outcome Tracking│
                    └─────────────────┘
```

---

## 1. Data Collection

### Financial News

Financial headlines are collected using Google News RSS through predefined search queries covering:

- NIFTY 50 and the Indian stock market
- Sensex and NSE/BSE
- Indian economy and macroeconomic developments
- RBI, inflation, and interest rates
- Corporate earnings and major Indian companies

The news collector retains relevant metadata:

- Headline
- Publication timestamp
- Publisher/source
- Article URL
- Matched search terms
- Collection timestamp

The collector uses incremental updates with an overlapping recent collection window to discover newly indexed articles while retaining previously collected records.

Duplicate articles are primarily identified using their URLs.

### Market Data

Historical NIFTY 50 index data is obtained through Yahoo Finance using the `yfinance` library and the `^NSEI` ticker.

The collected variables include:

- Open
- High
- Low
- Close
- Adjusted Close
- Volume

Derived variables include daily returns, percentage returns, intraday range, and other price-based features.

The project maintains historical data from January 2021 onward, subject to data availability.

---

## 2. Data Storage

The project uses SQLite for persistent local storage.

The main database is:

`data/market.db`

Important tables include:

| Table | Purpose |
|---|---|
| `news` | Stores collected financial news and metadata |
| `market_prices` | Stores historical market observations |
| `forecast_history` | Stores generated forecasts |
| Forecast outcome tables | Track subsequent actual market outcomes |

CSV files are used for intermediate datasets, processed data, model inputs, and forecast outputs.

The incremental import process uses unique identifiers to prevent duplicate news and price records.

---

## 3. Data Cleaning and Market Alignment

Before sentiment analysis, news records undergo preprocessing.

The implemented cleaning operations include:

- Parsing publication timestamps.
- Converting timestamps to UTC-aware datetime objects.
- Converting publication times to Indian Standard Time (IST).
- Removing invalid timestamps.
- Removing missing or blank headlines.
- Trimming leading and trailing whitespace.
- Deduplicating articles using URLs.
- Using headline and feature date as a fallback deduplication key.

### News-to-Market Alignment

A critical part of the project is assigning news to the correct market session.

The pipeline uses a **15:45 IST information cutoff** for each actual NIFTY 50 trading session.

An article is assigned to the first available trading-session cutoff that occurs after its publication timestamp.

This ensures that news published after the cutoff is not incorrectly included in that session's features.

For example:

- News published at 11:00 IST is assigned to that trading session.
- News published at 16:30 IST is assigned to the next available trading session.
- Weekend and holiday news is assigned to the next available trading session.

The next-session return target is constructed using the subsequent available trading session rather than the next calendar day.

---

## 4. Sentiment Analysis Using FinBERT

The project uses the pretrained **ProsusAI/finbert** model for financial sentiment analysis.

FinBERT is a BERT-based language model adapted to financial text classification.

Unlike traditional sentiment analysis approaches based on word dictionaries, FinBERT uses contextual language representations to classify financial headlines.

### Model Input

The model receives individual financial headlines rather than complete article bodies.

Each headline is passed through the tokenizer and classification model.

The inference configuration includes:

- Maximum sequence length: 512 tokens
- Batch size: 32
- Truncation enabled
- GPU acceleration when CUDA is available

### Sentiment Classification

FinBERT returns three class probabilities:

\[
P(\text{Positive}),\quad
P(\text{Negative}),\quad
P(\text{Neutral})
\]

A continuous sentiment score is calculated as:

\[
S = P(\text{Positive}) - P(\text{Negative})
\]

The resulting score ranges approximately from -1 to +1.

| Score | Interpretation |
|---|---|
| Close to +1 | Strong positive sentiment |
| Close to -1 | Strong negative sentiment |
| Close to 0 | Weak directional sentiment or balanced positive/negative evidence |

The dominant probability determines the discrete sentiment label.

### Incremental Sentiment Scoring

Previously calculated sentiment scores are cached.

When the pipeline runs again, only headlines without an existing valid sentiment score need to be processed.

This avoids repeatedly running FinBERT on the entire historical archive.

### Daily Sentiment Aggregation

Individual article-level scores are aggregated by market feature date.

The resulting variables include:

- Mean sentiment
- Median sentiment
- Positive headline fraction
- Negative headline fraction
- Neutral headline fraction

The three sentiment fractions represent the proportion of headlines assigned to each class.

---

## 5. Historical Sampling and Training Dataset Construction

The project maintains two separate news datasets:

1. The complete scored news archive.
2. A sampled historical archive used for model training.

The complete scored archive is retained permanently.

### Monthly Sampling Strategy

To limit the influence of months containing exceptionally large numbers of articles, the training dataset uses a monthly sampling strategy.

- Completed months are capped at 500 articles.
- Months containing fewer than 500 articles are retained in full.
- Months exceeding the cap are sampled using daily-stratified allocation.
- The current calendar month remains uncapped.
- Sampling is performed without replacement.
- A deterministic date-based seed is used for reproducibility.

A sampling audit records the number of articles before and after sampling, represented days, and daily allocation statistics.

### Training Dataset

The sampled articles are aggregated by feature date and merged with the corresponding market observations.

The resulting dataset contains:

- Market variables
- Daily news volume
- Unique source counts
- Daily sentiment aggregates
- Seven-calendar-day news features
- Next-session return targets

The final historical model dataset is:

`data/processed/model_dataset.csv`

### Historical and Live Feature Construction

Historical training uses the sampled news archive.

Live forecasting uses the complete scored archive to construct the latest seven-calendar-day news features.

This distinction is intentional.

---

## 6. Feature Engineering

The model uses both market-derived and news-derived predictors.

### Market Features

| Feature | Description |
|---|---|
| Daily return | Current session's percentage return |
| Lagged returns | Returns from previous trading sessions |
| 5-session volatility | Rolling standard deviation of returns |
| 20-session volatility | Longer-term rolling standard deviation |
| Momentum | Cumulative returns over selected lookback windows |

### News Features

| Feature | Description |
|---|---|
| Article count | Number of news articles |
| Unique sources | Number of distinct publishers |
| Mean sentiment | Average FinBERT sentiment score |
| Median sentiment | Median FinBERT sentiment score |
| Positive fraction | Proportion of positive headlines |
| Negative fraction | Proportion of negative headlines |
| Neutral fraction | Proportion of neutral headlines |

### Seven-Day Rolling Features

The pipeline also calculates news and sentiment features over a trailing seven-calendar-day window.

These include:

- `news_7d_article_count`
- `news_7d_unique_sources`
- `news_7d_mean_sentiment`
- `news_7d_median_sentiment`
- `news_7d_positive_fraction`
- `news_7d_negative_fraction`
- `news_7d_neutral_fraction`

### Logarithmic Transformation

Count-valued predictors are transformed using:

\[
x' = \log(1+x)
\]

The transformation is applied to article counts and unique-source counts in both historical and live inputs.

This compresses large count values while preserving their ordering.

Sentiment scores and sentiment fractions are not transformed.

### Missing Values and Scaling

The machine learning pipelines use:

1. Median imputation using `SimpleImputer`.
2. Feature standardization using `StandardScaler`.
3. The selected regression or classification estimator.

Preprocessing is fitted within the training pipeline to avoid fitting the scaler or imputer using future test observations.

---

## 7. Machine Learning Models

The project implements three forecasting tasks.

### A. Return Prediction — Ridge Regression

Ridge Regression estimates the next-session percentage return.

The model minimizes:

\[
\sum_{i=1}^{n}(y_i-X_i\beta)^2
+
\alpha\sum_{j=1}^{p}\beta_j^2
\]

The regularization parameter is:

\[
\alpha = 10
\]

Two feature configurations are evaluated:

- Market-only model
- Market + sentiment model

Ridge regularization penalizes large coefficients and helps manage correlated predictors.

### B. Direction Prediction — Logistic Regression

The direction classifier estimates the probability of a positive next-session return.

The binary target is:

\[
Y =
\begin{cases}
1 & \text{if next-session return} > 0\\
0 & \text{otherwise}
\end{cases}
\]

Logistic Regression models:

\[
P(Y=1|X)=
\frac{1}{1+e^{-(\beta_0+X\beta)}}
\]

The output is interpreted as an estimated probability of an UP session.

### C. Volatility Prediction — Logistic Regression

The volatility model estimates the probability of an unusually large next-session absolute return.

The threshold is calculated using:

- 252-session historical lookback
- Minimum 60 observations
- 80th percentile of historical absolute daily returns

The target is:

\[
V_t =
\mathbb{1}(|r_{t+1}| > Q_{0.80,t})
\]

The classifier estimates the probability of this event.

The forecast also reports operational alerts at 20% and 30% probability thresholds.

---

## 8. Walk-Forward Validation and Adaptive Model Selection

Because the project deals with time-series data, model evaluation is chronological rather than randomly shuffled.

The pipeline implements two training-window strategies.

### Expanding Window

The model is trained on all eligible historical observations before each prediction.

The training dataset grows as new observations become available.

### Rolling Window

The model uses a fixed historical window of up to 252 trading sessions.

Older observations are discarded as the window moves forward.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

Measures average absolute return-prediction error.

**Root Mean Squared Error (RMSE)**

\[
RMSE =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

Penalizes larger prediction errors more heavily.

**Directional Accuracy**

Measures the proportion of correctly predicted market directions.

**Brier Score**

\[
BS = \frac{1}{n}\sum_{i=1}^{n}(p_i-o_i)^2
\]

Evaluates probabilistic direction predictions. Lower values are better.

### Adaptive Selection

The adaptive selector compares the baseline, market-only, and sentiment-based configurations.

For return prediction:

- The expanding mean-return baseline is the reference.
- A feature model must improve MAE by at least 2% to replace it.

For direction prediction:

- Brier score is the primary selection metric.
- Accuracy is used as a tie-breaker.

The selector uses recent out-of-sample performance over approximately two calendar months, requiring at least 30 predictions.

If feature models fail the return improvement criterion, the baseline is retained.

---

## 9. Forecast Generation

The forecasting script uses the latest available NIFTY 50 session and the corresponding historical training dataset.

The process:

1. Loads historical training observations.
2. Constructs market lag and rolling features.
3. Loads the full scored news archive.
4. Builds the latest seven-calendar-day news features.
5. Applies the same count transformations used during training.
6. Runs adaptive model selection.
7. Fits the selected historical configuration.
8. Generates return, direction, and volatility estimates.
9. Saves the forecast to CSV and SQLite.

### Forecast Outputs

The principal forecast file is:

`data/processed/latest_forecast.csv`

It contains the outputs of the individual models and the adaptively selected configurations.

| Output | Interpretation |
|---|---|
| Market-only predicted return | Return estimated using market-related features |
| Market-only P(UP) | Estimated probability of positive return using market features |
| Market + sentiment predicted return | Return estimate including sentiment features |
| Market + sentiment P(UP) | Estimated UP probability including sentiment |
| Selected predicted return | Output from the adaptive return strategy |
| Selected direction probability | Output from the adaptive direction strategy |
| Volatility probability | Estimated probability of exceeding the historical volatility threshold |
| Volatility threshold | Current historical absolute-return threshold |

The individual market and sentiment model outputs must not be confused with the adaptively selected forecast.

---

## 10. Forecast Outcome Tracking

Forecasts are stored in the SQLite database.

`src/record_forecast_outcome.py` checks whether a later NIFTY 50 trading-session price is available for each pending forecast.

Once the target session becomes available, the system records:

- Actual next-session return
- Actual market direction
- Actual absolute return
- Volatility event outcome
- Evaluation status

Forecasts from different model versions are kept separate during official performance evaluation.

This enables prospective evaluation of the model using predictions that were genuinely generated before their outcomes became known.

---

## Project Structure

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

*The structure above describes the main project organization; individual generated files may vary depending on the pipeline run.*

---

## Installation and Usage

### Requirements

- Python 3.10 or later
- Conda or another Python environment manager
- Internet connection for fresh data collection
- Sufficient memory for FinBERT inference
- CUDA-compatible GPU (optional)

### 1. Clone the repository

```bash
git clone git@github.com:SVDKWTD/finance-IDC.git
cd finance-IDC
```

### 2. Create and activate the environment

```bash
conda create -n finance-IDC python=3.11
conda activate finance-IDC
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch installation may need to be configured separately depending on whether CPU or CUDA execution is required. Follow the official PyTorch installation instructions for your hardware.

### 4. Run the complete pipeline

```bash
python run_pipeline.py
```

### 5. Run using existing collected data

```bash
python run_pipeline.py --skip-refresh
```

This skips the collection refresh and uses the existing local data.

### 6. Run selected stages

```bash
python run_pipeline.py --only align,sentiment,dataset,forecast,outcomes
```

### 7. Explore the project notebook

Open `Finance_IDC_Project_Walkthrough.ipynb` in Jupyter Notebook or JupyterLab to inspect the datasets, understand preprocessing, visualize sentiment distributions, and examine the forecasting methodology.

---

## Key Libraries

| Library | Application |
|---|---|
| Pandas | Data manipulation and processing |
| NumPy | Numerical computations |
| yfinance | NIFTY 50 price collection |
| Requests / Feedparser | Financial-news RSS collection |
| SQLite3 | Persistent data storage |
| Transformers | FinBERT sentiment analysis |
| PyTorch | Deep-learning inference backend |
| Scikit-learn | Regression, classification, preprocessing, and evaluation |
| Matplotlib | Data visualization |
| Jupyter | Interactive analysis and documentation |

---

## Limitations and Future Improvements

This project is an experimental forecasting system, not a production-grade financial trading system.

Important limitations include:

1. **News coverage:** Google News RSS search results do not represent every financial article published.
2. **Headline-only sentiment:** FinBERT analyses headlines rather than complete article content.
3. **Sampling bias:** Monthly sampling may change the distribution of topics, publishers, and event types represented in historical training data.
4. **Historical/live feature differences:** Historical training uses sampled news, whereas live forecasts use the complete archive.
5. **Probability calibration:** Logistic Regression outputs have not yet been demonstrated to be well calibrated.
6. **Limited prospective evaluation:** A successful pipeline execution does not establish predictive performance.
7. **Market data availability:** Forecast quality depends on timely and accurate market-price updates.
8. **Trading costs:** Transaction costs, slippage, position sizing, and portfolio risk have not been fully modelled.

Future work should prioritize:

- Collecting and evaluating more prospective forecasts.
- Testing probability calibration.
- Improving news relevance and source coverage.
- Investigating the effects of historical sampling on live predictions.
- Evaluating model stability across different market regimes.
- Incorporating realistic transaction costs before considering any trading application.

---

## Current Project Status

The pipeline successfully executes the following stages:

- Incremental news and market-data processing.
- News-market alignment.
- FinBERT sentiment scoring.
- Historical dataset construction.
- Feature engineering.
- Walk-forward adaptive model selection.
- Next-session return, direction, and volatility forecasting.
- Forecast outcome tracking.

However, successful execution should not be confused with demonstrated predictive skill. The current evaluation has not established that sentiment-based models consistently outperform the historical mean-return baseline.

The primary objective is to build a reproducible framework for studying the relationship between financial news sentiment and short-term market dynamics.

---

## Disclaimer

This project is intended for academic and research purposes only.

Forecasts are model-generated estimates and should not be interpreted as financial advice, guaranteed market predictions, or recommendations to buy or sell securities.