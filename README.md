# NIFTY 50 Forecasting Using News Sentiment and Market Data

A Python-based research project investigating whether financial news sentiment, news volume, and historical market behaviour can help predict the next-session return and direction of the NIFTY 50 index.

The project implements an end-to-end pipeline for collecting financial news and market prices, processing news sentiment, aligning information with trading sessions, training forecasting models, evaluating their performance, and tracking predictions against subsequently observed market outcomes.

> **Current status:** The data collection and forecasting pipeline is implemented. Initial walk-forward evaluation has not shown that the news-based models outperform a simple mean-return baseline. The project is therefore a forecasting research prototype, not a validated trading system.

## Table of Contents

- [1. Project Objective](#1-project-objective)
- [2. Research Questions](#2-research-questions)
- [3. Tools and Technologies](#3-tools-and-technologies)
- [4. How the System Works](#4-how-the-system-works)
- [5. Pipeline Architecture](#5-pipeline-architecture)
- [6. News and Market Data Processing](#6-news-and-market-data-processing)
- [7. Sentiment Analysis](#7-sentiment-analysis)
- [8. Forecasting Models](#8-forecasting-models)
- [9. Evaluation and Current Results](#9-evaluation-and-current-results)
- [10. Running the Project](#10-running-the-project)
- [11. Understanding a Forecast](#11-understanding-a-forecast)
- [12. Data and Database Structure](#12-data-and-database-structure)
- [13. Limitations and Future Work](#13-limitations-and-future-work)
- [14. Disclaimer](#14-disclaimer)

## 1. Project Objective

Financial markets respond to many interacting factors, including historical price movements, volatility, macroeconomic conditions, investor expectations, and information released through financial news.

This project investigates whether publicly available financial news contains information that can improve forecasts of the next trading session's NIFTY 50 return.

The central hypothesis is:

**News sentiment and news activity may contain additional information about future market movements beyond what is available from historical market data alone.**

To investigate this, the project combines two types of information:

1. **Market data:** Historical OHLC prices, volume, returns, and volatility-related features.
2. **News data:** Financial news headlines, publication timestamps, source information, and sentiment scores generated using FinBERT.

The resulting dataset is used to compare market-only models against models that also incorporate news-derived features.

The project also includes a forecast history and outcome-tracking mechanism so that predictions can be evaluated after the corresponding market session has completed.

## 2. Research Questions

The project is designed to investigate the following questions:

- Can historical NIFTY 50 returns and volatility help predict the next-session return?
- Does financial news sentiment have a measurable relationship with subsequent returns?
- Does incorporating sentiment improve forecasting performance relative to market-only models?
- Does news volume provide useful information beyond sentiment?
- Are predicted market directions more accurate than a simple majority-direction baseline?
- Do forecast errors vary with market volatility?
- Can forecasts be recorded, evaluated against actual outcomes, and analysed over time?

These are empirical questions. A relationship between sentiment and returns does not automatically imply useful predictive power, and predictive performance must be established through out-of-sample testing.

## 3. Tools and Technologies

| Tool or library | Role in the project |
|---|---|
| Python | Main programming language and pipeline orchestration |
| pandas and NumPy | Tabular data processing, feature engineering, numerical calculations, and dataset construction |
| Google News RSS | Collection of historical and recent financial news headlines |
| yfinance | Retrieval of NIFTY 50 historical market data |
| FinBERT (`ProsusAI/finbert`) | Financial-domain sentiment classification |
| Hugging Face Transformers | Loading and running the pretrained sentiment model |
| scikit-learn | Ridge regression, logistic regression, and forecasting evaluation |
| SQLite | Persistent storage of collected data and forecast records |
| Matplotlib | Visualisation and exploratory data analysis |
| Jupyter Notebook | Interactive exploration and analysis of the datasets |
| Git and GitHub | Version control and source-code hosting |
| Conda | Python environment management |

The exact dependencies and versions should be recorded in a `requirements.txt` or `environment.yml` file to make the project reproducible on another machine.

## 4. How the System Works

The project follows a sequence of data-processing and modelling stages.

1. Collect historical and recent news headlines.
2. Collect historical NIFTY 50 prices.
3. Clean and normalise the collected data.
4. Align news publication times with market sessions.
5. Run financial sentiment analysis on news headlines.
6. Aggregate article-level sentiment into daily features.
7. Construct a supervised learning dataset.
8. Train models to estimate next-session returns and market direction.
9. Generate and store a forecast.
10. Retrieve subsequent completed market prices and evaluate the forecast.

The distinction between data collection, model training, prediction, and evaluation is important. A forecast is generated using information available at its origin date; its actual outcome is only known after the target trading session has finished.

## 5. Pipeline Architecture

The primary entry point is `run_pipeline.py`.

The pipeline executes the following stages in sequence:

```text
                  NEWS DATA
                      |
          Historical and recent RSS
                      |
                Cleaning and
             timestamp processing
                      |
                      v
                NEWS DATASET
                      |
                      +----------------------+
                                             |
                                             v
                                  News-market alignment
                                             |
                                             v
                                      FinBERT sentiment
                                             |
                                             v
                                  Daily sentiment features
                                             |
                                             |
MARKET DATA                               |
    |                                        |
    v                                        |
NIFTY 50 price collection                   |
    |                                        |
    v                                        |
Returns, price features                      |
    |                                        |
    +-------------------+--------------------+
                        |
                        v
               Model dataset creation
                        |
                        v
              Return and direction models
                        |
                        v
                 Forecast generation
                        |
                        v
                SQLite forecast history
                        |
                        v
              Outcome tracking and scoring
```

The pipeline is automated, but each stage can also be run independently for debugging and research.

### Pipeline execution order

| Stage | Script | Purpose |
|---|---|---|
| 1 | `src/collect_prices.py` | Collect and update NIFTY 50 price data |
| 2 | `src/collect_recent_news.py` | Collect recent news incrementally |
| 3 | `src/align_news_market.py` | Align news with market sessions |
| 4 | `src/analyze_news_sentiment.py` | Score headlines and aggregate sentiment |
| 5 | `src/build_model_dataset.py` | Construct the supervised modelling dataset |
| 6 | `src/generate_forecast.py` | Generate and save the latest forecast |
| 7 | `src/record_forecast_outcome.py` | Evaluate forecasts whose target sessions have completed |

The pipeline stops if a stage fails, preventing subsequent stages from silently running on incomplete intermediate outputs.

The repository also contains supporting scripts for historical news collection, data cleaning, timestamp normalisation, database auditing, exploratory analysis, dataset validation, and model evaluation.

## 6. News and Market Data Processing

### 6.1 News collection

News is collected using Google News RSS searches for selected financial and market-related terms.

The project separates historical collection from recent incremental collection.

- Historical collection builds a multi-year archive of headlines.
- Recent collection searches a rolling time window and merges new records with previously saved data.
- URL-based deduplication reduces repeated articles across overlapping searches.
- Publication timestamps and source metadata are retained where available.

The rolling-window approach allows the project to update the dataset without downloading the entire recent news archive every time.

However, RSS collection is not guaranteed to retrieve every article published by every source. The resulting dataset should be treated as a collected sample of news rather than a complete representation of all financial information.

### 6.2 Market data

NIFTY 50 historical price data is retrieved using `yfinance`.

The dataset includes fields such as:

- Open
- High
- Low
- Close
- Volume
- Adjusted close, where provided

Incremental updates merge recently downloaded observations with existing data.

A distinction is made between raw data and completed market sessions. The pipeline excludes the current incomplete session from the processed price dataset so that an unfinished daily candle is not inadvertently treated as a completed trading day.

### 6.3 Returns and the prediction target

The daily return is calculated from consecutive closing prices:

\[
r_t = \frac{C_t-C_{t-1}}{C_{t-1}}
\]

The percentage return is:

\[
r_t^{(\%)} = 100r_t
\]

The primary target is the next trading session's return:

\[
y_t = 100\left(\frac{C_{t+1}}{C_t}-1\right)
\]

Here, \(C_t\) is the closing price on session \(t\), and \(C_{t+1}\) is the closing price on the next available trading session.

The model therefore estimates the next-session close-to-close return, not the intraday return, the next opening price, or the probability of a profitable trade after transaction costs.

The final observation in a dataset generally has no target because its next-session closing price is not yet available.

### 6.4 News-market alignment

News publication timestamps are converted to Indian Standard Time (IST), and headlines are assigned to a market-session date.

The current alignment implementation uses a 15:30 IST market-close reference to assign a headline to the first eligible closing time at or after publication.

This is a practical approximation and requires further auditing before the resulting forecasts can be considered fully leakage-free. In particular, the information available at the precise forecast-generation time must be distinguished from information that became available later.

## 7. Sentiment Analysis

The project uses FinBERT, a pretrained transformer model designed for financial text, to classify news headlines into three sentiment categories:

- **Positive:** The headline is classified as positive.
- **Negative:** The headline is classified as negative.
- **Neutral:** The headline is classified as neutral.

Sentiment is first calculated at the article level. Article-level results are then aggregated into daily features.

The current feature set includes:

| Feature | Interpretation |
|---|---|
| `article_count` | Number of aligned news articles |
| `unique_sources` | Number of distinct news sources represented |
| `mean_sentiment` | Mean article-level sentiment score |
| `median_sentiment` | Median article-level sentiment score |
| `positive_fraction` | Fraction of classified articles labelled positive |
| `negative_fraction` | Fraction of classified articles labelled negative |
| `neutral_fraction` | Fraction of classified articles labelled neutral |

Sentiment scores are cached by article URL, with a fallback key based on headline and publication time. This prevents previously processed headlines from being scored repeatedly during incremental runs.

Daily aggregation reduces thousands of article-level observations to a smaller set of features that can be combined with market data.

**Important:** FinBERT sentiment is a classification of text, not a direct measurement of investor sentiment, market impact, or the probability that prices will rise.

## 8. Forecasting Models

The current implementation uses two related prediction tasks.

### 8.1 Return regression

The regression task predicts a numerical next-session percentage return.

Two model configurations are compared:

**Market-only model**

Uses historical market features such as:

- Recent return lags
- Short-term volatility
- Daily market returns
- News-independent price information

**Market-plus-news model**

Adds sentiment and news-activity features, including lagged sentiment and daily sentiment aggregates.

The implementation uses Ridge regression, a linear regression method with L2 regularisation. Regularisation discourages excessively large coefficients and can help control overfitting when features are correlated.

The regularisation parameter is currently set to `alpha=10`.

### 8.2 Direction classification

A separate logistic regression model estimates the probability of an UP or DOWN next-session move.

The output is a probability distribution over the two direction classes. A probability above 50% corresponds to an UP prediction under the current decision rule.

Direction classification and return regression solve different problems:

- Regression estimates the expected numerical return.
- Classification estimates the probability of the direction label.

They are trained separately, so their outputs can disagree. For example, a model can predict a small positive expected return while assigning less than 50% probability to an UP session.

Such disagreement should be reported as uncertainty rather than resolved by arbitrarily trusting one model.

### 8.3 Feature lags

The walk-forward evaluation uses features such as:

- `return_lag1`
- `return_lag2`
- `return_lag5`
- `volatility_5`
- `sentiment_lag1`
- `daily_return_pct`
- `article_count`
- `unique_sources`
- Sentiment averages and class fractions

Lagged features are intended to ensure that a forecast uses historical observations rather than the target session's realised values.

However, lagging alone does not guarantee the absence of leakage. News timestamps, aggregation boundaries, preprocessing, and the timing of each forecast must also be audited.

## 9. Evaluation and Current Results

Forecasts are evaluated using out-of-sample testing. The walk-forward evaluation repeatedly trains models on past observations and evaluates them on subsequent observations, which better reflects the intended forecasting use than evaluating only on training data.

### 9.1 Return prediction

The current walk-forward evaluation produced the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean-return baseline | 0.6146% | 0.8322% | -0.0034 |
| Market model | 0.6276% | 0.8447% | -0.0337 |
| Market-plus-sentiment model | 0.6299% | 0.8454% | -0.0356 |

Lower MAE and RMSE indicate smaller prediction errors. A higher R² indicates a better fit relative to a constant-mean reference under the evaluation definition.

The mean-return baseline currently outperforms both forecasting models on MAE and RMSE.

Negative R² values indicate that the model's predictions perform worse than the corresponding mean-prediction reference under this metric.

### 9.2 Direction prediction

| Model | Directional accuracy | Brier score |
|---|---:|---:|
| Market model | 47.10% | 0.2576 |
| Market-plus-sentiment model | 47.56% | 0.2580 |

Directional accuracy measures the fraction of correctly classified market directions.

The Brier score evaluates probabilistic predictions; lower values are better. The sentiment model has slightly higher directional accuracy here, but its Brier score is marginally worse.

These results do not establish a reliable improvement from sentiment features.

### 9.3 Relationship between sentiment and returns

The initial sentiment-relationship analysis found:

- Pearson correlation: approximately 0.0303, with \(p \approx 0.2532\).
- Spearman correlation: approximately 0.0541, with \(p \approx 0.0410\).
- Welch test comparing subsequent returns on positive- and negative-sentiment days: \(p \approx 0.2949\).

The observed correlations are small. The Spearman result provides limited evidence of a monotonic association in this sample, but it does not demonstrate economically meaningful predictive power. Multiple testing, temporal dependence, and the overall experimental design also need consideration.

### 9.4 Interpretation

The current results suggest that sentiment, as presently measured and incorporated, does not improve return prediction.

Possible explanations include:

- News sentiment is noisy or weakly related to next-session index returns.
- Market prices already incorporate much of the publicly available information.
- Daily aggregation removes important intraday timing and event information.
- The selected features and linear models do not capture relevant nonlinear relationships.
- The dataset and news collection procedure introduce sampling or alignment biases.

These are hypotheses to investigate, not established explanations.

## 10. Running the Project

### 10.1 Requirements

The project uses Python and a Conda environment. It also requires access to the internet for collecting data and downloading model files when they are not already cached.

The dependencies used by the current code include the relevant packages for data processing, market data retrieval, RSS collection, FinBERT inference, modelling, and visualisation.

Before running the project on a new machine, create a reproducible dependency file such as `environment.yml` or `requirements.txt` containing the actual installed package versions.

### 10.2 Activate the environment

Open Anaconda Prompt and navigate to the repository:

```cmd
cd C:\Users\surya\Documents\finance-IDC-Project
conda activate finance-IDC
```

Replace the path if the repository is stored elsewhere.

### 10.3 Run the complete pipeline

Execute:

```cmd
python run_pipeline.py
```

The script runs the collection, alignment, sentiment, dataset construction, forecasting, and outcome-tracking stages in sequence.

On the first run, sentiment inference may take substantial time because each previously unseen headline needs to be processed. Later runs should be faster because previously scored headlines are cached.

The pipeline requires internet access for data collection and may also need it to download the pretrained FinBERT model.

### 10.4 Run individual stages

To execute a particular stage independently, run the corresponding script from the project root. For example:

```cmd
python src/collect_prices.py
python src/collect_recent_news.py
python src/align_news_market.py
python src/analyze_news_sentiment.py
python src/build_model_dataset.py
python src/generate_forecast.py
python src/record_forecast_outcome.py
```

These commands are intended for targeted reruns and debugging. For routine updates, use `python run_pipeline.py`.

### 10.5 Inspect the results

The main files to inspect are:

- `data/processed/model_dataset.csv` — final modelling dataset.
- `data/processed/news_sentiment_daily.csv` — daily sentiment aggregates.
- `data/processed/walk_forward_predictions.csv` — historical walk-forward predictions.
- `data/processed/walk_forward_return_comparison.csv` — walk-forward regression metrics.
- `data/processed/walk_forward_direction_comparison.csv` — walk-forward direction metrics.
- `data/processed/forecast_model_comparison.csv` — results from the initial forecast evaluation.
- `data/processed/sentiment_relationship_summary.csv` — summary of the initial sentiment-return analysis.
- `data/market.db` — local SQLite database containing stored data and forecast history.

Some evaluation files are generated by separate analysis scripts rather than by the main daily pipeline. Run the relevant evaluation script when you need to regenerate them.

## 11. Understanding a Forecast

The `src/generate_forecast.py` script trains the configured models on the available labelled data and generates a forecast for the next session.

The current implementation stores forecasts in the SQLite `forecast_history` table. Each record includes fields such as:

- Forecast origin date
- Target session date, where known
- Forecast creation timestamp
- Number of training observations
- Market-only return estimate
- Market-plus-sentiment return estimate
- Direction probabilities
- Model version
- Warnings and evaluation fields

The target date can initially be unknown because the next exchange trading session is not necessarily the next calendar day. The outcome tracker subsequently resolves pending forecasts against available completed price data.

### Forecast lifecycle

1. A forecast is generated after the available market and news data have been processed.
2. The forecast is saved with its origin date and model metadata.
3. The target session is resolved when the next completed price observation becomes available.
4. The actual return and direction are recorded.
5. The forecast can then be included in performance summaries.

The forecast-history mechanism is designed to make predictions auditable instead of overwriting the latest prediction each time the script runs.

The current forecast-generation script also attaches warnings when the model has weak historical performance, probabilities are close to 50%, or probabilities have not been calibrated.

**A generated prediction is not evidence of a profitable strategy.** It is an estimate that must be assessed against subsequent observations and appropriate baselines.

## 12. Data and Database Structure

The `data/` directory stores local datasets and generated outputs. It is excluded from Git tracking to avoid uploading large datasets and the local database.

A simplified structure is:

```text
finance-IDC-Project/
│
├── run_pipeline.py
├── README.md
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── nifty50_news_eda.ipynb
│
├── src/
│   ├── collect_news.py
│   ├── collect_recent_news.py
│   ├── collect_prices.py
│   ├── audit_historical_news.py
│   ├── normalize_news_timestamps.py
│   ├── align_news_market.py
│   ├── analyze_news_sentiment.py
│   ├── analyze_sentiment_relationship.py
│   ├── build_model_dataset.py
│   ├── evaluate_forecasts.py
│   ├── evaluate_walk_forward.py
│   ├── generate_forecast.py
│   ├── record_forecast_outcome.py
│   └── ...
│
└── data/                     # Local, generated data; ignored by Git
    ├── market.db
    ├── raw/
    │   ├── historical_news_2021_2025.csv
    │   ├── recent_news_2026.csv
    │   └── nifty50_prices.csv
    │
    └── processed/
        ├── news_market_alignment.csv
        ├── news_sentiment_scored.csv
        ├── news_sentiment_daily.csv
        ├── model_dataset.csv
        └── ...
```

The exact set of files depends on which collection, validation, and evaluation scripts have been run.

### Database

SQLite provides persistent storage without requiring a separate database server.

The database is used to store collected market/news records and forecast history. Forecast records support subsequent outcome evaluation and prevent the latest prediction from being the only retained record.

The database and generated CSVs are local project artifacts. They are not included in the GitHub repository by default.

## 13. Limitations and Future Work

The project currently provides an end-to-end forecasting workflow, but several issues need to be addressed before making stronger claims about predictive performance.

### 13.1 Audit information timing

The news-alignment procedure must be audited carefully. A headline should only influence a forecast if it was available by the actual forecast-generation time.

The final design should explicitly define whether the forecast is generated before market open, at market close, or at another fixed time. News published after that cutoff must not enter the corresponding feature set.

### 13.2 Establish a stronger experimental design

The existing walk-forward results indicate that the news-based models do not beat the simple mean-return baseline.

Future experiments should preserve chronological ordering, avoid training on future information, and compare models on identical evaluation dates.

Any feature selection, hyperparameter tuning, or probability calibration should be conducted using training data only, with a separate period reserved for final evaluation.

### 13.3 Improve the features

Potential extensions include:

- Better handling of market holidays and session boundaries.
- Explicit pre-market, intraday, and post-market news features.
- Deduplication of near-identical headlines from multiple sources.
- Entity-level features for companies, sectors, macroeconomic indicators, and policy announcements.
- Measures of sentiment strength, disagreement, novelty, and news volume relative to historical activity.
- Additional lagged returns, volatility measures, and market-regime features.

These are proposed experiments, not features already validated by the current model.

### 13.4 Compare additional models

After establishing a leakage-safe baseline, possible comparisons include:

- Regularised linear models.
- Tree-based models such as Random Forest and gradient boosting.
- Models with explicit nonlinear interactions between news and market features.
- Alternative sentiment aggregation strategies.

A more complex model should only be retained if it improves performance on genuinely out-of-sample data.

### 13.5 Calibrate probabilities

The direction model's output should not be treated as a trustworthy probability until calibration has been tested.

Future work should include reliability diagrams, Brier scores, and calibration methods fitted using training-period or validation-period predictions only.

### 13.6 Add transaction costs and trading evaluation

Return prediction and trading profitability are different objectives.

A future trading simulation should specify a trading rule, execution assumptions, brokerage and transaction costs, slippage, turnover, and appropriate risk metrics. It should also avoid selecting the strategy using the final test period.

### 13.7 Improve reproducibility

Recommended additions include:

- A verified `requirements.txt` or `environment.yml`.
- A documented configuration for collection and forecast timing.
- A fixed random seed wherever applicable.
- Unit tests for return calculation, news alignment, and target construction.
- Automated tests for duplicate records and future-data leakage.
- A small example dataset for demonstrating the pipeline without downloading the full archive.
- A documented procedure for rebuilding all generated datasets from source data.

### 13.8 Define a forecast-generation schedule

The current pipeline is manually executable. A later version could run automatically after each completed market session, record the forecast, and evaluate it when the next target session becomes available.

Automation should use a consistent cutoff time and account for Indian market holidays and incomplete data updates.

## 14. Disclaimer

This project is intended for educational and research purposes. It is not investment advice and does not provide a validated trading strategy.

Historical relationships may not persist, and model forecasts may be wrong. The current evaluation does not demonstrate that incorporating news sentiment improves predictive performance over a simple baseline.

No investment decisions should be based solely on these forecasts.

---

**Project focus:** Financial news processing, sentiment analysis, time-series feature engineering, supervised learning, walk-forward evaluation, and reproducible forecasting workflows.



## 13. Integrated volatility forecasting

The routine pipeline now produces three complementary outputs from the latest completed market session:

1. **Market-only return forecast** — Ridge regression estimate of next-session percentage return.
2. **Market + news/sentiment forecast** — Ridge regression estimate and a separate Logistic Regression probability for an UP move.
3. **Next-session volatility probability** — Logistic Regression probability that the absolute next-session close-to-close return exceeds the trailing 80th percentile of absolute daily returns, using a 252-session window with at least 60 observations.

The volatility model uses an expanding historical training set and a fixed specification. Its probability is stored with a historical threshold and exploratory alert indicators at 20% and 30%. These thresholds are not trading recommendations and should not be re-tuned against the prospective evaluation period.

### Outcome tracking

`src/record_forecast_outcome.py` resolves the next available market session from the completed price data and records the actual return. For forecasts with a stored volatility threshold, it also records absolute return and whether the session met the pre-defined volatility criterion. The original return and direction outcomes remain intact.

### Current evidence and limitations

Retrospective evaluation of 413 previously inspected forecasts showed volatility ROC-AUC around 0.651 and a lower Brier score than a historical-prevalence baseline. Performance varied over chronological thirds, and the latest third was close to chance-level ranking. These observations are promising but do not establish prospective reliability.

The first prospective forecast (feature date 2026-10-27) assigned 13.7% probability to volatility with a historical absolute-return threshold of 0.9777%. Its outcome must be evaluated against the next actual completed trading session. One prospective forecast is not sufficient to assess performance.

Same-date news features are only valid if those features were available by the forecast cutoff. The current alignment approximation must be audited before describing the entire workflow as fully leakage-free. Direction probabilities have not been demonstrated to be calibrated, and the return models have not outperformed the mean-return baseline in the recorded walk-forward evaluation.

### Run the routine pipeline

From the repository root in the configured Conda environment:

```cmd
python run_pipeline.py
```

The pipeline calls `src/generate_forecast.py` and then `src/record_forecast_outcome.py`. The SQLite schema is extended additively for volatility fields, preserving prior records. New forecast rows use a new model version identifier so previous forecast history is not overwritten.
