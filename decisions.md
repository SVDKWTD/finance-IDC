# decisions.md — Project Decisions and Rationale

This file records important project choices, alternatives, and reasons. It also preserves changes from the initial plan so that the project can be explained accurately in a report or presentation.

## Decision 1: Use scripts for repeatable pipeline stages and notebooks for exploration

- **Decision:** Use `.py` scripts under `src/` for collection, processing, modelling, and forecast tracking; use Jupyter notebooks for EDA and analysis.
- **Alternative:** Put all code in one notebook or one large script.
- **Reason:** Scripts make repeated pipeline runs easier to automate, while notebooks support charts, inspection, and explanation.

## Decision 2: Focus on the NIFTY 50 index

- **Initial plan:** Consider NIFTY 50, Reliance, TCS, gold, INR/USD, and oil as possible variables.
- **Current implementation:** The primary target is the NIFTY 50 index. The current model dataset uses NIFTY 50 price features and news-derived features.
- **Reason:** Keeping the target focused makes the data pipeline and forecasting question easier to define and evaluate.
- **Status:** Reliance, TCS, gold, INR/USD, and oil are not part of the current documented modelling pipeline unless later added explicitly.

## Decision 3: Predict both numerical return and direction

- **Decision:** Build a regression model for next-session percentage return and a classification model for UP/DOWN direction.
- **Models:** Ridge regression for return estimates; Logistic Regression for direction probabilities.
- **Alternatives:** Predict only the closing price, or use a more complex model immediately.
- **Reason:** Return is a more directly interpretable target than raw price level, while direction is a separate useful research question. Simple models provide a baseline before trying more complex approaches.
- **Important distinction:** Regression and classification can disagree because they estimate different quantities.

## Decision 4: Use time-ordered evaluation, including walk-forward testing

- **Decision:** Preserve chronology and evaluate forecasts on later observations. Use walk-forward evaluation in addition to a simple held-out split.
- **Alternative:** Randomly split daily observations.
- **Reason:** Random splitting can leak future patterns into training and does not simulate forecasting future sessions.
- **Metrics:** Regression uses MAE, RMSE, and R². Direction evaluation uses accuracy and Brier score; further work should include F1, confusion matrix, and calibration diagnostics where appropriate.
- **Baseline:** Compare model forecasts with a simple mean-return baseline and simple direction baselines.
- **Current outcome:** The news-based return models have not outperformed the mean-return baseline in the latest walk-forward results.

## Decision 5: Use incremental data collection and caching

- **Decision:** Use rolling-window incremental news collection with URL deduplication, incremental price updates with overlap, and cached sentiment scores.
- **Alternative:** Download and rescore the entire dataset on every run.
- **Reason:** Incremental collection reduces repeated work, helps accommodate late-arriving RSS items, and makes routine pipeline runs more efficient.
- **Trade-off:** RSS coverage is incomplete and overlapping windows do not guarantee that every article is collected. Coverage reports and deduplication checks remain important.

## Decision 6: Use free data sources and avoid paid dependencies

- **Decision:** Keep the project runnable using free data sources and open-source libraries.
- **Market data:** `yfinance` is used to collect NIFTY 50 prices.
- **News data:** Google News RSS is used for historical and recent headline collection.
- **Alternatives considered in the initial plan:** NewsAPI, GDELT, and Alpha Vantage.
- **Reason:** Paid plans are outside the project's constraints. Google News RSS was used in the implemented collection scripts; NewsAPI and Alpha Vantage are not required by the current pipeline.
- **Trade-off:** Free RSS collection can be incomplete, rate-limited, and less reproducible than a licensed archival feed.
- **Historical correction:** The initial plan mentioned VADER for sentiment. The implemented pipeline uses FinBERT (`ProsusAI/finbert`) instead.

## Decision 7: Use FinBERT for financial headline sentiment

- **Decision:** Use the pretrained `ProsusAI/finbert` model through Hugging Face Transformers to classify headlines as positive, negative, or neutral.
- **Alternative:** Use a general-purpose lexicon such as VADER or manually assign sentiment.
- **Reason:** FinBERT is designed for financial text and supplies model-based class scores.
- **Efficiency choice:** Cache previously calculated article scores using URL, with a fallback key based on headline and publication timestamp.
- **Limitation:** A headline's sentiment label is not a direct measurement of market impact or investor behaviour.

## Decision 8: Aggregate article sentiment into daily features

- **Decision:** Combine article-level sentiment with market observations using daily aggregates, including article count, unique source count, mean/median sentiment, and positive/negative/neutral fractions.
- **Alternative:** Feed every article independently into a sequence model.
- **Reason:** Daily aggregation creates a manageable feature table aligned with the daily market target.
- **Limitation:** Aggregation may lose event timing, article importance, and disagreement among sources.

## Decision 9: Define the target as next-session close-to-close return

- **Decision:** Predict the percentage change from the current completed close to the next available trading-session close.
- **Target:** `target_next_session_return_pct`.
- **Reason:** This is a clearly defined supervised learning target and naturally skips weekends and exchange holidays by using the next available trading observation.
- **Limitation:** It does not directly model intraday returns, next-session opening gaps, transaction costs, or executable trading returns.

## Decision 10: Use a local SQLite database for persistence

- **Decision:** Use SQLite at `data/market.db` for structured storage, including forecast history and later outcome evaluation.
- **Alternative:** Keep all records only in CSVs or deploy a database server.
- **Reason:** SQLite is free, local, portable, and sufficient for a small academic project.
- **Design:** Forecast records retain origin date, model version, estimates/probabilities, warnings, and later actual outcomes where available.

## Decision 11: Separate raw, processed, and model-ready data

- **Decision:** Keep source downloads under `data/raw/`, transformed datasets under `data/processed/`, and the database under `data/`.
- **Reason:** Separating stages makes it easier to audit how raw records become modelling features and targets.
- **Git choice:** Ignore the generated `data/` directory, including `market.db`, raw/processed CSVs, and plots. Do not commit the full local dataset by default. If needed, add a small explicitly selected example dataset later.

## Decision 12: Automate the routine workflow with `run_pipeline.py`

- **Decision:** Provide one entry point that executes the main stages in order: price collection, recent news collection, alignment, sentiment, dataset construction, forecast generation, and outcome tracking.
- **Reason:** Reduces the risk of running stages in the wrong order and makes routine updates easier.
- **Operational note:** Evaluation and EDA scripts may be run separately when their outputs need to be refreshed.

## Decision 13: Report model limitations honestly

- **Decision:** Do not claim that the model is predictive or profitable until out-of-sample evidence supports that claim.
- **Reason:** Forecasting models should be judged against simple baselines on unseen, chronologically later observations.
- **Current walk-forward return results:**

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean-return baseline | 0.6146% | 0.8322% | -0.0034 |
| Market model | 0.6276% | 0.8447% | -0.0337 |
| Market-plus-sentiment model | 0.6299% | 0.8454% | -0.0356 |

- **Current direction results:**

| Model | Accuracy | Brier score |
|---|---:|---:|
| Market model | 47.10% | 0.2576 |
| Market-plus-sentiment model | 47.56% | 0.2580 |

- **Interpretation:** The market-only and market-plus-sentiment regression models performed worse than the mean-return baseline in this evaluation. The sentiment direction model had slightly higher accuracy but a slightly worse Brier score. Direction probabilities have not been calibrated.
- **Consequence:** Further work should focus first on leakage auditing, fair baselines, feature timing, and robust validation rather than increasing model complexity immediately.

## Decision 14: Treat news-time alignment as a known risk requiring audit

- **Current implementation:** News timestamps are converted to IST and assigned to the first eligible 15:30 IST close reference at or after publication time.
- **Reason for this rule:** It is a practical initial mapping between article publication time and daily market sessions.
- **Known risk:** It may not exactly match the real time at which a forecast would be generated. A headline must not affect a forecast made before that headline became available.
- **Next action:** Define a fixed forecast cutoff and audit all feature timestamps to ensure no future information enters the feature set.

## Fallback log

| ID | Earlier option / issue | Current free choice | Effect |
|---|---|---|---|
| F1 | NewsAPI was considered for historical news, but full multi-year history was not suitable under the free-only constraint. | Google News RSS for historical and recent headline collection. | No paid API required; archive completeness is not guaranteed. |
| F2 | Alpha Vantage was considered for market prices, but premium history is unnecessary for this project. | `yfinance` for NIFTY 50 price collection. | No paid market-data subscription required. |
| F3 | VADER was proposed in the original plan. | FinBERT via Hugging Face Transformers. | Financial-domain sentiment model is used; initial model download may require internet and local caching. |

## Open decisions / next work

1. Choose and document a fixed forecast-generation time and audit news/market timestamp alignment for look-ahead leakage.
2. Export and test a reproducible Conda environment (`environment.yml`) or dependency file.
3. Re-run walk-forward evaluation after any timing corrections, using identical evaluation dates for all baselines and models.
4. Investigate simpler feature sets and assess whether news features add information.
5. Add F1/confusion matrix for direction classification and evaluate probability calibration.
6. Consider alternative models only after the evaluation protocol is reliable.
7. If public reproducibility requires data examples, create a small sample dataset instead of committing the full `data/` directory.
8. Document that trading profitability has not been established; a future strategy evaluation would need costs, slippage, and risk metrics.


## Decision 10: Add volatility forecasting without replacing the original forecasts

- **Decision:** Preserve existing Ridge return forecasts and market/news direction probabilities, and add a separate Logistic Regression classifier for unusually large next-session absolute returns.
- **Volatility label:** The next-session absolute close-to-close return exceeds the trailing 80th percentile of absolute daily returns at the forecast origin; 252-session rolling window and 60-observation minimum history.
- **Evaluation:** The model is kept fixed for prospective evaluation. Alert indicators at 20% and 30% are informational and must not be optimized on the prospective period.
- **Storage:** Add volatility fields to the existing SQLite forecast table using additive migrations. A new model version keeps the integrated forecasts distinct from earlier records.
- **Evidence:** Retrospective results showed ROC-AUC 0.651 and Brier score 0.1493 compared with a 0.1624 historical-prevalence baseline, but performance varied substantially over time. This is a research lead, not a validated trading signal.
- **Limitation:** Forecast-time availability of same-date news features remains to be verified.
