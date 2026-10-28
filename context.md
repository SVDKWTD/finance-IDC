# context.md — Project Handoff

> Purpose: help a future assistant continue the project without asking the user to repeat established details. Update the **Current State** and **Next Actions** when the project changes.

## 1. How to work with the user

- The user is learning Python, data science, and Git. Give clear, step-by-step instructions and explain unfamiliar terms.
- Prefer small, verifiable steps over long batches of commands.
- Use plain-language explanations, readable code, and practical examples.
- Do not claim that a model is useful unless evaluation supports that claim.
- **Free-only rule:** avoid paid data providers and services. If a paid option comes up, recommend a free alternative and record the choice in `decisions.md`.
- Keep `context.md`, `decisions.md`, and (if maintained) `flow.md` aligned with the actual project, not the original plan when it has changed.

## 2. Project overview

- **Course:** IDC409, Topic 3: Web Scraping I — Finance + News.
- **Repository/project folder:** `finance-IDC-Project`
- **Primary goal:** investigate whether financial-news sentiment and news activity improve prediction of the next-session NIFTY 50 return or direction when compared with historical market features and simple baselines.
- **Current status:** end-to-end data collection, preprocessing, sentiment analysis, model generation, forecast persistence, and outcome tracking are implemented. This is a research prototype; current evaluation does **not** show that the news-based models outperform a simple mean-return baseline.
- **Forecast target:** next available trading session's close-to-close return, expressed as a percentage. A separate classifier predicts UP/DOWN direction.
- **Important:** forecasts are experimental estimates, not investment advice or a validated trading strategy.

## 3. Current tools and sources

- Python, pandas, NumPy
- Google News RSS for historical/recent headline collection
- `yfinance` for NIFTY 50 market prices
- Hugging Face Transformers and `ProsusAI/finbert` for financial headline sentiment
- scikit-learn: Ridge regression and Logistic Regression
- SQLite for persistent storage and forecast history
- Matplotlib / Seaborn and Jupyter for exploratory analysis
- Conda environment: `finance-IDC`
- Git and GitHub for version control

The project currently uses Google News RSS rather than NewsAPI/GDELT, and FinBERT rather than VADER. Do not repeat the old proposed source plan as if it were implemented.

## 4. Pipeline entry point and order

Run from Anaconda Prompt in the project root with the `finance-IDC` environment active:

```cmd
python run_pipeline.py
```

The main pipeline runs these scripts in order:

1. `src/collect_prices.py` — incrementally collect/update NIFTY 50 prices.
2. `src/collect_recent_news.py` — collect recent headlines using a rolling overlap window and deduplicate against saved records.
3. `src/align_news_market.py` — align news timestamps to market-session dates.
4. `src/analyze_news_sentiment.py` — score headlines with FinBERT, cache scores, and rebuild daily aggregates.
5. `src/build_model_dataset.py` — combine market and news features and construct the next-session target.
6. `src/generate_forecast.py` — fit the configured models, generate forecasts, and save them to SQLite.
7. `src/record_forecast_outcome.py` — evaluate forecasts once target-session prices are available.

Other scripts support historical collection, cleaning, timestamp normalisation, database/data audits, EDA, and model evaluation. See the actual `run_pipeline.py` and `src/` files if the order changes.

## 5. Current data flow and main outputs

`data/` is local generated data and is ignored by Git. Do not add the whole directory to GitHub.

Raw data examples:
- `data/raw/historical_news_2021_2025.csv`
- `data/raw/recent_news_2026.csv`
- `data/raw/nifty50_prices.csv`
- monthly coverage reports

Processed data examples:
- `data/processed/news_2021_2025_clean.csv`
- `data/processed/news_market_alignment.csv`
- `data/processed/news_market_daily.csv`
- `data/processed/news_sentiment_scored.csv`
- `data/processed/news_sentiment_daily.csv`
- `data/processed/nifty50_processed.csv`
- `data/processed/model_dataset.csv`
- walk-forward predictions and model comparison CSVs
- EDA plots and summaries

Database:
- `data/market.db` — SQLite database containing stored market/news records and forecast history.

The exact files depend on which scripts have been run. The repository intentionally excludes generated datasets and the database to avoid committing large local artifacts.

## 6. Data and target details

- The current price pipeline excludes the current incomplete market session from processed prices; raw downloads may include an incomplete current-day row.
- News timestamps are converted to India Standard Time (IST) and assigned to the first eligible 15:30 IST market-close reference at or after publication time. This is an approximation and must be audited against the actual forecast cutoff to prevent look-ahead leakage.
- The model dataset contains market features, news counts/source counts, daily sentiment aggregates, and `target_next_session_return_pct`.
- FinBERT labels headlines positive, negative, or neutral. Daily features include mean/median sentiment and positive/negative/neutral fractions.
- Forecast target dates can initially be blank because the next exchange trading session may not be the next calendar day. Outcome tracking resolves pending forecasts using later completed prices.

## 7. Models and evaluation

Current forecasting implementation:
- Ridge regression for numerical next-session return.
- Logistic Regression for UP/DOWN direction probabilities.
- Market-only and market-plus-news/sentiment variants.
- Time-ordered walk-forward evaluation.

Most recently recorded walk-forward results:
- Mean-return baseline: MAE 0.6146%, RMSE 0.8322%, R² -0.0034.
- Market model: MAE 0.6276%, RMSE 0.8447%, R² -0.0337.
- Market-plus-sentiment model: MAE 0.6299%, RMSE 0.8454%, R² -0.0356.
- Direction model: market accuracy 47.10%, Brier score 0.2576; market-plus-sentiment accuracy 47.56%, Brier score 0.2580.

Interpretation: the news-based return models did not beat the mean baseline on this evaluation. The sentiment direction model had slightly higher accuracy but a slightly worse Brier score. Do not claim proven predictive value. Probabilities are not calibrated.

Initial sentiment relationship analysis:
- Pearson r ≈ 0.0303, p ≈ 0.2532.
- Spearman r ≈ 0.0541, p ≈ 0.0410.
- Welch test comparing next-session returns on positive- and negative-sentiment days: p ≈ 0.2949.
These are small associations and do not establish economically useful prediction.

## 8. Git/GitHub status and workflow

- The project is on branch `main`, connected to `origin/main`.
- `.gitignore` excludes `data/`, local database files, Python caches, environment files, and other generated artifacts.
- The code and notebooks were staged/committed/pushed in the previous Git workflow. The user has also been preparing a root `README.md`; confirm with `git status` whether the README and these handoff files have been committed yet.
- Useful commands from the project root:

```cmd
git status
git add README.md context.md decisions.md
git commit -m "Update project documentation and handoff notes"
git push origin main
```

Only run the commit commands if those files exist and the user has reviewed them. Do not add `data/` or use `git add .` without inspecting the staged changes.

## 9. Current State (as of 2026-10-28)

### Completed
- Historical and incremental news collection scripts.
- Incremental NIFTY 50 price collection.
- SQLite storage and data import/audit utilities.
- News-market alignment and timestamp processing.
- FinBERT sentiment scoring with a cache and daily aggregation.
- Model dataset construction.
- Exploratory data analysis and sentiment/return analysis.
- Initial split and walk-forward forecast evaluation.
- Forecast generation and persistence to SQLite.
- Pending forecast outcome tracking.
- Main `run_pipeline.py` executes successfully.
- A README draft has been prepared explaining the architecture, tools, pipeline, evaluation, limitations, and execution commands.
- Git repository configured and `.gitignore` excludes local generated data.

### Known limitations
- News-session alignment has not yet been fully audited for leakage.
- Model performance does not beat the mean-return baseline in the latest walk-forward evaluation.
- Direction probabilities are not calibrated.
- Current RSS collection is a sample, not a complete archive of all financial news.
- Reproducible environment/dependency documentation should be verified against the actual Conda environment.
- Generated data is excluded from GitHub; a small sample dataset may be useful later for reproducible demonstrations.

## 10. Next actions (recommended order)

1. Update and commit `README.md`, `context.md`, and `decisions.md` to GitHub; verify with `git status` and the GitHub repository page.
2. Export the actual Conda environment to `environment.yml` (or create a tested `requirements.txt`) so setup can be reproduced.
3. Audit the news timestamp assignment and define an explicit forecast cutoff; ensure every feature uses only information available at that time.
4. Re-run time-ordered walk-forward evaluation after the leakage audit and compare all models on the same dates.
5. Investigate baseline performance and feature usefulness; test simpler and alternative feature sets without tuning on the final test period.
6. Calibrate direction probabilities using only training/validation predictions, and report calibration diagnostics.
7. Add a small, non-sensitive sample dataset or a reproducible sample-output section if helpful; keep the full generated `data/` directory out of Git.
8. Add tests for return/target calculations, deduplication, alignment, and forecast outcome matching.

## 11. Conventions

- Keep API credentials and secrets out of source control; `.env` should remain ignored.
- Keep code comments clear and explain the purpose of each stage.
- Record meaningful methodology changes and reasons in `decisions.md`.
- Prefer chronological validation for time-series forecasting.
- Clearly distinguish implemented features from proposed future work.
