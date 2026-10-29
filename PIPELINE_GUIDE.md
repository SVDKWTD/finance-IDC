# Pipeline and Script Guide

## Active pipeline

Run from the repository root with the `finance-IDC` environment active:

```bat
python run_pipeline.py
```

The production pipeline uses the **monthly-sampled 2026 dataset as the official model input**. The cap is at most 500 articles per month in 2026; months below the cap and all articles from 2021–2025 are retained.

The runner executes, in order:

1. `src/collect_prices.py` — refresh market prices and processed market data.
2. `src/collect_recent_news.py` — collect recent Google News RSS headlines.
3. `src/import_collected_data.py` — import market/news records into SQLite.
4. `src/align_news_market.py` — assign news to market-session dates and build daily market/news features using the 15:45 IST cutoff.
5. `src/analyze_news_sentiment.py` — incrementally score headlines with FinBERT and aggregate daily sentiment.
6. `src/build_model_dataset.py` — build the full-corpus dataset for optional research comparison only.
7. `src/build_monthly_sampled_dataset.py` — build `data/processed/model_dataset_sampled_monthly_2026.csv`; this is the official forecast input.
8. `src/generate_forecast.py` — run the adaptive selector on the most recent two calendar months of completed out-of-sample targets; choose a return model/window and a direction model/window; fit the selected configurations; save the official forecast to `data/processed/latest_forecast.csv`; and persist the first-issued forecast in SQLite under model version `monthly_sampled_2026_adaptive_walkforward_v1`.
9. `src/record_forecast_outcome.py` — update pending database outcomes, report baseline/market/sentiment and adaptively selected model metrics, and write `data/processed/forecast_outcome_history.csv`. Previous capped-model records remain included where available; legacy full-corpus records remain excluded.

**Official output files:**

- `latest_forecast.csv` — latest freshly computed forecast from the capped dataset.
- `forecast_outcome_history.csv` — evaluated prospective outcomes for the capped model only.
- `model_dataset_sampled_monthly_2026.csv` — the sampled model input.
- `adaptive_model_selection.csv` — recent out-of-sample metrics by model family and training window.
- `adaptive_model_selection.json` — the selected return/direction configurations, evaluation dates, sample size, and fallback reason.

The full-corpus dataset is not used for the official forecast. Its comparison is available only when explicitly running `python src/generate_comparison_forecasts.py`; that script writes `latest_forecast_comparison.csv` for research and does not affect the official forecast or its prospective tracking.

**Forecast-history rule:** SQLite preserves the first-issued forecast for each `(forecast_origin_date, model_version)`. Re-running the pipeline can update `latest_forecast.csv` with a fresh calculation but will not overwrite the original prospective record for the same date and model version. This prevents evaluation from being rewritten after the outcome becomes known.

## Adaptive model selection (part of the normal pipeline)

No separate experiment command is required. Every normal `python run_pipeline.py` run invokes adaptive selection from inside `src/generate_forecast.py` using the official monthly-capped dataset and completed historical targets.

- Evaluation window: most recent two calendar months of eligible, genuinely out-of-sample predictions.
- Candidate training windows: expanding history and rolling 252 trading sessions.
- Return candidates: mean-return baseline, market/news-volume Ridge, and market/news/sentiment Ridge.
- Direction candidates: market/news-volume and market/news/sentiment logistic classifiers.
- Return selection: a feature model must have at least 30 recent predictions and beat the expanding mean baseline MAE by at least 2%; otherwise the mean baseline is selected.
- Direction selection: lowest recent Brier score among eligible classifiers; this is a separate choice from the return model.
- Insufficient recent predictions: conservative expanding-window defaults are used.
- The selected choices and candidate metrics are saved to `adaptive_model_selection.json` and `adaptive_model_selection.csv`.

This is periodic model selection/retraining, not continuous online learning. Two months can be noisy, so the selector logs its sample size and fallback reason and does not claim that the selected model is reliably profitable.

Historical dataset/sampling comparisons remain available as research scripts, but they are not run by the production pipeline and are not required for routine forecasting.

## Dataset contract

- `model_dataset.csv`: full-corpus sentiment and full-corpus article/source counts.
- `model_dataset_sampled_monthly_2026.csv`: all 2021–2025 articles plus at most 500 sampled articles per 2026 month; its article/source counts and sentiment are recomputed from the sampled corpus.
- `latest_forecast.csv`: official current-origin forecast from the monthly-sampled dataset only. The optional `generate_comparison_forecasts.py` writes `latest_forecast_comparison.csv` if you deliberately want a research comparison.

## Information timing caveat

`align_news_market.py` assigns an article to the first 15:45 IST feature cutoff at or after its publication timestamp, providing a 15-minute buffer after the nominal 15:30 IST market close. Articles published after 15:45 IST map to the next available trading session. Forecasts using session-t news features must be issued only after those features are available. Keep this cutoff consistent across dataset construction and evaluation.

## Supporting and experimental scripts

- Historical model evaluation: `evaluate_walk_forward.py`, `evaluate_forecasts.py`.
- Sampling evaluation: `monthly_sampling_experiment.py`, `compare_sampling_evaluation.py`.
- Alternative daily sampling: `sample_news_sentiment.py`, `build_sampled_model_dataset.py`.
- Feature experiments: `feature_ablation.py`, `new_feature_experiment.py`, `pca_experiment.py`.
- Validation and inference: `controlled_validation.py`, `statistical_significance.py`, `test_sentiment_significance.py`, `analyze_sentiment_relationship.py`.
- Volatility experiments: `volatility_forecast_experiment.py`, `validate_volatility_experiment.py`, `prospective_volatility_forecast.py`.
- Data auditing/cleaning: `audit_database.py`, `audit_historical_news.py`, `validate_historical_data.py`, `check_dataset_alignment.py`, `normalize_news_timestamps.py`, `process_data.py`, `clean.py`.
- Alternative/legacy workflows: `collect_news.py`, `import_data.py`, `analyze_sentiment.py`. These are not called by the active pipeline.
- Database helper: `db.py`. Tests: `test_db.py`, `test_historical_rss.py`.
- `backfill_historical_news.py` is a separate historical collection utility and is not called by the active pipeline.

Forecasts are experimental estimates, not investment advice. The current reported walk-forward results have not demonstrated that news-augmented models outperform the mean-return baseline. The official capped model needs its own prospective outcomes; legacy full-corpus scores are not evidence of capped-model performance.
