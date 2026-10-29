# Integration and cleanup notes

## Integrated changes
- `src/generate_forecast.py`: preserves the existing market-only and market-plus-sentiment return and direction forecasts, adds a frozen volatility classifier, and stores volatility probability/threshold/alerts in SQLite.
- `src/record_forecast_outcome.py`: preserves prior outcome tracking and records whether the realised absolute return exceeded the threshold saved with each forecast.
- `README.md`, `context.md`, `decisions.md`: document the new model and its limitations.
- `run_pipeline.py` already invokes forecast generation and outcome tracking in the correct order; no change is required.

## Important limitations
- Forecast-time availability of same-date news features is not yet verified.
- Historical volatility metrics were computed on previously inspected dates. Do not claim independent validation or a validated trading strategy.
- The 20% and 30% alerts are informational. Do not tune them on prospective outcomes.
- Existing forecasts remain in SQLite. The new model uses version `ridge_logistic_volatility_v2`.

## Files not required by the routine pipeline
These are analysis / one-off experiment scripts, not generated files. They can stay locally for reproducibility; don't delete them just because the routine pipeline does not call them:
- `src/controlled_validation.py`
- `src/feature_ablation.py`
- `src/monthly_sampling_experiment.py`
- `src/new_feature_experiment.py`
- `src/pca_experiment.py`
- `src/statistical_significance.py`
- `src/validate_volatility_experiment.py`
- `src/volatility_forecast_experiment.py`
- `src/prospective_volatility_forecast.py`

Before removing any of these from GitHub, check whether they are currently tracked and whether the report refers to them. If you want to stop tracking them but keep local copies, use `git rm --cached` only after reviewing `git ls-files`.

## Generated files
`data/` is already ignored by `.gitignore`. Keep datasets, SQLite databases, prediction CSVs, bootstrap outputs, caches, and generated charts out of Git. Do not run `git add .` blindly.

## Safe review and commit procedure (Windows Anaconda Prompt)
First back up the local SQLite database and check the working tree:
```cmd
copy data\market.db data\market_backup_before_volatility_integration.db
git status --short
git ls-files
git diff -- .gitignore
```

After copying the integrated files into the project, stage only the intended code/docs:
```cmd
git add src/generate_forecast.py src/record_forecast_outcome.py README.md context.md decisions.md
git diff --cached --stat
git diff --cached
```
Inspect the staged diff. If it contains only the expected source and documentation changes:
```cmd
git commit -m "Integrate volatility forecasting into pipeline"
git push origin main
```
If Git reports conflicts, unexpected deletions, or a staged data/database file, stop and inspect before committing.
