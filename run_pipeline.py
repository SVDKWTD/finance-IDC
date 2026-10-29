"""Run the pipeline with the monthly-capped dataset as the official model."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Update NIFTY 50 prices", "src/collect_prices.py"),
    ("Collect recent financial news", "src/collect_recent_news.py"),
    ("Import collected data into database", "src/import_collected_data.py"),
    ("Align news with market sessions", "src/align_news_market.py"),
    ("Score news sentiment (incremental)", "src/analyze_news_sentiment.py"),
    ("Build full-corpus dataset for optional research only", "src/build_model_dataset.py"),
    ("Build official monthly-capped dataset", "src/build_monthly_sampled_dataset.py"),
    ("Generate and persist official capped-dataset forecast", "src/generate_forecast.py"),
    ("Evaluate official capped-model forecast outcomes", "src/record_forecast_outcome.py"),
]


def main():
    for description, script in STEPS:
        script_path = ROOT / script
        print("\n" + "=" * 72, flush=True)
        print(f"STEP: {description}", flush=True)
        print("=" * 72, flush=True)
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")
        result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=False)
        if result.returncode:
            print(f"\nPipeline stopped: {description} failed with exit code {result.returncode}.", flush=True)
            return result.returncode
    print("\nPipeline completed successfully.")
    print("Official forecast: data/processed/latest_forecast.csv (monthly-capped dataset only).")
    print("Adaptive selection: recent two-calendar-month walk-forward evaluation runs inside forecast generation.")
    print("Selection metrics: data/processed/adaptive_model_selection.csv")
    print("Selected configuration: data/processed/adaptive_model_selection.json")
    print("Official prospective outcomes: data/processed/forecast_outcome_history.csv.")
    print("No separate experiment command is required for the normal workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
