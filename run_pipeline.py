
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
    ("Build model dataset", "src/build_model_dataset.py"),
    ("Generate next-session forecast", "src/generate_forecast.py"),
    ("Evaluate available forecast outcomes", "src/record_forecast_outcome.py"),
]
def main():
    for description, script in STEPS:
        script_path = ROOT / script

        print("\n" + "=" * 70, flush=True)
        print(f"STEP: {description}", flush=True)
        print("=" * 70, flush=True)

        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"\nPipeline stopped: {description} failed "
                f"with exit code {result.returncode}.",
                flush=True,
            )
            return result.returncode

    print("\nPipeline completed successfully.", flush=True)
    print("Review the forecast warnings and evaluation metrics before interpreting predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
