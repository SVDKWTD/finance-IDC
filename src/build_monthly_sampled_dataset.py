"""Build the monthly-capped 2026 dataset without running historical evaluation."""
from monthly_sampling_experiment import sample_articles, build_sampled_model_dataset


def main():
    sampled_daily = sample_articles()
    build_sampled_model_dataset(sampled_daily)
    print("Monthly-sampled model dataset is ready for live forecast comparison.")


if __name__ == "__main__":
    main()
