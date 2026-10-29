
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

INPUT_FILE = DATA / "walk_forward_predictions.csv"

SUMMARY_FILE = DATA / "sentiment_significance_summary.csv"
DISAGREEMENTS_FILE = DATA / "sentiment_prediction_disagreements.csv"
BOOTSTRAP_FILE = DATA / "sentiment_bootstrap_results.csv"

RANDOM_SEED = 42
BOOTSTRAP_REPETITIONS = 10000
BLOCK_LENGTH = 5
CONFIDENCE_LEVEL = 0.95


# ============================================================
# MOVING-BLOCK BOOTSTRAP
# ============================================================

def moving_block_bootstrap_ci(
    values,
    repetitions=BOOTSTRAP_REPETITIONS,
    block_length=BLOCK_LENGTH,
    seed=RANDOM_SEED,
):
    """
    Circular moving-block bootstrap for the mean of paired
    per-forecast differences.

    Contiguous blocks preserve some short-range temporal
    dependence between adjacent forecasts.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    if block_length < 1 or block_length > n:
        raise ValueError("Invalid block length.")

    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    bootstrap_means = np.empty(repetitions)

    for b in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)

        indices = np.concatenate([
            (start + offsets) % n
            for start in starts
        ])[:n]

        bootstrap_means[b] = values[indices].mean()

    alpha = 1 - CONFIDENCE_LEVEL

    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(lower), float(upper), bootstrap_means


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found:\n{INPUT_FILE}\n"
            "Run src\\evaluate_walk_forward.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = [
        "date",
        "target_next_session_date",
        "actual_up",
        "market_predicted_up",
        "sentiment_predicted_up",
        "market_p_up",
        "sentiment_p_up",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing values found in required prediction columns."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate forecast dates found; paired analysis stopped."
        )

    # Require valid binary labels and probabilities.
    for column in [
        "actual_up",
        "market_predicted_up",
        "sentiment_predicted_up",
    ]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    for column in ["market_p_up", "sentiment_p_up"]:
        if not df[column].between(0, 1).all():
            raise ValueError(f"{column} must be between 0 and 1.")

    actual = df["actual_up"].astype(int).to_numpy()
    market_pred = df["market_predicted_up"].astype(int).to_numpy()
    sentiment_pred = df["sentiment_predicted_up"].astype(int).to_numpy()

    market_prob = df["market_p_up"].astype(float).to_numpy()
    sentiment_prob = df["sentiment_p_up"].astype(float).to_numpy()

    n = len(df)

    # Correctness indicators.
    market_correct = (market_pred == actual).astype(int)
    sentiment_correct = (sentiment_pred == actual).astype(int)

    # Paired differences: positive means sentiment is better.
    accuracy_difference = sentiment_correct - market_correct

    # Per-observation Brier losses.
    market_brier_loss = (market_prob - actual) ** 2
    sentiment_brier_loss = (sentiment_prob - actual) ** 2

    # Negative Brier difference means sentiment is better.
    brier_difference = sentiment_brier_loss - market_brier_loss

    market_accuracy = market_correct.mean()
    sentiment_accuracy = sentiment_correct.mean()

    market_brier = market_brier_loss.mean()
    sentiment_brier = sentiment_brier_loss.mean()

    accuracy_diff = accuracy_difference.mean()
    brier_diff = brier_difference.mean()

    # McNemar's exact test:
    # Count forecasts where only one of the two models is correct.
    market_only_correct = int(
        ((market_correct == 1) & (sentiment_correct == 0)).sum()
    )
    sentiment_only_correct = int(
        ((market_correct == 0) & (sentiment_correct == 1)).sum()
    )

    discordant = market_only_correct + sentiment_only_correct

    if discordant > 0:
        mcnemar_p = binomtest(
            sentiment_only_correct,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        mcnemar_p = 1.0

    # Bootstrap confidence intervals for paired differences.
    accuracy_ci_low, accuracy_ci_high, accuracy_boot = (
        moving_block_bootstrap_ci(
            accuracy_difference,
            seed=RANDOM_SEED,
        )
    )

    brier_ci_low, brier_ci_high, brier_boot = (
        moving_block_bootstrap_ci(
            brier_difference,
            seed=RANDOM_SEED + 1,
        )
    )

    # Direction disagreement diagnostics.
    prediction_disagrees = market_pred != sentiment_pred

    df["market_correct"] = market_correct
    df["sentiment_correct"] = sentiment_correct
    df["accuracy_difference_sentiment_minus_market"] = accuracy_difference
    df["market_brier_loss"] = market_brier_loss
    df["sentiment_brier_loss"] = sentiment_brier_loss
    df["brier_difference_sentiment_minus_market"] = brier_difference
    df["predictions_disagree"] = prediction_disagrees

    disagreements = df.loc[prediction_disagrees].copy()
    disagreements.to_csv(DISAGREEMENTS_FILE, index=False)

    summary = pd.DataFrame([
        {
            "metric": "Accuracy",
            "market_model": market_accuracy,
            "sentiment_model": sentiment_accuracy,
            "difference_sentiment_minus_market": accuracy_diff,
            "ci_95_lower": accuracy_ci_low,
            "ci_95_upper": accuracy_ci_high,
            "test": "Moving-block bootstrap",
            "p_value": np.nan,
        },
        {
            "metric": "Brier score",
            "market_model": market_brier,
            "sentiment_model": sentiment_brier,
            "difference_sentiment_minus_market": brier_diff,
            "ci_95_lower": brier_ci_low,
            "ci_95_upper": brier_ci_high,
            "test": "Moving-block bootstrap",
            "p_value": np.nan,
        },
        {
            "metric": "Accuracy",
            "market_model": market_accuracy,
            "sentiment_model": sentiment_accuracy,
            "difference_sentiment_minus_market": accuracy_diff,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
            "test": "Exact McNemar test",
            "p_value": mcnemar_p,
        },
    ])

    summary.to_csv(SUMMARY_FILE, index=False)

    bootstrap_results = pd.DataFrame({
        "accuracy_difference": accuracy_boot,
        "brier_difference": brier_boot,
    })
    bootstrap_results.to_csv(BOOTSTRAP_FILE, index=False)

    # ========================================================
    # REPORT
    # ========================================================

    print("\n===== DATA AND EVALUATION PERIOD =====")
    print(f"Forecasts: {n}")
    print(f"From: {df['date'].min().date()}")
    print(f"To: {df['date'].max().date()}")
    print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}")
    print(f"Block length: {BLOCK_LENGTH} forecasts")

    print("\n===== ACCURACY =====")
    print(f"Market-only accuracy:     {market_accuracy:.4%}")
    print(f"Market + sentiment:       {sentiment_accuracy:.4%}")
    print(f"Difference:               {accuracy_diff:+.4%}")
    print(
        "95% block-bootstrap CI:   "
        f"[{accuracy_ci_low:+.4%}, {accuracy_ci_high:+.4%}]"
    )

    print("\n===== EXACT MCNEMAR TEST =====")
    print(f"Market correct, sentiment wrong: {market_only_correct}")
    print(f"Sentiment correct, market wrong: {sentiment_only_correct}")
    print(f"Total discordant forecasts:      {discordant}")
    print(f"Two-sided p-value:                {mcnemar_p:.6f}")

    print("\n===== BRIER SCORE =====")
    print(f"Market-only Brier score:   {market_brier:.6f}")
    print(f"Market + sentiment:        {sentiment_brier:.6f}")
    print(f"Difference (sentiment - market): {brier_diff:+.6f}")
    print(
        "95% block-bootstrap CI:   "
        f"[{brier_ci_low:+.6f}, {brier_ci_high:+.6f}]"
    )

    print("\n===== PREDICTION DISAGREEMENTS =====")
    print(f"Predictions differ on {len(disagreements)} of {n} dates.")
    print(f"Saved disagreement details:\n{DISAGREEMENTS_FILE}")

    print("\n===== SAVED OUTPUTS =====")
    print(SUMMARY_FILE)
    print(BOOTSTRAP_FILE)
    print(DISAGREEMENTS_FILE)

    print("\n===== PRELIMINARY INTERPRETATION =====")

    if accuracy_ci_low > 0:
        print(
            "The bootstrap interval for accuracy improvement is above zero."
        )
    elif accuracy_ci_high < 0:
        print(
            "The bootstrap interval suggests lower accuracy with sentiment."
        )
    else:
        print(
            "The bootstrap interval includes zero; accuracy improvement "
            "is not clearly distinguishable from zero."
        )

    if brier_ci_high < 0:
        print(
            "The bootstrap interval supports a lower Brier score with "
            "sentiment."
        )
    elif brier_ci_low > 0:
        print(
            "The bootstrap interval supports a higher Brier score with "
            "sentiment."
        )
    else:
        print(
            "The Brier-score interval includes zero; calibration improvement "
            "is uncertain."
        )

    if mcnemar_p < 0.05:
        print(
            "McNemar's test finds a statistically significant imbalance "
            "in discordant correctness outcomes at the 5% level."
        )
    else:
        print(
            "McNemar's test does not find a statistically significant "
            "difference at the 5% level."
        )

    print(
        "\nThese are exploratory estimates. The block-bootstrap interval "
        "depends on the chosen block length and the limited evaluation "
        "period; neither test establishes future trading profitability."
    )


if __name__ == "__main__":
    main()
