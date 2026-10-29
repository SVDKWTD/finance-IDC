
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    DATA_DIR / "controlled_validation_holdout_predictions.csv"
)

OUTPUT_SUMMARY = DATA_DIR / "statistical_significance_summary.csv"
OUTPUT_BOOTSTRAP = DATA_DIR / "statistical_significance_bootstrap.csv"
OUTPUT_DISAGREEMENTS = DATA_DIR / "statistical_significance_disagreements.csv"

N_BOOTSTRAP = 10000
BLOCK_LENGTH = 5
RANDOM_SEED = 42


# =========================================================
# CONFIDENCE INTERVALS
# =========================================================

def wilson_interval(successes, n, confidence_z=1.959963984540054):
    """Approximate 95% Wilson confidence interval for a proportion."""

    if n == 0:
        return np.nan, np.nan

    p = successes / n
    z = confidence_z
    denom = 1 + z**2 / n

    centre = (p + z**2 / (2 * n)) / denom
    margin = (
        z
        * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        / denom
    )

    return centre - margin, centre + margin


# =========================================================
# CIRCULAR MOVING-BLOCK BOOTSTRAP
# =========================================================

def circular_block_bootstrap_means(
    values,
    n_bootstrap=N_BOOTSTRAP,
    block_length=BLOCK_LENGTH,
    seed=RANDOM_SEED,
):
    """
    Resample consecutive blocks, wrapping around at the end.
    This preserves some short-range serial dependence.

    Returns bootstrap estimates of the mean.
    """

    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    if block_length < 1:
        raise ValueError("Block length must be positive.")

    block_length = min(block_length, n)
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)

    results = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)

        indices = (
            starts[:, None] + offsets[None, :]
        ) % n

        sample = values[indices.ravel()[:n]]
        results[b] = np.mean(sample)

    return results


def bootstrap_summary(
    values,
    label,
    n_bootstrap=N_BOOTSTRAP,
    block_length=BLOCK_LENGTH,
    seed=RANDOM_SEED,
    null_value=None,
):
    """
    Calculate a block-bootstrap percentile CI.

    If null_value is provided, calculate a two-sided approximate
    bootstrap p-value by resampling a series recentered at the null.
    """

    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))

    boot = circular_block_bootstrap_means(
        values,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        seed=seed,
    )

    lower, upper = np.quantile(boot, [0.025, 0.975])

    result = {
        "test": label,
        "n": len(values),
        "observed_estimate": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_p_value": np.nan,
        "block_length": min(block_length, len(values)),
        "bootstrap_repetitions": n_bootstrap,
    }

    if null_value is not None:
        # Recenter the observed series under the null hypothesis.
        null_series = values - observed + null_value

        boot_null = circular_block_bootstrap_means(
            null_series,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed + 1,
        )

        observed_distance = abs(observed - null_value)
        null_distances = np.abs(boot_null - null_value)

        p_value = (
            1 + np.sum(null_distances >= observed_distance)
        ) / (n_bootstrap + 1)

        result["bootstrap_p_value"] = float(p_value)

    return result, boot


# =========================================================
# LOAD AND ALIGN HOLDOUT PREDICTIONS
# =========================================================

def load_predictions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find holdout predictions:\n{INPUT_FILE}\n\n"
            "Run controlled_validation.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    required = [
        "date",
        "variant",
        "actual_return_pct",
        "predicted_return_pct",
        "actual_direction",
        "predicted_direction",
        "probability_up",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in holdout file: {missing}")

    pca = df[df["variant"] == "PCA-2"].copy()
    full = df[df["variant"] == "Full features"].copy()

    if pca.empty or full.empty:
        available = sorted(df["variant"].dropna().unique())
        raise ValueError(
            "Expected variants 'PCA-2' and 'Full features'. "
            f"Found: {available}"
        )

    # Align by date rather than assuming row order.
    pca = pca.rename(columns={
        "actual_return_pct": "actual_return_pca",
        "predicted_return_pct": "predicted_return_pca",
        "actual_direction": "actual_direction_pca",
        "predicted_direction": "predicted_direction_pca",
        "probability_up": "probability_up_pca",
    })

    full = full.rename(columns={
        "actual_return_pct": "actual_return_full",
        "predicted_return_pct": "predicted_return_full",
        "actual_direction": "actual_direction_full",
        "predicted_direction": "predicted_direction_full",
        "probability_up": "probability_up_full",
    })

    merged = pca.merge(
        full,
        on="date",
        how="inner",
        suffixes=("_pca_row", "_full_row"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No overlapping dates between PCA-2 and full model.")

    if not np.array_equal(
        merged["actual_direction_pca"].to_numpy(dtype=int),
        merged["actual_direction_full"].to_numpy(dtype=int),
    ):
        raise ValueError("Actual direction labels differ across models.")

    if not np.allclose(
        merged["actual_return_pca"].to_numpy(dtype=float),
        merged["actual_return_full"].to_numpy(dtype=float),
    ):
        raise ValueError("Actual returns differ across models.")

    merged = merged.sort_values("date").reset_index(drop=True)

    merged["pca_correct"] = (
        merged["predicted_direction_pca"].astype(int)
        == merged["actual_direction_pca"].astype(int)
    ).astype(int)

    merged["full_correct"] = (
        merged["predicted_direction_full"].astype(int)
        == merged["actual_direction_full"].astype(int)
    ).astype(int)

    # Positive difference means PCA-2 is better.
    merged["accuracy_diff_pca_minus_full"] = (
        merged["pca_correct"] - merged["full_correct"]
    )

    # Brier loss is squared probability error.
    actual = merged["actual_direction_pca"].to_numpy(dtype=float)
    p_pca = merged["probability_up_pca"].to_numpy(dtype=float)
    p_full = merged["probability_up_full"].to_numpy(dtype=float)

    merged["brier_pca_per_date"] = (p_pca - actual) ** 2
    merged["brier_full_per_date"] = (p_full - actual) ** 2

    # Positive difference means PCA-2 has lower Brier loss.
    merged["brier_improvement_pca_minus_full"] = (
        merged["brier_full_per_date"] - merged["brier_pca_per_date"]
    )

    if merged[
        ["probability_up_pca", "probability_up_full"]
    ].isna().any().any():
        raise ValueError("Missing predicted probabilities in comparison.")

    return merged


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()

    n = len(df)
    pca_correct = int(df["pca_correct"].sum())
    full_correct = int(df["full_correct"].sum())

    pca_accuracy = pca_correct / n
    full_accuracy = full_correct / n

    pca_ci = wilson_interval(pca_correct, n)
    full_ci = wilson_interval(full_correct, n)

    # Exact binomial tests: useful as reference, but assume independent
    # Bernoulli outcomes and do not account for serial dependence.
    pca_binomial = binomtest(
        pca_correct, n, p=0.5, alternative="two-sided"
    )
    full_binomial = binomtest(
        full_correct, n, p=0.5, alternative="two-sided"
    )

    # McNemar exact test for paired correctness outcomes.
    pca_right_full_wrong = int(
        (
            (df["pca_correct"] == 1)
            & (df["full_correct"] == 0)
        ).sum()
    )

    pca_wrong_full_right = int(
        (
            (df["pca_correct"] == 0)
            & (df["full_correct"] == 1)
        ).sum()
    )

    discordant = pca_right_full_wrong + pca_wrong_full_right

    mcnemar_p = (
        binomtest(
            pca_right_full_wrong,
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant > 0
        else 1.0
    )

    # Paired Brier scores.
    pca_brier = float(df["brier_pca_per_date"].mean())
    full_brier = float(df["brier_full_per_date"].mean())

    # Bootstrap intervals accounting for local serial dependence.
    bootstrap_rows = []
    bootstrap_distributions = {}

    series_to_test = [
        (
            "PCA-2 accuracy",
            df["pca_correct"].to_numpy(dtype=float),
            0.5,
        ),
        (
            "Full-feature accuracy",
            df["full_correct"].to_numpy(dtype=float),
            0.5,
        ),
        (
            "Accuracy difference: PCA-2 minus full",
            df["accuracy_diff_pca_minus_full"].to_numpy(dtype=float),
            0.0,
        ),
        (
            "Brier improvement: PCA-2 minus full",
            df["brier_improvement_pca_minus_full"].to_numpy(dtype=float),
            0.0,
        ),
    ]

    for i, (label, values, null_value) in enumerate(series_to_test):
        result, boot = bootstrap_summary(
            values,
            label,
            null_value=null_value,
            seed=RANDOM_SEED + i * 10,
        )

        bootstrap_rows.append(result)
        bootstrap_distributions[label] = boot

    # Save a per-date table for inspection.
    df[[
        "date",
        "actual_direction_pca",
        "predicted_direction_pca",
        "probability_up_pca",
        "pca_correct",
        "predicted_direction_full",
        "probability_up_full",
        "full_correct",
        "accuracy_diff_pca_minus_full",
        "brier_pca_per_date",
        "brier_full_per_date",
        "brier_improvement_pca_minus_full",
    ]].to_csv(OUTPUT_DISAGREEMENTS, index=False)

    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT_BOOTSTRAP, index=False
    )

    # Summary rows use common estimate / interval / p-value columns.
    summary_rows = [
        {
            "test": "PCA-2 accuracy (Wilson interval; independence-based)",
            "n": n,
            "estimate": pca_accuracy,
            "ci95_lower": pca_ci[0],
            "ci95_upper": pca_ci[1],
            "p_value": pca_binomial.pvalue,
            "note": "Exact binomial p-value assumes independent outcomes.",
        },
        {
            "test": "Full-feature accuracy (Wilson interval; independence-based)",
            "n": n,
            "estimate": full_accuracy,
            "ci95_lower": full_ci[0],
            "ci95_upper": full_ci[1],
            "p_value": full_binomial.pvalue,
            "note": "Exact binomial p-value assumes independent outcomes.",
        },
        {
            "test": "McNemar exact: PCA-2 vs full correctness",
            "n": n,
            "estimate": pca_accuracy - full_accuracy,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "p_value": mcnemar_p,
            "note": (
                f"PCA correct/full wrong={pca_right_full_wrong}; "
                f"PCA wrong/full correct={pca_wrong_full_right}. "
                "McNemar assumes independent paired units across dates."
            ),
        },
        {
            "test": "PCA-2 mean Brier score",
            "n": n,
            "estimate": pca_brier,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "p_value": np.nan,
            "note": "Lower is better.",
        },
        {
            "test": "Full-feature mean Brier score",
            "n": n,
            "estimate": full_brier,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "p_value": np.nan,
            "note": "Lower is better.",
        },
        {
            "test": "Brier improvement: PCA-2 minus full",
            "n": n,
            "estimate": full_brier - pca_brier,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "p_value": np.nan,
            "note": "Positive means PCA-2 has lower Brier score.",
        },
    ]

    # Append block-bootstrap results.
    for row in bootstrap_rows:
        summary_rows.append({
            "test": f"Block bootstrap: {row['test']}",
            "n": row["n"],
            "estimate": row["observed_estimate"],
            "ci95_lower": row["ci95_lower"],
            "ci95_upper": row["ci95_upper"],
            "p_value": row["bootstrap_p_value"],
            "note": (
                f"Circular moving-block bootstrap; "
                f"block length={row['block_length']}, "
                f"replicates={row['bootstrap_repetitions']}. "
                "Approximate inference; sensitive to block length."
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    # =====================================================
    # PRINT REPORT
    # =====================================================

    print("\n===== STATISTICAL SIGNIFICANCE ANALYSIS =====")
    print(f"Input: {INPUT_FILE}")
    print(f"Matched holdout dates: {n}")
    print(
        f"Date range: {df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )

    print("\n===== DIRECTION ACCURACY =====")
    print(
        f"PCA-2: {pca_correct}/{n} = {pca_accuracy:.4%}\n"
        f"  Wilson 95% CI: [{pca_ci[0]:.4%}, {pca_ci[1]:.4%}]\n"
        f"  Exact binomial p vs 50%: {pca_binomial.pvalue:.6f}"
    )
    print(
        f"\nFull features: {full_correct}/{n} = {full_accuracy:.4%}\n"
        f"  Wilson 95% CI: [{full_ci[0]:.4%}, {full_ci[1]:.4%}]\n"
        f"  Exact binomial p vs 50%: {full_binomial.pvalue:.6f}"
    )

    print("\n===== PAIRED CORRECTNESS TEST =====")
    print(f"PCA correct / full wrong: {pca_right_full_wrong}")
    print(f"PCA wrong / full correct: {pca_wrong_full_right}")
    print(f"Accuracy difference PCA - full: {pca_accuracy - full_accuracy:.4%}")
    print(f"Exact McNemar p-value: {mcnemar_p:.6f}")

    print("\n===== BRIER SCORES =====")
    print(f"PCA-2 Brier score: {pca_brier:.6f}")
    print(f"Full-feature Brier score: {full_brier:.6f}")
    print(
        "Brier improvement (full - PCA): "
        f"{full_brier - pca_brier:.6f}"
    )

    print("\n===== BLOCK BOOTSTRAP RESULTS =====")
    print(
        pd.DataFrame(bootstrap_rows).to_string(
            index=False,
            formatters={
                "observed_estimate": "{:.6f}".format,
                "ci95_lower": "{:.6f}".format,
                "ci95_upper": "{:.6f}".format,
                "bootstrap_p_value": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.6f}"
                ),
            },
        )
    )

    print("\n===== SAVED FILES =====")
    print(OUTPUT_SUMMARY)
    print(OUTPUT_BOOTSTRAP)
    print(OUTPUT_DISAGREEMENTS)

    print("\n===== INTERPRETATION NOTES =====")
    print(
        "1. Wilson intervals and exact binomial tests are reference analyses; "
        "they do not account for serial dependence."
    )
    print(
        "2. McNemar tests paired correctness but assumes independent dates."
    )
    print(
        "3. Block-bootstrap intervals account for short-range dependence "
        "approximately, not perfectly."
    )
    print(
        "4. Statistical significance does not establish profitability "
        "or eliminate selection bias from prior experimentation."
    )


if __name__ == "__main__":
    main()
