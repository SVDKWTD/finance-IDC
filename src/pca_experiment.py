
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "model_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

INITIAL_TRAIN_FRACTION = 0.70
RIDGE_ALPHA = 10.0
PCA_COMPONENTS = [2, 3, 4, 5, 6, 8, 10]
RANDOM_STATE = 42

# Original 12 predictors used in the feature-ablation experiment.
MARKET_CORE = [
    "daily_return_pct",
    "return_lag1",
    "return_lag2",
    "return_lag5",
    "volatility_5",
]

NEWS_VOLUME = [
    "article_count",
    "unique_sources",
]

SENTIMENT = [
    "mean_sentiment",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
    "sentiment_lag1",
]

FULL_FEATURES = MARKET_CORE + NEWS_VOLUME + SENTIMENT


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def return_metrics(actual, predicted):
    return {
        "MAE_pct": mean_absolute_error(actual, predicted),
        "RMSE_pct": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def direction_metrics(actual_direction, predicted_direction, probabilities):
    return {
        "accuracy": accuracy_score(
            actual_direction, predicted_direction
        ),
        "brier_score": brier_score_loss(
            actual_direction, probabilities
        ),
        "high_confidence_count": int(
            np.sum((probabilities >= 0.60) | (probabilities <= 0.40))
        ),
        "high_confidence_accuracy": (
            accuracy_score(
                actual_direction[
                    (probabilities >= 0.60) | (probabilities <= 0.40)
                ],
                predicted_direction[
                    (probabilities >= 0.60) | (probabilities <= 0.40)
                ],
            )
            if np.any(
                (probabilities >= 0.60) | (probabilities <= 0.40)
            )
            else np.nan
        ),
    }


# ---------------------------------------------------------
# LOAD AND PREPARE DATA
# ---------------------------------------------------------

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = [
        "date",
        "daily_return_pct",
        "article_count",
        "unique_sources",
        "mean_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
        "target_next_session_return_pct",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in model_dataset.csv: {missing}"
        )

    # Recreate derived predictors consistently with the ablation experiment.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)

    df["volatility_5"] = (
        df["daily_return_pct"]
        .rolling(window=5)
        .std()
    )

    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    # Keep only rows with a target and all required predictors.
    needed = FULL_FEATURES + ["target_next_session_return_pct"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} usable rows remain. Check the input dataset."
        )

    return df


# ---------------------------------------------------------
# WALK-FORWARD EVALUATION
# ---------------------------------------------------------

def evaluate_variant(df, name, features, n_components=None):
    """
    At each test date:
      1. Use only earlier rows for training.
      2. Fit the scaler on training features.
      3. Fit PCA on scaled training features.
      4. Transform training and test features.
      5. Fit Ridge and LogisticRegression on training data.
      6. Predict the next-session return and direction.
    """

    X = df[features].to_numpy(dtype=float)
    y_return = df["target_next_session_return_pct"].to_numpy(dtype=float)
    y_direction = (y_return > 0).astype(int)

    dates = df["date"].to_numpy()

    split = int(len(df) * INITIAL_TRAIN_FRACTION)
    if split <= 0 or split >= len(df):
        raise ValueError("Invalid chronological train/test split.")

    if n_components is not None:
        if n_components > len(features):
            raise ValueError(
                f"{n_components} components exceed {len(features)} features."
            )

    predictions = []

    for test_idx in range(split, len(df)):
        # Expanding window: all rows before the current forecast.
        X_train = X[:test_idx]
        X_test = X[test_idx:test_idx + 1]

        y_train_return = y_return[:test_idx]
        y_train_direction = y_direction[:test_idx]

        # Scaling is fitted on training data only.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if n_components is not None:
            pca = PCA(
                n_components=n_components,
                svd_solver="full",
            )
            X_train_model = pca.fit_transform(X_train_scaled)
            X_test_model = pca.transform(X_test_scaled)
            explained_variance = float(
                np.sum(pca.explained_variance_ratio_)
            )
        else:
            # Benchmark: standardized original features, no PCA.
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
            explained_variance = np.nan

        # Return regression.
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train_model, y_train_return)
        predicted_return = float(ridge.predict(X_test_model)[0])

        # Direction classification.
        logistic = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        logistic.fit(X_train_model, y_train_direction)

        classes = list(logistic.classes_)
        if 1 in classes:
            up_col = classes.index(1)
            probability_up = float(
                logistic.predict_proba(X_test_model)[0, up_col]
            )
        else:
            probability_up = 0.0

        predicted_direction = int(probability_up >= 0.5)

        predictions.append({
            "date": pd.Timestamp(dates[test_idx]),
            "variant": name,
            "n_features": len(features),
            "n_components": (
                n_components if n_components is not None else len(features)
            ),
            "actual_return_pct": y_return[test_idx],
            "predicted_return_pct": predicted_return,
            "actual_direction": y_direction[test_idx],
            "predicted_direction": predicted_direction,
            "probability_up": probability_up,
            "pca_explained_variance": explained_variance,
        })

    return pd.DataFrame(predictions)


def evaluate_mean_baseline(df):
    """Expanding-window historical mean baseline for returns."""

    y = df["target_next_session_return_pct"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()

    split = int(len(df) * INITIAL_TRAIN_FRACTION)
    predictions = []

    for test_idx in range(split, len(df)):
        train_mean = float(np.mean(y[:test_idx]))

        predictions.append({
            "date": pd.Timestamp(dates[test_idx]),
            "variant": "Mean baseline",
            "n_features": 0,
            "n_components": 0,
            "actual_return_pct": y[test_idx],
            "predicted_return_pct": train_mean,
            "actual_direction": int(y[test_idx] > 0),
            "predicted_direction": np.nan,
            "probability_up": np.nan,
            "pca_explained_variance": np.nan,
        })

    return pd.DataFrame(predictions)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # PCA cannot use more components than the number of input features.
    component_counts = [
        n for n in PCA_COMPONENTS
        if n <= len(FULL_FEATURES)
    ]

    print("\n===== PCA EXPERIMENT =====")
    print(f"Dataset: {DATA_FILE}")
    print(f"Usable rows: {len(df)}")
    print(f"Predictors before PCA: {len(FULL_FEATURES)}")
    print(f"Initial training fraction: {INITIAL_TRAIN_FRACTION:.0%}")
    print(f"PCA component counts: {component_counts}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    all_predictions = []

    # Return-only mean baseline.
    print("\nEvaluating mean baseline...")
    all_predictions.append(evaluate_mean_baseline(df))

    # Benchmarks: full model and market-core model without PCA.
    print("Evaluating market-core benchmark...")
    all_predictions.append(
        evaluate_variant(
            df,
            "Market core benchmark",
            MARKET_CORE,
            n_components=None,
        )
    )

    print("Evaluating full-feature benchmark...")
    all_predictions.append(
        evaluate_variant(
            df,
            "Full-feature benchmark",
            FULL_FEATURES,
            n_components=None,
        )
    )

    # PCA models, all using the same 12 original predictors.
    for n_components in component_counts:
        name = f"PCA {n_components} components"
        print(f"Evaluating {name}...")

        all_predictions.append(
            evaluate_variant(
                df,
                name,
                FULL_FEATURES,
                n_components=n_components,
            )
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # Confirm every model uses exactly the same forecast dates and targets.
    reference = predictions[
        predictions["variant"] == "Full-feature benchmark"
    ][["date", "actual_return_pct"]].sort_values("date").reset_index(drop=True)

    for name, group in predictions.groupby("variant"):
        check = (
            group[["date", "actual_return_pct"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if not check["date"].equals(reference["date"]):
            raise RuntimeError(
                f"Forecast dates differ for variant: {name}"
            )

        if not np.allclose(
            check["actual_return_pct"],
            reference["actual_return_pct"],
        ):
            raise RuntimeError(
                f"Actual targets differ for variant: {name}"
            )

    # -----------------------------------------------------
    # RETURN METRICS
    # -----------------------------------------------------

    return_rows = []

    for name, group in predictions.groupby("variant", sort=False):
        actual = group["actual_return_pct"].to_numpy(dtype=float)
        predicted = group["predicted_return_pct"].to_numpy(dtype=float)

        metrics = return_metrics(actual, predicted)

        return_rows.append({
            "variant": name,
            "n_features": int(group["n_features"].iloc[0]),
            "n_components": int(group["n_components"].iloc[0]),
            **metrics,
            "mean_explained_variance": (
                group["pca_explained_variance"].mean()
            ),
        })

    return_results = pd.DataFrame(return_rows).sort_values(
        "MAE_pct"
    )

    # -----------------------------------------------------
    # DIRECTION METRICS
    # -----------------------------------------------------

    direction_rows = []

    for name, group in predictions.groupby("variant", sort=False):
        # The mean baseline has no directional probability model.
        if group["probability_up"].isna().any():
            continue

        actual = group["actual_direction"].to_numpy(dtype=int)
        predicted = group["predicted_direction"].to_numpy(dtype=int)
        probabilities = group["probability_up"].to_numpy(dtype=float)

        metrics = direction_metrics(
            actual,
            predicted,
            probabilities,
        )

        direction_rows.append({
            "variant": name,
            "n_features": int(group["n_features"].iloc[0]),
            "n_components": int(group["n_components"].iloc[0]),
            **metrics,
            "mean_explained_variance": (
                group["pca_explained_variance"].mean()
            ),
        })

    direction_results = pd.DataFrame(direction_rows).sort_values(
        ["brier_score", "accuracy"],
        ascending=[True, False],
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    return_file = OUTPUT_DIR / "pca_return_results.csv"
    direction_file = OUTPUT_DIR / "pca_direction_results.csv"
    predictions_file = OUTPUT_DIR / "pca_predictions.csv"

    return_results.to_csv(return_file, index=False)
    direction_results.to_csv(direction_file, index=False)
    predictions.to_csv(predictions_file, index=False)

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\n===== RETURN METRICS (LOWER MAE/RMSE IS BETTER) =====")
    print(
        return_results.to_string(
            index=False,
            formatters={
                "MAE_pct": "{:.4f}".format,
                "RMSE_pct": "{:.4f}".format,
                "R2": "{:.4f}".format,
                "mean_explained_variance": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
            },
        )
    )

    print("\n===== DIRECTION METRICS =====")
    print(
        direction_results.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "high_confidence_accuracy": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
                "mean_explained_variance": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
            },
        )
    )

    print("\n===== SAVED FILES =====")
    print(return_file)
    print(direction_file)
    print(predictions_file)

    print(
        "\nNote: PCA explained variance measures predictor variance retained, "
        "not predictive accuracy."
    )
    print(
        "Use the results to select candidates only; do not treat the best "
        "variant on this test period as independently validated."
    )


if __name__ == "__main__":
    main()
