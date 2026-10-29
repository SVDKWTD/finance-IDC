
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

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "model_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

RIDGE_ALPHA = 10.0
INITIAL_TRAIN_FRACTION = 0.60
DEVELOPMENT_END_FRACTION = 0.80
PCA_COMPONENTS = [2, 3, 4, 5, 6, 8, 10]
RANDOM_STATE = 42

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


# =========================================================
# DATA PREPARATION
# =========================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Recreate predictors consistently with the earlier experiments.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    needed = FULL_FEATURES + ["target_next_session_return_pct"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed).reset_index(drop=True)

    if len(df) < 200:
        raise ValueError(f"Too few usable observations: {len(df)}")

    return df


# =========================================================
# MODEL FITTING
# =========================================================

def fit_and_predict(X_train, y_train_return, y_train_direction,
                    X_test, method, n_components=None):
    """
    Every scaler and PCA object is fitted on training rows only.
    Returns predicted return, probability of an up session,
    and average PCA variance explained.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    explained = np.nan

    if method == "pca":
        if n_components is None or n_components > X_train_scaled.shape[1]:
            raise ValueError("Invalid PCA component count.")

        pca = PCA(n_components=n_components, svd_solver="full")
        X_train_model = pca.fit_transform(X_train_scaled)
        X_test_model = pca.transform(X_test_scaled)
        explained = float(pca.explained_variance_ratio_.sum())

    elif method in ("market", "full"):
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled

    else:
        raise ValueError(f"Unknown model method: {method}")

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train_model, y_train_return)
    pred_return = float(ridge.predict(X_test_model)[0])

    logistic = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    logistic.fit(X_train_model, y_train_direction)

    classes = list(logistic.classes_)
    if 1 in classes:
        p_up = float(
            logistic.predict_proba(X_test_model)[0, classes.index(1)]
        )
    else:
        p_up = 0.0

    return pred_return, p_up, explained


def forecast_range(df, start_idx, end_idx, candidate_specs):
    """
    Expanding-window forecasts for indices [start_idx, end_idx).
    Every candidate forecasts exactly the same dates.
    """

    X_all = df[FULL_FEATURES].to_numpy(dtype=float)
    y_return = df["target_next_session_return_pct"].to_numpy(dtype=float)
    y_direction = (y_return > 0).astype(int)
    dates = df["date"].to_numpy()

    rows = []

    for test_idx in range(start_idx, end_idx):
        X_train_full = X_all[:test_idx]
        X_test_full = X_all[test_idx:test_idx + 1]

        yr_train = y_return[:test_idx]
        yd_train = y_direction[:test_idx]

        # Expanding historical-mean baseline.
        mean_prediction = float(np.mean(yr_train))

        rows.append({
            "date": pd.Timestamp(dates[test_idx]),
            "variant": "Mean baseline",
            "actual_return_pct": y_return[test_idx],
            "predicted_return_pct": mean_prediction,
            "actual_direction": y_direction[test_idx],
            "predicted_direction": np.nan,
            "probability_up": np.nan,
            "pca_explained_variance": np.nan,
        })

        for spec in candidate_specs:
            name = spec["name"]
            method = spec["method"]
            features = spec["features"]
            components = spec.get("components")

            feature_idx = [FULL_FEATURES.index(f) for f in features]
            X_train = X_train_full[:, feature_idx]
            X_test = X_test_full[:, feature_idx]

            pred_return, p_up, explained = fit_and_predict(
                X_train,
                yr_train,
                yd_train,
                X_test,
                method=method,
                n_components=components,
            )

            rows.append({
                "date": pd.Timestamp(dates[test_idx]),
                "variant": name,
                "actual_return_pct": y_return[test_idx],
                "predicted_return_pct": pred_return,
                "actual_direction": y_direction[test_idx],
                "predicted_direction": int(p_up >= 0.5),
                "probability_up": p_up,
                "pca_explained_variance": explained,
            })

        if (test_idx - start_idx + 1) % 50 == 0:
            print(
                f"Forecasted {test_idx - start_idx + 1} "
                f"of {end_idx - start_idx} dates..."
            )

    return pd.DataFrame(rows)


# =========================================================
# METRICS
# =========================================================

def summarize_predictions(predictions, period):
    return_rows = []
    direction_rows = []

    for name, group in predictions.groupby("variant", sort=False):
        actual = group["actual_return_pct"].to_numpy(dtype=float)
        predicted = group["predicted_return_pct"].to_numpy(dtype=float)

        return_rows.append({
            "period": period,
            "variant": name,
            "n_forecasts": len(group),
            "MAE_pct": mean_absolute_error(actual, predicted),
            "RMSE_pct": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
        })

        if group["probability_up"].isna().any():
            continue

        actual_dir = group["actual_direction"].to_numpy(dtype=int)
        predicted_dir = group["predicted_direction"].to_numpy(dtype=int)
        p_up = group["probability_up"].to_numpy(dtype=float)

        high = (p_up >= 0.60) | (p_up <= 0.40)

        direction_rows.append({
            "period": period,
            "variant": name,
            "n_forecasts": len(group),
            "accuracy": accuracy_score(actual_dir, predicted_dir),
            "brier_score": brier_score_loss(actual_dir, p_up),
            "high_confidence_count": int(high.sum()),
            "high_confidence_accuracy": (
                accuracy_score(actual_dir[high], predicted_dir[high])
                if high.any() else np.nan
            ),
        })

    return pd.DataFrame(return_rows), pd.DataFrame(direction_rows)


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    n = len(df)
    initial_train_end = int(n * INITIAL_TRAIN_FRACTION)
    development_end = int(n * DEVELOPMENT_END_FRACTION)

    if initial_train_end >= development_end or development_end >= n:
        raise ValueError("Invalid development/holdout split.")

    # Candidates evaluated on the development window.
    development_specs = [
        {
            "name": "Market core",
            "method": "market",
            "features": MARKET_CORE,
        },
        {
            "name": "Full features",
            "method": "full",
            "features": FULL_FEATURES,
        },
    ]

    for k in PCA_COMPONENTS:
        if k <= len(FULL_FEATURES):
            development_specs.append({
                "name": f"PCA-{k}",
                "method": "pca",
                "features": FULL_FEATURES,
                "components": k,
            })

    print("\n===== CONTROLLED VALIDATION =====")
    print(f"Dataset: {DATA_FILE}")
    print(f"Usable rows: {n}")
    print(f"Initial training rows: {initial_train_end}")
    print(
        f"Development window: rows {initial_train_end + 1} "
        f"to {development_end}"
    )
    print(
        f"Final holdout: rows {development_end + 1} to {n}"
    )
    print("The final holdout will not be used to select PCA components.")

    # -----------------------------------------------------
    # 1. DEVELOPMENT WINDOW
    # -----------------------------------------------------

    print("\n===== PHASE 1: DEVELOPMENT =====")
    print("Evaluating candidates for selection...")

    dev = forecast_range(
        df,
        initial_train_end,
        development_end,
        development_specs,
    )

    dev_returns, dev_directions = summarize_predictions(
        dev, "development"
    )

    # Select PCA component count using development Brier score.
    pca_dev = dev_directions[
        dev_directions["variant"].str.startswith("PCA-")
    ].copy()

    pca_dev = pca_dev.sort_values(
        ["brier_score", "accuracy"],
        ascending=[True, False],
    )

    if pca_dev.empty:
        raise RuntimeError("No PCA development results were produced.")

    selected_name = pca_dev.iloc[0]["variant"]
    selected_k = int(selected_name.split("-")[1])

    print("\nPCA selection based on development Brier score:")
    print(
        pca_dev.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
            },
        )
    )
    print(f"\nSelected PCA configuration: {selected_name}")

    # -----------------------------------------------------
    # 2. FINAL HOLDOUT
    # -----------------------------------------------------

    print("\n===== PHASE 2: FINAL HOLDOUT =====")
    print("Evaluating fixed benchmarks and selected PCA configuration...")

    holdout_specs = [
        {
            "name": "Market core",
            "method": "market",
            "features": MARKET_CORE,
        },
        {
            "name": "Full features",
            "method": "full",
            "features": FULL_FEATURES,
        },
        {
            "name": "PCA-2",
            "method": "pca",
            "features": FULL_FEATURES,
            "components": 2,
        },
        {
            "name": selected_name,
            "method": "pca",
            "features": FULL_FEATURES,
            "components": selected_k,
        },
    ]

    # Avoid duplicate candidate rows if PCA-2 is selected.
    unique_specs = []
    seen = set()
    for spec in holdout_specs:
        if spec["name"] not in seen:
            unique_specs.append(spec)
            seen.add(spec["name"])

    holdout = forecast_range(
        df,
        development_end,
        n,
        unique_specs,
    )

    holdout_returns, holdout_directions = summarize_predictions(
        holdout, "final_holdout"
    )

    # -----------------------------------------------------
    # 3. SAVE FILES
    # -----------------------------------------------------

    dev.to_csv(
        OUTPUT_DIR / "controlled_validation_development_predictions.csv",
        index=False,
    )
    holdout.to_csv(
        OUTPUT_DIR / "controlled_validation_holdout_predictions.csv",
        index=False,
    )

    pd.concat([dev_returns, holdout_returns], ignore_index=True).to_csv(
        OUTPUT_DIR / "controlled_validation_return_metrics.csv",
        index=False,
    )

    pd.concat([dev_directions, holdout_directions], ignore_index=True).to_csv(
        OUTPUT_DIR / "controlled_validation_direction_metrics.csv",
        index=False,
    )

    pd.DataFrame([{
        "selected_pca_model": selected_name,
        "selected_components": selected_k,
        "selection_metric": "development Brier score",
        "initial_train_rows": initial_train_end,
        "development_end_row": development_end,
        "total_usable_rows": n,
        "development_start_date": str(df.iloc[initial_train_end]["date"].date()),
        "development_end_date": str(df.iloc[development_end - 1]["date"].date()),
        "holdout_start_date": str(df.iloc[development_end]["date"].date()),
        "holdout_end_date": str(df.iloc[-1]["date"].date()),
    }]).to_csv(
        OUTPUT_DIR / "controlled_validation_selection.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 4. PRINT RESULTS
    # -----------------------------------------------------

    print("\n===== DEVELOPMENT RETURN METRICS =====")
    print(
        dev_returns.sort_values("MAE_pct").to_string(
            index=False,
            formatters={
                "MAE_pct": "{:.4f}".format,
                "RMSE_pct": "{:.4f}".format,
                "R2": "{:.4f}".format,
            },
        )
    )

    print("\n===== DEVELOPMENT DIRECTION METRICS =====")
    print(
        dev_directions.sort_values(
            ["brier_score", "accuracy"],
            ascending=[True, False],
        ).to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "high_confidence_accuracy": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
            },
        )
    )

    print("\n===== FINAL HOLDOUT RETURN METRICS =====")
    print(
        holdout_returns.sort_values("MAE_pct").to_string(
            index=False,
            formatters={
                "MAE_pct": "{:.4f}".format,
                "RMSE_pct": "{:.4f}".format,
                "R2": "{:.4f}".format,
            },
        )
    )

    print("\n===== FINAL HOLDOUT DIRECTION METRICS =====")
    print(
        holdout_directions.sort_values(
            ["brier_score", "accuracy"],
            ascending=[True, False],
        ).to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "high_confidence_accuracy": (
                    lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
            },
        )
    )

    print("\n===== SAVED FILES =====")
    for filename in [
        "controlled_validation_development_predictions.csv",
        "controlled_validation_holdout_predictions.csv",
        "controlled_validation_return_metrics.csv",
        "controlled_validation_direction_metrics.csv",
        "controlled_validation_selection.csv",
    ]:
        print(OUTPUT_DIR / filename)

    print("\nInterpretation safeguards:")
    print("- Use development metrics for model selection.")
    print("- Use final holdout metrics for the primary performance assessment.")
    print("- PCA selection uses development Brier score, not holdout performance.")
    print("- The holdout is untouched by this script's model-selection step.")
    print("- The holdout is not fully independent of any prior manual exploration "
          "of this historical dataset.")


if __name__ == "__main__":
    main()
