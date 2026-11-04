from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
INPUT_FILE = DATA / "model_dataset.csv"

REGRESSION_RESULTS = DATA / "model_comparison_regression.csv"
CLASSIFICATION_RESULTS = DATA / "model_comparison_classification.csv"
PREDICTIONS_FILE = DATA / "model_comparison_predictions.csv"

TARGET = "target_next_session_return_pct"
INITIAL_TRAIN_FRACTION = 0.70
RANDOM_STATE = 42
HIGH_CONFIDENCE_THRESHOLD = 0.60

MARKET_CORE = [
    "daily_return_pct",
    "return_lag1",
    "return_lag2",
    "return_lag5",
    "volatility_5",
]
EXTERNAL_MARKET = [
    "gold_return_pct",
    "usd_inr_return_pct",
    "brent_return_pct",
]
NEWS_VOLUME = ["article_count", "unique_sources"]
SENTIMENT = [
    "mean_sentiment",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
    "sentiment_lag1",
]
FEATURES = MARKET_CORE + EXTERNAL_MARKET + NEWS_VOLUME + SENTIMENT

# Models requested in the project instructions.
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM (SVR)": SVR(kernel="rbf", C=1.0, epsilon=0.1),
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "SVM (SVC)": SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}


def prepare_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {INPUT_FILE}\n"
            "Run src/build_model_dataset.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = {
        "date",
        "target_next_session_date",
        TARGET,
        "daily_return_pct",
        "gold_return_pct",
        "usd_inr_return_pct",
        "brent_return_pct",
        "article_count",
        "unique_sources",
        "mean_sentiment",
        "positive_fraction",
        "negative_fraction",
        "neutral_fraction",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Model dataset is missing columns: {sorted(missing)}")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in model dataset.")

    # Predictors are constructed using information available at the
    # forecast origin. Current-day return is known at that day's close.
    df["return_lag1"] = df["daily_return_pct"].shift(1)
    df["return_lag2"] = df["daily_return_pct"].shift(2)
    df["return_lag5"] = df["daily_return_pct"].shift(5)
    df["volatility_5"] = df["daily_return_pct"].rolling(5).std()
    df["sentiment_lag1"] = df["mean_sentiment"].shift(1)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def regression_pipeline(estimator, scale=True):
    steps = [SimpleImputer(strategy="median")]
    if scale:
        steps.append(StandardScaler())
    steps.append(estimator)
    return make_pipeline(*steps)


def classification_pipeline(estimator, scale=True):
    steps = [SimpleImputer(strategy="median")]
    if scale:
        steps.append(StandardScaler())
    steps.append(estimator)
    return make_pipeline(*steps)


def expanding_walk_forward(df):
    """Return chronological train/test indices; never shuffle."""
    start = int(len(df) * INITIAL_TRAIN_FRACTION)
    for test_idx in range(start, len(df)):
        yield np.arange(test_idx), test_idx


def run_regression(df):
    records = []
    model_specs = {
        name: (estimator, name not in {"Decision Tree", "Random Forest"})
        for name, estimator in REGRESSORS.items()
    }

    # Baseline: tomorrow's return is predicted to be zero.
    for _, test_idx in expanding_walk_forward(df):
        records.append({
            "model": "Tomorrow = today baseline",
            "date": df.iloc[test_idx]["date"],
            "target_next_session_date": df.iloc[test_idx][
                "target_next_session_date"
            ],
            "actual_return_pct": float(df.iloc[test_idx][TARGET]),
            "predicted_return_pct": 0.0,
        })

    for model_name, (estimator, scale) in model_specs.items():
        for train_idx, test_idx in expanding_walk_forward(df):
            train = df.iloc[train_idx]
            test = df.iloc[[test_idx]]
            y_train = train[TARGET].astype(float)

            model = regression_pipeline(estimator, scale=scale)
            model.fit(train[FEATURES], y_train)
            prediction = float(model.predict(test[FEATURES])[0])

            records.append({
                "model": model_name,
                "date": test.iloc[0]["date"],
                "target_next_session_date": test.iloc[0][
                    "target_next_session_date"
                ],
                "actual_return_pct": float(test.iloc[0][TARGET]),
                "predicted_return_pct": prediction,
            })

    predictions = pd.DataFrame(records)
    summaries = []

    for name, group in predictions.groupby("model", sort=False):
        actual = group["actual_return_pct"].to_numpy()
        predicted = group["predicted_return_pct"].to_numpy()

        summaries.append({
            "model": name,
            "forecasts": len(group),
            "MAE_pct": mean_absolute_error(actual, predicted),
            "RMSE_pct": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
        })

    results = (
        pd.DataFrame(summaries)
        .sort_values("MAE_pct")
        .reset_index(drop=True)
    )
    return predictions, results


def run_classification(df):
    records = []

    # Majority-class baseline.
    # At each forecast date, predict the most frequent class in the
    # training window. The test label is not used to form the prediction.
    for train_idx, test_idx in expanding_walk_forward(df):
        train = df.iloc[train_idx]
        test = df.iloc[[test_idx]]

        y_train = (train[TARGET].astype(float) > 0).astype(int)
        actual_up = int(float(test.iloc[0][TARGET]) > 0)

        # Probability of an up session is the training-window up fraction.
        p_up = float(y_train.mean())

        # Predict the majority class. In a tie, predict "up".
        majority_class = int(p_up >= 0.5)

        records.append({
            "model": "Majority-class baseline",
            "date": test.iloc[0]["date"],
            "target_next_session_date": test.iloc[0][
                "target_next_session_date"
            ],
            "actual_up": actual_up,
            "p_up": p_up,
            "predicted_up": majority_class,
            "confidence": max(p_up, 1 - p_up),
        })

    # Actual class is positive return = up; zero or negative = not-up.
    for model_name, estimator in CLASSIFIERS.items():
        for train_idx, test_idx in expanding_walk_forward(df):
            train = df.iloc[train_idx]
            test = df.iloc[[test_idx]]

            y_train = (train[TARGET].astype(float) > 0).astype(int)
            actual_up = int(float(test.iloc[0][TARGET]) > 0)

            # Some early windows may contain only one class.
            if y_train.nunique() < 2:
                p_up = float(y_train.mean())
            else:
                model = classification_pipeline(
                    estimator,
                    scale=model_name in {"Logistic Regression", "SVM (SVC)"},
                )
                model.fit(train[FEATURES], y_train)
                p_up = float(model.predict_proba(test[FEATURES])[0, 1])

            records.append({
                "model": model_name,
                "date": test.iloc[0]["date"],
                "target_next_session_date": test.iloc[0][
                    "target_next_session_date"
                ],
                "actual_up": actual_up,
                "p_up": p_up,
                "predicted_up": int(p_up >= 0.5),
                "confidence": max(p_up, 1 - p_up),
            })

    predictions = pd.DataFrame(records)
    summaries = []

    for name, group in predictions.groupby("model", sort=False):
        actual = group["actual_up"].astype(int)
        predicted = group["predicted_up"].astype(int)

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        high_conf = group["confidence"] >= HIGH_CONFIDENCE_THRESHOLD

        summaries.append({
            "model": name,
            "forecasts": len(group),
            "accuracy": accuracy_score(actual, predicted),
            "F1": f1_score(actual, predicted, zero_division=0),
            "brier_score": brier_score_loss(actual, group["p_up"]),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "high_confidence_forecasts": int(high_conf.sum()),
            "high_confidence_accuracy": (
                accuracy_score(actual[high_conf], predicted[high_conf])
                if high_conf.any()
                else np.nan
            ),
        })

    results = (
        pd.DataFrame(summaries)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )
    return predictions, results


def main():
    df = prepare_data()

    print("\n===== MODEL COMPARISON =====")
    print("Dataset:", INPUT_FILE)
    print("Rows with targets:", len(df))
    print("Initial training fraction:", f"{INITIAL_TRAIN_FRACTION:.0%}")
    print("Features:", len(FEATURES))
    print("Evaluation: expanding-window, chronological walk-forward")

    regression_predictions, regression_results = run_regression(df)
    classification_predictions, classification_results = run_classification(df)

    regression_results.to_csv(REGRESSION_RESULTS, index=False)
    classification_results.to_csv(CLASSIFICATION_RESULTS, index=False)

    # Combined file retains task-specific columns, with blank cells where
    # a metric does not apply.
    combined_predictions = pd.concat(
        [
            regression_predictions.assign(task="regression"),
            classification_predictions.assign(task="classification"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_predictions.to_csv(PREDICTIONS_FILE, index=False)

    print("\n===== REGRESSION (LOWER MAE/RMSE IS BETTER) =====")
    print(regression_results.round(4).to_string(index=False))

    print("\n===== CLASSIFICATION =====")
    print(classification_results.round(4).to_string(index=False))

    first = df.iloc[int(len(df) * INITIAL_TRAIN_FRACTION)]["date"]
    last = df.iloc[-1]["date"]

    print("\nEvaluation period:", first.date(), "to", last.date())
    print(
        "Forecasts per regression model:",
        len(
            regression_predictions[
                regression_predictions["model"] == "Tomorrow = today baseline"
            ]
        ),
    )
    print(
        "Forecasts per classification model:",
        len(classification_predictions)
        // (len(CLASSIFIERS) + 1),
    )

    print("\nSaved files:")
    print(REGRESSION_RESULTS)
    print(CLASSIFICATION_RESULTS)
    print(PREDICTIONS_FILE)


if __name__ == "__main__":
    main()