"""Validate existing volatility forecasts without refitting or overwriting files.
Run: python src/validate_volatility_experiment.py
Requires: data/processed/volatility_experiment_predictions.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'data' / 'processed'
SOURCE = DIR / 'volatility_experiment_predictions.csv'
DETAIL = DIR / 'volatility_validation_metrics.csv'
BOOT = DIR / 'volatility_validation_bootstrap.csv'
CAL = DIR / 'volatility_validation_calibration.csv'
TIME = DIR / 'volatility_validation_time_slices.csv'
REPS = 5000
BLOCK = 5
SEED = 42


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def summary(y, p, name, threshold=0.5):
    labels = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return dict(model=name, threshold=threshold, n=len(y), positives=int(y.sum()),
                predicted_positive=int(labels.sum()), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                accuracy=accuracy_score(y, labels),
                precision=precision_score(y, labels, zero_division=0),
                recall=recall_score(y, labels, zero_division=0),
                specificity=tn/(tn+fp) if tn+fp else np.nan,
                brier=brier_score_loss(y, p), roc_auc=auc(y, p))


def circular_indices(n, rng):
    count = int(np.ceil(n/BLOCK))
    starts = rng.integers(0, n, size=count)
    return np.concatenate([(s+np.arange(BLOCK)) % n for s in starts])[:n]


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing {SOURCE}. Run the previous experiment first.')
    df = pd.read_csv(SOURCE, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    cols = ['actual_volatile', 'volatility_model_probability', 'volatility_baseline_probability']
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df['date'].duplicated().any():
        raise ValueError('Duplicate forecast dates found; aborting.')
    if df[cols].isna().any().any():
        raise ValueError('Missing outcomes or probabilities; aborting.')
    y = df['actual_volatile'].to_numpy(dtype=int)
    p = df['volatility_model_probability'].to_numpy(dtype=float)
    b = df['volatility_baseline_probability'].to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0,1}) or not np.isfinite(p).all() or not np.isfinite(b).all():
        raise ValueError('Invalid binary labels or probabilities.')
    if (p.min() < 0 or p.max() > 1 or b.min() < 0 or b.max() > 1):
        raise ValueError('Probabilities must lie between 0 and 1.')
    if len(y) < 30 or len(np.unique(y)) != 2:
        raise ValueError('Need at least 30 observations and both outcome classes.')

    rows = [summary(y, b, 'historical_prevalence_baseline'),
            summary(y, p, 'logistic_volatility_model')]
    # Operational threshold diagnostics are exploratory, NOT tuned selection.
    for threshold in [0.20, 0.30, 0.40]:
        rows.append(summary(y, p, 'logistic_threshold_exploratory', threshold))
    pd.DataFrame(rows).to_csv(DETAIL, index=False)

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(REPS):
        idx = circular_indices(len(y), rng)
        yy, pp, bb = y[idx], p[idx], b[idx]
        if len(np.unique(yy)) < 2:
            continue
        draws.append((auc(yy, pp), auc(yy, pp) - auc(yy, bb),
                      brier_score_loss(yy, bb) - brier_score_loss(yy, pp)))
    samples = np.asarray(draws)
    observed = np.array([auc(y, p), auc(y, p) - auc(y, b),
                         brier_score_loss(y, b) - brier_score_loss(y, p)])
    names = ['model_roc_auc', 'auc_advantage_over_prevalence', 'brier_improvement_over_prevalence']
    boot_rows = []
    for j, name in enumerate(names):
        ci = np.quantile(samples[:,j], [0.025, 0.975])
        null = 0.5 if name == 'model_roc_auc' else 0.0
        # Two-sided percentile-bootstrap tail diagnostic; approximate, not a formal null test.
        tail = min(1.0, 2*min(np.mean(samples[:,j] <= null), np.mean(samples[:,j] >= null)))
        boot_rows.append(dict(metric=name, observed=observed[j], ci95_lower=ci[0],
                              ci95_upper=ci[1], approximate_two_sided_tail_fraction=tail,
                              bootstrap_reps=len(samples), block_length=BLOCK))
    pd.DataFrame(boot_rows).to_csv(BOOT, index=False)

    # Fixed bins avoid optimizing calibration bins using test outcomes.
    bins = pd.cut(p, bins=np.linspace(0,1,6), include_lowest=True)
    calibration = pd.DataFrame({'bin': bins, 'predicted': p, 'actual': y})
    calibration = calibration.groupby('bin', observed=False).agg(
        n=('actual','size'), mean_predicted_probability=('predicted','mean'),
        observed_volatile_fraction=('actual','mean')).reset_index()
    calibration.to_csv(CAL, index=False)

    # Non-overlapping chronological thirds, diagnostics only.
    time_rows = []
    for part, idx in enumerate(np.array_split(np.arange(len(df)), 3), start=1):
        yy, pp, bb = y[idx], p[idx], b[idx]
        time_rows.append(dict(period=part, start=str(df['date'].iloc[idx[0]].date()),
                              end=str(df['date'].iloc[idx[-1]].date()), n=len(idx),
                              actual_volatile_rate=float(yy.mean()),
                              model_auc=auc(yy, pp),
                              model_brier=brier_score_loss(yy, pp),
                              baseline_brier=brier_score_loss(yy, bb),
                              brier_improvement=brier_score_loss(yy, bb)-brier_score_loss(yy, pp)))
    pd.DataFrame(time_rows).to_csv(TIME, index=False)

    print('\n=== VOLATILITY VALIDATION (NO RETRAINING) ===')
    print(f'Forecast dates: {df.date.min().date()} to {df.date.max().date()} | N={len(df)} | Volatile={y.sum()}')
    print('\nClassification metrics (0.50 threshold unless noted):')
    print(pd.DataFrame(rows)[['model','threshold','predicted_positive','tp','fp','fn','precision','recall','accuracy','brier','roc_auc']].round(4).to_string(index=False))
    print('\nCircular block-bootstrap uncertainty (5-session blocks):')
    print(pd.DataFrame(boot_rows)[['metric','observed','ci95_lower','ci95_upper','approximate_two_sided_tail_fraction']].round(4).to_string(index=False))
    print('\nCalibration (empty bins mean no predictions in that range):')
    print(calibration.round(4).to_string(index=False))
    print('\nChronological thirds:')
    print(pd.DataFrame(time_rows).round(4).to_string(index=False))
    print('\nNew output files:')
    for path in [DETAIL, BOOT, CAL, TIME]:
        print(path)
    print('\nCAUTIONS:')
    print('Bootstrap CIs are approximate and depend on the 5-session block length.')
    print('Bootstrap tail fractions are descriptive, NOT formal p-values.')
    print('Threshold 0.20/0.30/0.40 results are exploratory; do not select one on this same test set.')
    print('These previously inspected forecasts are NOT an untouched holdout.')
    print('News timestamp availability at the forecast cutoff remains unverified.')

if __name__ == '__main__':
    main()
