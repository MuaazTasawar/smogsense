"""Small, honest hyperparameter search to reduce overfitting.

The baseline (n_estimators=300, learning_rate=0.05, max_depth=3,
subsample=1.0, min_samples_leaf=1) showed train RMSE 24.67 vs val
RMSE 50.41 -- a real train/val gap. This script searches over a small
grid of *more regularized* configurations (shallower trees, larger
leaves, row subsampling for stochastic GBM) evaluated against the
existing chronological val split -- no cross-validation shuffling,
since that would break time-series validity.

Every candidate is logged to reports/tuning_runs.csv (a dedicated
file, kept separate from reports/runs.csv -- that file already has a
fixed column schema from Phase 4's train.py, and this search's
candidates have a different shape; reusing it would silently corrupt
the existing log rather than error). The test set is never touched
here -- selection happens purely on val, and the winning config gets a
single, final test-set evaluation via evaluate.py afterward.

Run as:
    python -m src.tune
"""

import csv
import itertools
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.models.model import split_X_y
from src.utils.seed import set_seed

# Small, deliberately conservative grid -- every candidate is at least
# as regularized as the original baseline (max_depth<=3, subsample<=1,
# min_samples_leaf>=1), never less.
GRID = {
    "max_depth": [2, 3],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.7, 0.85, 1.0],
    "min_samples_leaf": [1, 5, 10],
}

TUNING_LOG_PATH = "reports/tuning_runs.csv"


def log_tuning_run(row: dict) -> None:
    """Append one tuning candidate's params + metrics to its own CSV.

    Kept separate from reports/runs.csv (see module docstring) so a
    differently-shaped row never silently corrupts that file.

    Args:
        row: Dict of param names and metric values for one candidate.
    """
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
    file_exists = os.path.exists(TUNING_LOG_PATH)
    os.makedirs(os.path.dirname(TUNING_LOG_PATH), exist_ok=True)
    with open(TUNING_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_candidate(params: dict, X_train, y_train, X_val, y_val, config: Config) -> dict:
    """Fit one candidate config on train and score it on val.

    Args:
        params: Dict of GradientBoostingRegressor kwargs for this
            candidate.
        X_train, y_train, X_val, y_val: Train/val features and target.
        config: Project config (for random_state).

    Returns:
        Dict with the candidate's params plus train/val RMSE/MAE and
        the train-val RMSE gap (the quantity we're trying to shrink).
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=config.random_state, **params)),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))

    return {
        **params,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "train_val_gap": val_rmse - train_rmse,
    }


def main() -> None:
    """Run the grid search, log every candidate, and print the winner."""
    config = Config()
    set_seed(config.random_state)

    train_df = pd.read_csv(f"{config.processed_dir}/train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(f"{config.processed_dir}/val.csv", index_col=0, parse_dates=True)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"Searching {len(combos)} candidate configs...")

    results = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        result = evaluate_candidate(params, X_train, y_train, X_val, y_val, config)
        results.append(result)
        log_tuning_run({"run_type": "tune_candidate", **result})
        print(f"[{i}/{len(combos)}] {params} -> val_rmse={result['val_rmse']:.2f}, gap={result['train_val_gap']:.2f}")

    results_df = pd.DataFrame(results)

    best = results_df.loc[results_df["val_rmse"].idxmin()]

    print("\n=== Best candidate (by val RMSE) ===")
    print(best.to_string())

    baseline_val_rmse = 50.41
    baseline_train_rmse = 24.67
    print(f"\nBaseline was: train_rmse={baseline_train_rmse}, val_rmse={baseline_val_rmse}, gap={baseline_val_rmse - baseline_train_rmse:.2f}")
    print(f"Best found:   train_rmse={best['train_rmse']:.2f}, val_rmse={best['val_rmse']:.2f}, gap={best['train_val_gap']:.2f}")


if __name__ == "__main__":
    main()