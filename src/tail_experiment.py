"""Tests whether sample-weighting high-AQI rows reduces tail underprediction.

The evaluation diagnostics (eval_regression_diagnostics.png) showed the
model systematically underpredicts actual AQI above ~300 -- the target
distribution is right-skewed, so an MSE-trained model naturally
optimizes for the dense 100-200 range at the tail's expense.

This script compares two variants on val, honestly:
  A) unweighted (current tuned baseline)
  B) sample_weight = y_train (linear upweighting by AQI magnitude)

Overall val RMSE/MAE alone can hide a tail-specific fix (or tail-
specific harm), so this also reports MAE restricted to rows where
actual AQI > 250 (the "tail subset") -- that's the number that answers
the actual question, not the aggregate one.

Run as:
    python -m src.tail_experiment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.models.model import split_X_y
from src.utils.seed import set_seed

TAIL_THRESHOLD = 250  # AQI value above which we call it "the tail"


def fit_and_score(X_train, y_train, X_val, y_val, config: Config, sample_weight=None) -> dict:
    """Fit one variant and score it overall + on the tail subset."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    max_depth=config.max_depth,
                    subsample=config.subsample,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    val_pred = pipeline.predict(X_val)

    overall_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    overall_mae = float(mean_absolute_error(y_val, val_pred))

    tail_mask = y_val > TAIL_THRESHOLD
    n_tail = int(tail_mask.sum())
    if n_tail > 0:
        tail_mae = float(mean_absolute_error(y_val[tail_mask], val_pred[tail_mask]))
        tail_mean_signed_error = float(np.mean(val_pred[tail_mask] - y_val[tail_mask]))
    else:
        tail_mae, tail_mean_signed_error = None, None

    return {
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "n_tail_rows": n_tail,
        "tail_mae": tail_mae,
        "tail_mean_signed_error": tail_mean_signed_error,
    }


def main() -> None:
    """Compare unweighted vs sample-weighted training, report honestly."""
    config = Config()
    set_seed(config.random_state)

    train_df = pd.read_csv(f"{config.processed_dir}/train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(f"{config.processed_dir}/val.csv", index_col=0, parse_dates=True)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)

    print(f"Tail subset (val, actual AQI > {TAIL_THRESHOLD}): {(y_val > TAIL_THRESHOLD).sum()} of {len(y_val)} rows")

    print("\n=== Variant A: unweighted (current baseline) ===")
    result_a = fit_and_score(X_train, y_train, X_val, y_val, config, sample_weight=None)
    for k, v in result_a.items():
        print(f"  {k}: {v}")

    print("\n=== Variant B: sample_weight = y_train (linear upweight by AQI) ===")
    result_b = fit_and_score(X_train, y_train, X_val, y_val, config, sample_weight=y_train.values)
    for k, v in result_b.items():
        print(f"  {k}: {v}")

    print("\n=== Comparison ===")
    print(f"Overall val RMSE:      {result_a['overall_rmse']:.2f} -> {result_b['overall_rmse']:.2f}")
    print(f"Overall val MAE:       {result_a['overall_mae']:.2f} -> {result_b['overall_mae']:.2f}")
    print(f"Tail MAE (>250):       {result_a['tail_mae']:.2f} -> {result_b['tail_mae']:.2f}")
    print(f"Tail mean signed err:  {result_a['tail_mean_signed_error']:.2f} -> {result_b['tail_mean_signed_error']:.2f}  (closer to 0 = less biased)")


if __name__ == "__main__":
    main()