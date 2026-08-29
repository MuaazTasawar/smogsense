"""Trains the low/high quantile regressors for a calibrated uncertainty band.

Run as:
    python -m src.train_quantiles

Separate from train.py (which fits the point-forecast model) because
these have a different loss function (pinball, not squared error) and
are not sample-weighted. Reports empirical coverage on val: what
fraction of actual values actually fell inside the predicted
[low, high] band. A well-calibrated 80% interval (quantile_low=0.1,
quantile_high=0.9) should cover close to 80% of val rows -- reported
honestly, not assumed.
"""

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.models.model import build_quantile_pipeline, split_X_y
from src.utils.seed import set_seed


def main() -> None:
    """Fit both quantile models and report empirical val coverage."""
    config = Config()
    set_seed(config.random_state)

    train_df = pd.read_csv(f"{config.processed_dir}/train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(f"{config.processed_dir}/val.csv", index_col=0, parse_dates=True)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)

    low_pipeline = build_quantile_pipeline(config, alpha=config.quantile_low)
    low_pipeline.fit(X_train, y_train)
    joblib.dump(low_pipeline, config.quantile_low_model_path)

    high_pipeline = build_quantile_pipeline(config, alpha=config.quantile_high)
    high_pipeline.fit(X_train, y_train)
    joblib.dump(high_pipeline, config.quantile_high_model_path)

    low_pred = low_pipeline.predict(X_val)
    high_pred = high_pipeline.predict(X_val)

    crossed = np.sum(low_pred > high_pred)

    covered = np.sum((y_val.values >= low_pred) & (y_val.values <= high_pred))
    coverage = covered / len(y_val)
    target_coverage = config.quantile_high - config.quantile_low

    print(f"Saved quantile models to {config.quantile_low_model_path}, {config.quantile_high_model_path}")
    print(f"Target coverage: {target_coverage:.0%} (quantile_low={config.quantile_low}, quantile_high={config.quantile_high})")
    print(f"Empirical val coverage: {coverage:.1%} ({covered}/{len(y_val)} rows fell inside [low, high])")
    print(f"Rows where low > high (crossing): {crossed}")
    print(f"Mean band width: {np.mean(high_pred - low_pred):.1f} AQI points")


if __name__ == "__main__":
    main()