"""Model definitions for next-day AQI forecasting.

Two model types are built here:
  - build_pipeline: the point forecast (StandardScaler + GBM), trained
    on squared error, optionally sample-weighted for tail bias.
  - build_quantile_pipeline: low/high quantile models for a real,
    calibrated uncertainty band, trained on pinball loss instead.

references/domain-specifics.md's guidance: start classical before
reaching for an LSTM on a dataset this size (~1,285 usable rows).
"""

from typing import Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a processed frame into features (X) and target (y).

    Args:
        df: DataFrame from preprocess.py with a target column and
            all other columns as features.

    Returns:
        Tuple of (X, y).
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def build_pipeline(config: Config) -> Pipeline:
    """Build the scaler + Gradient Boosting Regressor pipeline.

    Args:
        config: Project config with model hyperparameters
            (n_estimators, learning_rate, max_depth, subsample,
            min_samples_leaf, random_state).

    Returns:
        An unfit sklearn Pipeline.
    """
    return Pipeline(
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


def build_quantile_pipeline(config: Config, alpha: float) -> Pipeline:
    """Build a quantile-regression pipeline for a calibrated uncertainty band.

    Unlike the point-forecast model (trained on squared error), this
    minimizes pinball loss at the given quantile -- e.g. alpha=0.1
    trains a model whose predictions are exceeded by the true value
    only ~10% of the time. Training a low and high quantile model
    (e.g. 0.1 and 0.9) gives a real, data-derived 80% interval, instead
    of the naive forecast +/- test_RMSE heuristic used previously.

    Uses the same tree structure as the point-forecast model, but is
    NOT sample-weighted -- tail sample-weighting was tuned for the
    point forecast's accuracy/bias trade-off specifically, and
    reapplying it here would conflate two different fixes.

    Args:
        config: Project config with tree hyperparameters.
        alpha: The quantile to fit (e.g. 0.1 for the 10th percentile).

    Returns:
        An unfit sklearn Pipeline using quantile loss.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
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