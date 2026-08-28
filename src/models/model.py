"""Model definition for next-day AQI forecasting.

Baseline is a scikit-learn Pipeline (StandardScaler + Gradient
Boosting Regressor) over lag/rolling-window/weather features — see
references/domain-specifics.md's guidance to start classical before
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
        df: DataFrame from `preprocess.py` with a `target` column and
            all other columns as features.

    Returns:
        Tuple of (X, y) — X keeps the original feature column names
        and order, y is the `target` Series.
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def build_pipeline(config: Config) -> Pipeline:
    """Build the scaler + Gradient Boosting Regressor pipeline.

    Args:
        config: Project config with model hyperparameters
            (`n_estimators`, `learning_rate`, `max_depth`,
            `random_state`).

    Returns:
        An unfit sklearn Pipeline. All features here are numeric
        (lags, rolling stats, weather readings), so no
        ColumnTransformer/OneHotEncoder is needed — a plain
        StandardScaler step is sufficient before the estimator.
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
                    random_state=config.random_state,
                ),
            ),
        ]
    )