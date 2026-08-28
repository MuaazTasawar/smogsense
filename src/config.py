from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Central config for the SmogSense forecasting pipeline.

    No hyperparameters or paths should live anywhere else in the
    codebase — every script imports this dataclass instead of
    hardcoding values.
    """

    # Paths
    raw_data_path: str = "data/raw/lahore_aqi.csv"
    processed_dir: str = "data/processed"
    figures_dir: str = "reports/figures"
    runs_log_path: str = "reports/runs.csv"
    metrics_path: str = "reports/metrics.json"
    model_path: str = "reports/model.joblib"

    # Column names — adjust these three if the downloaded CSV's
    # headers differ from Kaggle's default export.
    date_column: str = "Date"
    target_column: str = "AQI"
    temperature_column: str = "Temperature"

    # Chronological split — NEVER random-shuffle a time series.
    # The most recent `test_frac` of the series is held out for
    # final evaluation, and the `val_frac` before that for tuning.
    test_frac: float = 0.15
    val_frac: float = 0.15

    # Feature engineering
    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14])

    # Model hyperparameters (GradientBoostingRegressor baseline)
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3

    random_state: int = 42