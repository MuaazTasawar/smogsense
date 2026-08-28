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

    # Column names — match the actual Kaggle export exactly.
    date_column: str = "date"
    target_column: str = "aqi_pm2.5"

    # Weather feature columns used for forecasting. Using the
    # avg_* variants as the primary signal (min/max kept available
    # in raw data but not used as features by default — adding them
    # is a straightforward extension, noted in README Future Work).
    weather_columns: List[str] = field(
        default_factory=lambda: [
            "avg_temp_f",
            "avg_dew_point_f",
            "avg_humidity_percent",
            "avg_wind_speed_mph",
            "avg_pressure_in",
        ]
    )

    # Gap handling: after reindexing to a continuous daily series,
    # interpolate runs of missing days up to this length. Longer gaps
    # are left as NaN and dropped rather than synthetically filled —
    # ~21% of the target series is missing (see MODEL_CARD.md), so
    # this threshold is a real, stated modeling decision, not a detail.
    max_interpolation_gap_days: int = 3

    # Chronological split — NEVER random-shuffle a time series.
    # The most recent `test_frac` of the series is held out for
    # final evaluation, and the `val_frac` before that for tuning.
    test_frac: float = 0.15
    val_frac: float = 0.15

    # Feature engineering — lags/rolling windows on the target itself
    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14])

    # Model hyperparameters (GradientBoostingRegressor baseline)
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3

    random_state: int = 42