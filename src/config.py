from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Central config for the SmogSense forecasting pipeline.

    No hyperparameters or paths should live anywhere else in the
    codebase -- every script imports this dataclass instead of
    hardcoding values.
    """

    # Paths
    raw_data_path: str = "data/raw/lahore_aqi.csv"
    processed_dir: str = "data/processed"
    figures_dir: str = "reports/figures"
    runs_log_path: str = "reports/runs.csv"
    metrics_path: str = "reports/metrics.json"
    model_path: str = "reports/model.joblib"

    # Column names -- match the actual Kaggle export exactly.
    date_column: str = "date"
    target_column: str = "aqi_pm2.5"

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
    # are left as NaN and dropped rather than synthetically filled --
    # ~13.4% of rows have a null target value (see MODEL_CARD.md), so
    # this threshold is a real, stated modeling decision, not a detail.
    max_interpolation_gap_days: int = 3

    # Chronological split -- NEVER random-shuffle a time series.
    test_frac: float = 0.15
    val_frac: float = 0.15

    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14])

    # Model hyperparameters (GradientBoostingRegressor baseline).
    # Selected by src/tune.py's grid search over the chronological val
    # split: val RMSE 50.41 -> 46.54, gap 25.74 -> 13.66.
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.85
    min_samples_leaf: int = 10

    # Tail sample-weighting: trains with sample_weight=y_train (linear
    # upweighting by AQI magnitude). This is a genuine trade-off, not a
    # free improvement -- see src/tail_experiment.py's results: it
    # improves tail MAE (>250 AQI) and shrinks the underprediction bias
    # on high-AQI days, at the cost of worse overall RMSE/MAE. Enabled
    # by default because MODEL_CARD.md's Ethical Considerations section
    # already states a false "safe" prediction on a hazardous day is a
    # more costly error than a false alarm -- this setting is that
    # principle applied, not an arbitrary default. Set to False to
    # revert to the pure-accuracy-optimized baseline.
    use_tail_sample_weighting: bool = True

    random_state: int = 42