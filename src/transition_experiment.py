"""Tests whether rate-of-change features fix the transition-lag limitation.

The diagnostics (eval_actual_vs_predicted.png, eval_residuals_over_time.png)
showed the model reacts slowly during rapid AQI swings (smog onset),
consistent with it leaning heavily on aqi_lag_1 and aqi_roll_mean_7 --
both of which describe "recent level," not "recent direction of
change."

This adds four momentum/deviation features and tests, honestly, on
val:
  - aqi_delta_1: day-over-day change (lag_1 - lag_2)
  - aqi_delta_3: 3-day momentum (lag_1 - lag_3)
  - aqi_accel: change in the rate of change (2nd derivative)
  - aqi_deviation_from_roll_mean_7: how far the latest value has
    already strayed from its recent baseline -- a direct "is a spike
    happening right now" signal

Specifically checks error on HIGH-VOLATILITY days (where actual AQI
moved a lot from the prior day) -- the exact regime the plots flagged
as weak -- not just overall RMSE, which could hide a fix that only
helps in the volatile subset.

Run as:
    python -m src.transition_experiment
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import Config
from src.data.load_data import load_raw_data
from src.data.preprocess import chronological_split, reindex_and_interpolate
from src.models.model import build_pipeline, split_X_y
from src.utils.seed import set_seed

VOLATILITY_THRESHOLD = 40  # AQI points of day-over-day change to call "high volatility"


def build_baseline_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Rebuild the current (baseline) feature set -- mirrors preprocess.py."""
    out = pd.DataFrame(index=df.index)
    for lag in config.lag_days:
        out[f"aqi_lag_{lag}"] = df[config.target_column].shift(lag)
    for window in config.rolling_windows:
        shifted = df[config.target_column].shift(1)
        out[f"aqi_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"aqi_roll_std_{window}"] = shifted.rolling(window).std()
    for col in config.weather_columns:
        out[col] = df[col]
    return out


def build_momentum_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Baseline features plus rate-of-change / momentum features."""
    out = build_baseline_features(df, config)
    lag1 = df[config.target_column].shift(1)
    lag2 = df[config.target_column].shift(2)
    lag3 = df[config.target_column].shift(3)
    roll_mean_7 = df[config.target_column].shift(1).rolling(7).mean()

    out["aqi_delta_1"] = lag1 - lag2
    out["aqi_delta_3"] = lag1 - lag3
    out["aqi_accel"] = (lag1 - lag2) - (lag2 - lag3)
    out["aqi_deviation_from_roll_mean_7"] = lag1 - roll_mean_7
    return out


def make_supervised(feature_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach next-day target and drop incomplete rows."""
    out = feature_df.copy()
    out["target"] = df["aqi_pm2.5"].shift(-1)
    return out.dropna()


def evaluate_on_val(train_df, val_df, config: Config) -> dict:
    """Fit the current tuned pipeline and score it, overall + on volatile days."""
    set_seed(config.random_state)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)

    pipeline = build_pipeline(config)
    sample_weight = y_train.values if config.use_tail_sample_weighting else None
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    pred = pipeline.predict(X_val)

    overall_rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    overall_mae = float(mean_absolute_error(y_val, pred))

    if "aqi_lag_1" in val_df.columns:
        day_over_day_change = (y_val - val_df["aqi_lag_1"]).abs()
        volatile_mask = day_over_day_change > VOLATILITY_THRESHOLD
        n_volatile = int(volatile_mask.sum())
        volatile_mae = (
            float(mean_absolute_error(y_val[volatile_mask], pred[volatile_mask])) if n_volatile > 0 else None
        )
    else:
        n_volatile, volatile_mae = None, None

    return {
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "n_volatile_rows": n_volatile,
        "volatile_mae": volatile_mae,
    }


def main() -> None:
    """Compare baseline vs momentum-augmented features, report honestly."""
    config = Config()
    raw = load_raw_data(config)
    daily = reindex_and_interpolate(raw, config)

    print("=== Variant A: current baseline features ===")
    baseline_features = make_supervised(build_baseline_features(daily, config), daily)
    train_a, val_a, _ = chronological_split(baseline_features, config)
    result_a = evaluate_on_val(train_a, val_a, config)
    for k, v in result_a.items():
        print(f"  {k}: {v}")

    print("\n=== Variant B: baseline + momentum/deviation features ===")
    momentum_features = make_supervised(build_momentum_features(daily, config), daily)
    train_b, val_b, _ = chronological_split(momentum_features, config)
    result_b = evaluate_on_val(train_b, val_b, config)
    for k, v in result_b.items():
        print(f"  {k}: {v}")

    print("\n=== Comparison ===")
    print(f"Overall val RMSE:  {result_a['overall_rmse']:.2f} -> {result_b['overall_rmse']:.2f}")
    print(f"Overall val MAE:   {result_a['overall_mae']:.2f} -> {result_b['overall_mae']:.2f}")
    print(f"Volatile-day MAE:  {result_a['volatile_mae']:.2f} -> {result_b['volatile_mae']:.2f}  (n={result_a['n_volatile_rows']})")


if __name__ == "__main__":
    main()