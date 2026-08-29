"""Next-day AQI inference from a short window of recent daily history.

Reuses build_feature_columns from preprocess.py so the feature vector
at inference time is guaranteed identical to what the model was
trained on.

Run as a demo against the tail of the raw dataset:
    python -m src.predict
"""

import json
import os
from typing import Optional

import joblib
import pandas as pd

from src.config import Config
from src.data.preprocess import build_feature_columns, reindex_and_interpolate


def _load_test_rmse(config: Config) -> Optional[float]:
    """Load the test RMSE from metrics.json, used only as a fallback band."""
    try:
        with open(config.metrics_path) as f:
            return json.load(f)["test_rmse"]
    except (FileNotFoundError, KeyError):
        return None


def _quantile_models_available(config: Config) -> bool:
    """Check whether both quantile models have been trained and saved."""
    return os.path.exists(config.quantile_low_model_path) and os.path.exists(config.quantile_high_model_path)


def predict_next_day(history: pd.DataFrame, config: Config) -> dict:
    """Forecast next-day AQI from a recent daily-history DataFrame.

    Args:
        history: DataFrame indexed by date (ascending, most recent
            date last) with columns config.target_column and all of
            config.weather_columns. Must cover at least
            max(lag_days + rolling_windows) consecutive days ending
            on the most recent date.
        config: Project config.

    Returns:
        Dict with as_of_date, forecast_aqi, uncertainty_band, and
        band_type ("quantile" if the trained quantile models were
        used, "naive_rmse" as a fallback if they haven't been trained
        yet -- see MODEL_CARD.md for the quantile band's real,
        measured coverage: 73.4% empirical against an 80% target).

    Raises:
        ValueError: If the built feature row contains NaN, meaning
            history doesn't cover enough consecutive days.
    """
    daily = reindex_and_interpolate(history, config)
    features = build_feature_columns(daily, config)
    latest_row = features.iloc[[-1]]

    if latest_row.isna().any(axis=None):
        missing = latest_row.columns[latest_row.isna().iloc[0]].tolist()
        raise ValueError(
            f"Insufficient history to build a full feature row -- missing: {missing}. "
            f"Provide at least {max(config.lag_days + config.rolling_windows)} consecutive days."
        )

    pipeline = joblib.load(config.model_path)
    forecast = float(pipeline.predict(latest_row)[0])

    if _quantile_models_available(config):
        low_pipeline = joblib.load(config.quantile_low_model_path)
        high_pipeline = joblib.load(config.quantile_high_model_path)
        low = float(low_pipeline.predict(latest_row)[0])
        high = float(high_pipeline.predict(latest_row)[0])
        low, high = min(low, high), max(low, high)
        band = {"low": round(low, 1), "high": round(high, 1)}
        band_type = "quantile"
    else:
        test_rmse = _load_test_rmse(config)
        band = (
            {"low": round(forecast - test_rmse, 1), "high": round(forecast + test_rmse, 1)}
            if test_rmse is not None
            else None
        )
        band_type = "naive_rmse" if band is not None else None

    return {
        "as_of_date": str(daily.index[-1].date()),
        "forecast_aqi": round(forecast, 1),
        "uncertainty_band": band,
        "band_type": band_type,
    }


def main() -> None:
    """Demo: forecast using the last ~20 rows of the raw dataset."""
    config = Config()
    from src.data.load_data import load_raw_data

    raw = load_raw_data(config)
    recent_history = raw.tail(20)

    result = predict_next_day(recent_history, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()