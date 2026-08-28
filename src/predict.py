"""Next-day AQI inference from a short window of recent daily history.

Reuses `build_feature_columns` from preprocess.py so the feature
vector at inference time is guaranteed identical in definition and
order to what the model was trained on — no separate/duplicated
feature logic to drift out of sync.

Run as a demo against the tail of the processed test set:
    python -m src.predict
"""

import json
from typing import Optional

import joblib
import pandas as pd

from src.config import Config
from src.data.preprocess import build_feature_columns, reindex_and_interpolate


def _load_test_rmse(config: Config) -> Optional[float]:
    """Load the test RMSE from metrics.json for a naive uncertainty band.

    Args:
        config: Project config with `metrics_path`.

    Returns:
        The test RMSE as a float, or None if metrics.json doesn't
        exist yet (Phase 5 hasn't been run).
    """
    try:
        with open(config.metrics_path) as f:
            return json.load(f)["test_rmse"]
    except (FileNotFoundError, KeyError):
        return None


def predict_next_day(history: pd.DataFrame, config: Config) -> dict:
    """Forecast next-day AQI from a recent daily-history DataFrame.

    Args:
        history: DataFrame indexed by date (ascending, most recent
            date last) with columns `config.target_column` and all of
            `config.weather_columns`. Must cover at least
            `max(lag_days + rolling_windows)` consecutive days ending
            on the most recent date, or the feature row will contain
            NaN and this raises a ValueError.
        config: Project config.

    Returns:
        Dict with `as_of_date` (the last date in history — the
        forecast is for the day after this), `forecast_aqi`, and a
        naive `uncertainty_band` (forecast ± test RMSE, a rough
        symmetric band from the Phase 5 test-set error — NOT a
        calibrated prediction interval; see MODEL_CARD.md).

    Raises:
        ValueError: If the built feature row contains NaN, meaning
            `history` doesn't cover enough consecutive days.
    """
    daily = reindex_and_interpolate(history, config)
    features = build_feature_columns(daily, config)
    latest_row = features.iloc[[-1]]

    if latest_row.isna().any(axis=None):
        missing = latest_row.columns[latest_row.isna().iloc[0]].tolist()
        raise ValueError(
            f"Insufficient history to build a full feature row — missing: {missing}. "
            f"Provide at least {max(config.lag_days + config.rolling_windows)} consecutive days."
        )

    pipeline = joblib.load(config.model_path)
    forecast = float(pipeline.predict(latest_row)[0])

    test_rmse = _load_test_rmse(config)
    band = (
        {"low": round(forecast - test_rmse, 1), "high": round(forecast + test_rmse, 1)}
        if test_rmse is not None
        else None
    )

    return {
        "as_of_date": str(daily.index[-1].date()),
        "forecast_aqi": round(forecast, 1),
        "uncertainty_band": band,
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