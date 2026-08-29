"""FastAPI serving for SmogSense next-day AQI forecasts.

Run with:
    uvicorn src.app.main:app --reload
Then POST daily history to /predict — see the HistoryRequest schema.
"""

from datetime import date as date_type
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import Config
from src.predict import predict_next_day

app = FastAPI(
    title="SmogSense API",
    description="Next-day AQI (PM2.5) forecasting for Lahore from recent daily history.",
    version="0.1.0",
)

config = Config()


class DailyRecord(BaseModel):
    """One day of AQI + weather history, matching the training schema."""

    model_config = ConfigDict(populate_by_name=True)

    record_date: date_type = Field(..., alias="date", description="Calendar date of this record.")
    aqi_pm2_5: float = Field(..., alias="aqi_pm2.5", description="Observed AQI (PM2.5) for this date.")
    avg_temp_f: float
    avg_dew_point_f: float
    avg_humidity_percent: float
    avg_wind_speed_mph: float
    avg_pressure_in: float


class HistoryRequest(BaseModel):
    """Request body: a chronologically ordered window of recent history."""

    records: List[DailyRecord] = Field(
        ...,
        description=(
            "Daily records in ascending date order, most recent last. "
            "Must cover at least max(lag_days + rolling_windows) "
            "consecutive days ending on the forecast 'as of' date, with "
            "no gaps longer than the model's interpolation tolerance."
        ),
    )


class ForecastResponse(BaseModel):
    """Response: the forecast plus an uncertainty band.

    band_type is "quantile" once src/train_quantiles.py has been run
    (real, asymmetric, data-derived interval — 73.4% empirical coverage
    against an 80% target, see MODEL_CARD.md), or "naive_rmse" as a
    fallback if the quantile models haven't been trained yet.
    """

    as_of_date: str
    forecast_aqi: float
    uncertainty_band: Optional[dict]
    band_type: Optional[str]


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: HistoryRequest) -> dict:
    """Forecast next-day AQI from a window of recent daily history.

    Args:
        request: HistoryRequest with a `records` list of DailyRecord.

    Returns:
        ForecastResponse with the as-of date, forecast AQI, and an
        uncertainty band (see ForecastResponse docstring for band_type).

    Raises:
        HTTPException: 400 if the model can't be loaded or the history
            is too short/gappy to build a complete feature row.
    """
    rows = [r.model_dump(by_alias=False) for r in request.records]
    df = pd.DataFrame(rows).rename(columns={"record_date": "date", "aqi_pm2_5": config.target_column})
    history = df.set_index("date")
    history.index = pd.to_datetime(history.index)

    try:
        return predict_next_day(history, config)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not found — run src/train.py first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))