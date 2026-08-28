"""FastAPI serving for SmogSense next-day AQI forecasts.

Run with:
    uvicorn src.app.main:app --reload
Then POST daily history to /predict — see the HistoryRequest schema.
"""

from datetime import date
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

    date: date = Field(..., description="Calendar date of this record.")
    aqi_pm2_5: float = Field(..., alias="aqi_pm2.5", description="Observed AQI (PM2.5) for this date.")
    avg_temp_f: float
    avg_dew_point_f: float
    avg_humidity_percent: float
    avg_wind_speed_mph: float
    avg_pressure_in: float

    class Config:
        populate_by_name = True


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
    """Response: the forecast plus a naive uncertainty band."""

    as_of_date: str
    forecast_aqi: float
    uncertainty_band: Optional[dict]


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
        ForecastResponse with the as-of date, forecast AQI, and a
        naive uncertainty band derived from Phase 5's test RMSE.

    Raises:
        HTTPException: 400 if the model can't be loaded (Phase 4 not
            run yet) or the history is too short/gappy to build a
            complete feature row.
    """
    rows = [r.dict(by_alias=True) for r in request.records]
    history = pd.DataFrame(rows).set_index("date")
    history.index = pd.to_datetime(history.index)

    try:
        return predict_next_day(history, config)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not found — run Phase 4 (src/train.py) first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))