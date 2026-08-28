"""Sanity tests for the SmogSense pipeline.

These are not model-quality tests (that's what Phase 5's real
evaluation metrics are for) — they catch silent pipeline breakage:
wrong shapes, NaN leaking through where it shouldn't, chronological
ordering violations, and feature-vector drift between training and
inference.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.preprocess import build_feature_columns, build_features, chronological_split, reindex_and_interpolate
from src.models.model import build_pipeline, split_X_y
from src.utils.seed import set_seed


@pytest.fixture
def config():
    """A Config instance using default values."""
    return Config()


@pytest.fixture
def synthetic_raw(config):
    """A small synthetic daily AQI+weather series with one deliberate gap.

    Covers 40 days so lag/rolling windows (max 14) have enough history
    to produce non-NaN feature rows well before the series ends, and
    includes a 2-day gap to exercise the interpolation/gap-drop path
    without relying on the real (network-fetched) dataset.
    """
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    rng = np.random.default_rng(config.random_state)
    df = pd.DataFrame(
        {
            config.target_column: rng.uniform(50, 300, size=40),
            "avg_temp_f": rng.uniform(40, 90, size=40),
            "avg_dew_point_f": rng.uniform(20, 60, size=40),
            "avg_humidity_percent": rng.uniform(20, 80, size=40),
            "avg_wind_speed_mph": rng.uniform(2, 15, size=40),
            "avg_pressure_in": rng.uniform(29, 31, size=40),
        },
        index=dates,
    )
    df = df.drop(df.index[20:22])
    return df


def test_reindex_and_interpolate_fills_short_gaps(synthetic_raw, config):
    """A gap within max_interpolation_gap_days should be filled, not NaN."""
    daily = reindex_and_interpolate(synthetic_raw, config)
    assert len(daily) == 40
    assert daily[config.target_column].isna().sum() == 0


def test_build_feature_columns_no_leakage(synthetic_raw, config):
    """Feature columns must never equal the same-day target exactly."""
    daily = reindex_and_interpolate(synthetic_raw, config)
    features = build_feature_columns(daily, config)
    for col in features.columns:
        if col in config.weather_columns:
            continue
        aligned = features[col].dropna()
        target_aligned = daily[config.target_column].reindex(aligned.index)
        assert not aligned.equals(target_aligned), f"{col} suspiciously matches same-day target"


def test_build_features_no_nan_after_dropna(synthetic_raw, config):
    """The final training frame must contain zero NaN anywhere."""
    daily = reindex_and_interpolate(synthetic_raw, config)
    features = build_features(daily, config)
    assert features.isna().sum().sum() == 0
    assert len(features) > 0


def test_chronological_split_no_overlap_and_ordered(synthetic_raw, config):
    """Train/val/test splits must be time-ordered with zero index overlap."""
    daily = reindex_and_interpolate(synthetic_raw, config)
    features = build_features(daily, config)
    train_df, val_df, test_df = chronological_split(features, config)

    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

    all_idx = list(train_df.index) + list(val_df.index) + list(test_df.index)
    assert len(all_idx) == len(set(all_idx)), "overlapping dates across splits"


def test_model_forward_pass_shape(synthetic_raw, config):
    """A fitted pipeline must predict one value per input row."""
    set_seed(config.random_state)
    daily = reindex_and_interpolate(synthetic_raw, config)
    features = build_features(daily, config)
    X, y = split_X_y(features)

    pipeline = build_pipeline(config)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert preds.shape[0] == X.shape[0]
    assert np.all(np.isfinite(preds))


def test_inference_features_match_training_columns(synthetic_raw, config):
    """predict.py's feature builder must produce the exact training columns."""
    daily = reindex_and_interpolate(synthetic_raw, config)
    train_features = build_features(daily, config).drop(columns=["target"])
    inference_features = build_feature_columns(daily, config).dropna()

    assert list(train_features.columns) == list(inference_features.columns)