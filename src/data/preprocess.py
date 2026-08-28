"""Feature engineering and chronological train/val/test splitting.

Turns the raw, gap-ridden daily AQI series into a supervised
next-day-forecast dataset: features are built from information
available "as of" day t (lagged AQI, rolling AQI stats, and day-t
weather), and the target is AQI at day t+1.

Gap handling is explicit and bounded: short runs of missing days
(up to `config.max_interpolation_gap_days`) are linearly interpolated;
longer gaps are left as NaN and the resulting incomplete rows are
dropped by `dropna()` rather than synthetically filled.
"""

import os

import pandas as pd

from src.config import Config


def reindex_and_interpolate(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Reindex to a continuous daily series and interpolate short gaps.

    Args:
        df: DataFrame indexed by date, as returned by `load_raw_data`.
        config: Project config with `target_column`, `weather_columns`,
            and `max_interpolation_gap_days`.

    Returns:
        A continuous-daily-index DataFrame where gaps up to
        `max_interpolation_gap_days` are linearly interpolated; longer
        gaps remain NaN and are left for `dropna()` downstream.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = df.reindex(full_index)

    cols = [config.target_column] + config.weather_columns
    for col in cols:
        daily[col] = daily[col].interpolate(
            method="linear",
            limit=config.max_interpolation_gap_days,
            limit_area="inside",
        )
    return daily


def build_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build the supervised next-day-forecast feature/target frame.

    Features use information available as of day t: lagged AQI values,
    rolling AQI statistics computed over the window ending at day t
    (via `.shift(1).rolling(...)`, so no same-day leakage), and day-t
    weather readings. The target is AQI at day t+1.

    Note: using day-t *actual* weather as a feature is a stand-in for
    a day-t weather *forecast* in a real deployment — this dataset
    only has observed weather, not forecasts, which is called out as
    a limitation in MODEL_CARD.md.

    Args:
        df: Continuous-daily-index DataFrame from `reindex_and_interpolate`.
        config: Project config with `target_column`, `weather_columns`,
            `lag_days`, `rolling_windows`.

    Returns:
        DataFrame with feature columns, a `target` column (next-day
        AQI), and rows with any NaN (insufficient lag history, the
        final undated row, or an unfilled long gap) dropped.
    """
    out = pd.DataFrame(index=df.index)

    for lag in config.lag_days:
        out[f"aqi_lag_{lag}"] = df[config.target_column].shift(lag)

    for window in config.rolling_windows:
        shifted = df[config.target_column].shift(1)
        out[f"aqi_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"aqi_roll_std_{window}"] = shifted.rolling(window).std()

    for col in config.weather_columns:
        out[col] = df[col]

    out["target"] = df[config.target_column].shift(-1)

    before = len(out)
    out = out.dropna()
    after = len(out)
    print(f"build_features: dropped {before - after} rows with incomplete history/gaps, {after} usable rows remain")

    return out


def chronological_split(features: pd.DataFrame, config: Config):
    """Split a feature frame into train/val/test by time, not randomly.

    The most recent `test_frac` of rows become the test set, the
    `val_frac` before that becomes validation, and everything earlier
    is training. This mirrors how the model would actually be used:
    trained on the past, evaluated on the most recent unseen future.

    Args:
        features: Feature/target DataFrame from `build_features`,
            already in chronological order (its index is dates).
        config: Project config with `test_frac`, `val_frac`.

    Returns:
        Tuple of (train_df, val_df, test_df), each still containing
        the `target` column.
    """
    n = len(features)
    n_test = int(n * config.test_frac)
    n_val = int(n * config.val_frac)
    n_train = n - n_test - n_val

    train_df = features.iloc[:n_train]
    val_df = features.iloc[n_train:n_train + n_val]
    test_df = features.iloc[n_train + n_val:]

    return train_df, val_df, test_df


def preprocess_and_split(config: Config):
    """Run the full preprocessing pipeline and save train/val/test CSVs.

    Args:
        config: Project config.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames, each with
        feature columns plus a `target` column.
    """
    from src.data.load_data import load_raw_data

    raw = load_raw_data(config)
    daily = reindex_and_interpolate(raw, config)
    features = build_features(daily, config)
    train_df, val_df, test_df = chronological_split(features, config)

    os.makedirs(config.processed_dir, exist_ok=True)
    train_df.to_csv(f"{config.processed_dir}/train.csv")
    val_df.to_csv(f"{config.processed_dir}/val.csv")
    test_df.to_csv(f"{config.processed_dir}/test.csv")

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Train range: {train_df.index.min().date()} to {train_df.index.max().date()}")
    print(f"Val range:   {val_df.index.min().date()} to {val_df.index.max().date()}")
    print(f"Test range:  {test_df.index.min().date()} to {test_df.index.max().date()}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    preprocess_and_split(Config())