"""Loads and validates the raw Lahore AQI dataset."""

import pandas as pd

from src.config import Config


def load_raw_data(config: Config) -> pd.DataFrame:
    """Load the raw Kaggle CSV into a clean, chronologically sorted frame.

    Parses the date column, sorts ascending, deduplicates by date
    (keeping the last occurrence if the export has repeats), and sets
    the date as the DataFrame index without inventing or dropping any
    rows.

    Args:
        config: Project config with `raw_data_path` and `date_column`.

    Returns:
        DataFrame indexed by date, sorted chronologically ascending,
        with all original columns intact.

    Raises:
        FileNotFoundError: If the raw CSV isn't at `config.raw_data_path`.
    """
    df = pd.read_csv(config.raw_data_path)
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    df = df.sort_values(config.date_column)
    df = df.drop_duplicates(subset=config.date_column, keep="last")
    df = df.set_index(config.date_column)
    return df


def audit_missing_data(df: pd.DataFrame, config: Config) -> dict:
    """Report missing values and date-coverage gaps in the raw series.

    This does not fill or drop anything — it only reports, so decisions
    about how to handle gaps stay explicit and visible in Phase 2's
    preprocessing step rather than happening silently here.

    Args:
        df: DataFrame indexed by date, as returned by `load_raw_data`.
        config: Project config with `target_column` and `weather_columns`.

    Returns:
        Dict with per-column null counts, total row count, expected row
        count for a gap-free daily series, and the number of missing
        calendar days.
    """
    cols = [config.target_column] + config.weather_columns
    null_counts = {c: int(df[c].isna().sum()) for c in cols}

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_days = len(full_range) - len(df.index.unique())

    return {
        "null_counts": null_counts,
        "row_count": len(df),
        "expected_daily_rows": len(full_range),
        "missing_calendar_days": missing_days,
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
    }