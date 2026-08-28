"""Exploratory analysis for the SmogSense AQI dataset.

Generates and saves every EDA figure to reports/figures/ and prints a
missing-data audit to the console. Run as a plain script:
    python notebooks/eda.py
"""

import json
import os
import sys

# Make the project root importable when this script is run directly
# (e.g. `python notebooks\eda.py`), since Python only puts the
# script's own directory on sys.path by default, not the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import Config
from src.data.load_data import audit_missing_data, load_raw_data


def plot_raw_series(df: pd.DataFrame, config: Config, out_path: str) -> None:
    """Save a line plot of the raw AQI target over the full date range."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[config.target_column], linewidth=0.8, color="#C0392B")
    ax.set_title("Lahore AQI (PM2.5) — Raw Series")
    ax.set_xlabel("Date")
    ax.set_ylabel(config.target_column)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_decomposition(df: pd.DataFrame, config: Config, out_path: str) -> None:
    """Save a trend/seasonal/residual decomposition of the AQI series.

    The raw series is reindexed to a continuous daily frequency and
    linearly interpolated ONLY for this decomposition plot — the
    interpolated version is never written back to processed data used
    for training; Phase 2 handles gaps explicitly and separately.
    """
    daily = df[config.target_column].asfreq("D")
    daily = daily.interpolate(method="linear")

    result = seasonal_decompose(daily, model="additive", period=365, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(result.observed, color="#2C3E50")
    axes[0].set_title("Observed")
    axes[1].plot(result.trend, color="#2980B9")
    axes[1].set_title("Trend")
    axes[2].plot(result.seasonal, color="#27AE60")
    axes[2].set_title("Seasonal")
    axes[3].plot(result.resid, color="#8E44AD")
    axes[3].set_title("Residual")
    fig.suptitle("Lahore AQI — Seasonal Decomposition (annual period)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, config: Config, out_path: str) -> None:
    """Save a correlation heatmap of AQI against the weather features."""
    cols = [config.target_column] + config.weather_columns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("AQI vs Weather Feature Correlation")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame, config: Config, out_path: str) -> None:
    """Save a histogram of the AQI target distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[config.target_column].dropna(), kde=True, ax=ax, color="#C0392B")
    ax.set_title("AQI (PM2.5) Distribution")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the full EDA pass and print the missing-data audit."""
    config = Config()
    os.makedirs(config.figures_dir, exist_ok=True)

    df = load_raw_data(config)
    audit = audit_missing_data(df, config)

    print("=== Missing Data Audit ===")
    print(json.dumps(audit, indent=2))

    plot_raw_series(df, config, f"{config.figures_dir}/eda_raw_series.png")
    plot_seasonal_decomposition(df, config, f"{config.figures_dir}/eda_seasonal_decompose.png")
    plot_correlation_heatmap(df, config, f"{config.figures_dir}/eda_correlation_heatmap.png")
    plot_target_distribution(df, config, f"{config.figures_dir}/eda_target_distribution.png")

    print(f"\nSaved 4 figures to {config.figures_dir}/")


if __name__ == "__main__":
    main()