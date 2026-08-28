"""Evaluates the trained pipeline on the held-out test set.

Saves RMSE/MAE to reports/metrics.json (never invents numbers — every
value here comes from an actual prediction pass) and three diagnostic
plots: actual-vs-predicted over the test window, residuals over time
(to distinguish regime shift from uniform overfitting), and a
predicted-vs-actual scatter with residual-vs-predicted panel.

Run as:
    python -m src.evaluate
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import Config
from src.models.model import split_X_y


def _reindex_continuous(dates, y_true, y_pred):
    """Reindex test predictions onto a continuous daily date range.

    The test set has gap days removed by preprocessing's dropna(), so
    plotting the raw (dates, values) arrays as a connected line draws
    a misleading straight segment across real data gaps. Reindexing
    onto a continuous daily range and leaving gaps as NaN makes
    matplotlib break the line at gaps instead of faking continuity.

    Args:
        dates: DatetimeIndex for the test set (may have gaps).
        y_true: Actual AQI values aligned with `dates`.
        y_pred: Predicted AQI values aligned with `dates`.

    Returns:
        Tuple of (full_index, y_true_series, y_pred_series), each
        Series reindexed onto `full_index` with NaN at gap dates.
    """
    full_index = pd.date_range(dates.min(), dates.max(), freq="D")
    y_true_series = pd.Series(np.array(y_true), index=dates).reindex(full_index)
    y_pred_series = pd.Series(np.array(y_pred), index=dates).reindex(full_index)
    return full_index, y_true_series, y_pred_series


def plot_actual_vs_predicted_over_time(dates, y_true, y_pred, out_path: str) -> None:
    """Save a line plot of actual vs predicted AQI across the test window.

    Reindexed onto a continuous daily range first so real data gaps
    show up as breaks in the line, not straight-line interpolation.

    Args:
        dates: DatetimeIndex for the test set, chronologically ordered.
        y_true: Actual AQI values.
        y_pred: Predicted AQI values.
        out_path: Destination PNG path.
    """
    full_index, y_true_series, y_pred_series = _reindex_continuous(dates, y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(full_index, y_true_series, label="Actual", color="#2C3E50", linewidth=1.2)
    ax.plot(full_index, y_pred_series, label="Predicted", color="#E67E22", linewidth=1.2, alpha=0.85)
    ax.set_title("Test Set — Actual vs Predicted Next-Day AQI")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI (PM2.5)")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_over_time(dates, y_true, y_pred, out_path: str) -> None:
    """Save residuals (actual - predicted) plotted across the test window.

    Reindexed onto a continuous daily range first (see
    `_reindex_continuous`) so gaps show as breaks, not a misleading
    straight line. A residual pattern clustered around specific dates
    suggests a regime the model struggles with (e.g. rapid AQI
    transitions); a uniformly elevated band suggests overfitting.

    Args:
        dates: DatetimeIndex for the test set, chronologically ordered.
        y_true: Actual AQI values.
        y_pred: Predicted AQI values.
        out_path: Destination PNG path.
    """
    full_index, y_true_series, y_pred_series = _reindex_continuous(dates, y_true, y_pred)
    residuals = y_true_series - y_pred_series
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(full_index, residuals, color="#8E44AD", linewidth=1.0)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Test Set — Residuals Over Time (Actual - Predicted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression_diagnostics(y_true, y_pred, out_path: str) -> None:
    """Save predicted-vs-actual and residual-vs-predicted plots side by side.

    Args:
        y_true: Actual AQI values.
        y_pred: Predicted AQI values.
        out_path: Destination PNG path.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(y_true, y_pred, alpha=0.5, color="#2980B9")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual")

    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, color="#C0392B")
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual vs Predicted")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: dict, out_path: str) -> None:
    """Persist computed metrics so downstream docs can cite real numbers.

    Args:
        metrics: Dict of metric name -> value.
        out_path: Destination JSON path.
    """
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    """Evaluate the saved pipeline on the test set and save all artifacts."""
    config = Config()
    os.makedirs(config.figures_dir, exist_ok=True)

    test_df = pd.read_csv(f"{config.processed_dir}/test.csv", index_col=0, parse_dates=True)
    X_test, y_test = split_X_y(test_df)

    pipeline = joblib.load(config.model_path)
    y_pred = pipeline.predict(X_test)

    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_mae = float(mean_absolute_error(y_test, y_pred))

    print(f"Test RMSE: {test_rmse:.2f}  MAE: {test_mae:.2f}")

    plot_actual_vs_predicted_over_time(
        test_df.index, y_test, y_pred, f"{config.figures_dir}/eval_actual_vs_predicted.png"
    )
    plot_residuals_over_time(
        test_df.index, y_test, y_pred, f"{config.figures_dir}/eval_residuals_over_time.png"
    )
    plot_regression_diagnostics(y_test, y_pred, f"{config.figures_dir}/eval_regression_diagnostics.png")

    save_metrics(
        {
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_rows": len(test_df),
            "test_date_range": [str(test_df.index.min().date()), str(test_df.index.max().date())],
        },
        config.metrics_path,
    )
    print(f"Saved metrics to {config.metrics_path} and 3 figures to {config.figures_dir}/")


if __name__ == "__main__":
    main()