"""Trains the GBM baseline and logs metrics + a feature-importance plot.

Run as:
    python -m src.train
"""

import os
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import Config
from src.models.model import build_pipeline, split_X_y
from src.utils.seed import set_seed


def plot_feature_importance(feature_names, importances, out_path: str, top_n: int = 15) -> None:
    """Save a horizontal bar chart of the top-N most important features."""
    order = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(np.array(feature_names)[order], np.array(importances)[order], color="#2980B9")
    ax.set_title(f"Top {top_n} Feature Importances -- GBM AQI Forecast")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_run(config: Config, metrics: dict) -> None:
    """Append one row of run metadata + metrics to reports/runs.csv.

    Uses pandas concat + rewrite rather than a raw CSV append. This
    project's run schema will keep evolving across phases (subsample
    and min_samples_leaf are new this phase, use_tail_sample_weighting
    arrives next phase) -- a naive csv.DictWriter append would silently
    misalign columns the moment a new run's dict has different keys
    than whatever the file's existing header happens to be. Reading +
    concatenating + rewriting with pandas instead means mismatched
    columns become NaN for the rows that don't have them, rather than
    corrupting the file.

    Args:
        config: Project config with runs_log_path.
        metrics: Dict of metric name -> value, plus any run metadata,
            to write as one CSV row.
    """
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
    os.makedirs(os.path.dirname(config.runs_log_path), exist_ok=True)

    new_row_df = pd.DataFrame([row])
    if os.path.exists(config.runs_log_path):
        existing_df = pd.read_csv(config.runs_log_path)
        combined = pd.concat([existing_df, new_row_df], ignore_index=True, sort=False)
    else:
        combined = new_row_df

    combined.to_csv(config.runs_log_path, index=False)


def main() -> None:
    """Fit the GBM pipeline on train, evaluate on val, save artifacts."""
    config = Config()
    set_seed(config.random_state)
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    train_df = pd.read_csv(f"{config.processed_dir}/train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(f"{config.processed_dir}/val.csv", index_col=0, parse_dates=True)

    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    train_mae = float(mean_absolute_error(y_train, train_pred))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))

    print(f"Train RMSE: {train_rmse:.2f}  MAE: {train_mae:.2f}")
    print(f"Val   RMSE: {val_rmse:.2f}  MAE: {val_mae:.2f}")

    joblib.dump(pipeline, config.model_path)
    print(f"Saved fitted pipeline to {config.model_path}")

    importances = pipeline.named_steps["model"].feature_importances_
    plot_feature_importance(
        X_train.columns,
        importances,
        f"{config.figures_dir}/train_feature_importance.png",
    )

    log_run(
        config,
        {
            "model": "GradientBoostingRegressor",
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "max_depth": config.max_depth,
            "subsample": config.subsample,
            "min_samples_leaf": config.min_samples_leaf,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
        },
    )
    print(f"Logged run to {config.runs_log_path}")


if __name__ == "__main__":
    main()