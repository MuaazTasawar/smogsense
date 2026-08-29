"""Direct, apples-to-apples comparison of the GBM and LSTM on the same test set.

Both models see the same held-out test period. GBM uses the
row-based feature pipeline (preprocess.py); LSTM uses raw sequences
(train_lstm.py's build_sequences). The two pipelines drop slightly
different rows near split boundaries (different lag/window
requirements), so the exact test row counts won't match exactly --
this is noted in the output rather than hidden.

Run as:
    python -m src.compare_models
"""

import numpy as np
import pandas as pd
import joblib
import torch

from src.config import Config
from src.data.load_data import load_raw_data
from src.data.preprocess import reindex_and_interpolate
from src.models.model import split_X_y
from src.train_lstm import (
    AQILSTM,
    SEQUENCE_LENGTH,
    build_sequences,
    chronological_split_sequences,
    normalize_X,
    normalize_y,
    train_lstm,
)

TAIL_THRESHOLD = 250
VOLATILITY_THRESHOLD = 40


def score(y_true, y_pred, dates=None) -> dict:
    """Compute RMSE, MAE, tail MAE, and volatile-day MAE for one model's predictions."""
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    tail_mask = y_true > TAIL_THRESHOLD
    tail_mae = float(np.mean(np.abs(y_pred[tail_mask] - y_true[tail_mask]))) if tail_mask.sum() > 0 else None

    if dates is not None:
        date_diffs = dates[1:] - dates[:-1]
        truly_consecutive = date_diffs == pd.Timedelta(days=1)
        day_over_day = np.abs(y_true[1:] - y_true[:-1])
        volatile_mask = np.concatenate([[False], (day_over_day > VOLATILITY_THRESHOLD) & truly_consecutive])
    else:
        day_over_day = np.abs(np.diff(y_true))
        volatile_mask = np.concatenate([[False], day_over_day > VOLATILITY_THRESHOLD])

    n_volatile = int(volatile_mask.sum())
    volatile_mae = float(np.mean(np.abs(y_pred[volatile_mask] - y_true[volatile_mask]))) if n_volatile > 0 else None

    return {"rmse": rmse, "mae": mae, "n_tail": int(tail_mask.sum()), "tail_mae": tail_mae,
            "n_volatile": n_volatile, "volatile_mae": volatile_mae}


def main() -> None:
    """Score both models on their respective test sets and print side by side."""
    config = Config()

    # --- GBM ---
    gbm_test_df = pd.read_csv(f"{config.processed_dir}/test.csv", index_col=0, parse_dates=True)
    X_test, y_test_gbm = split_X_y(gbm_test_df)
    gbm_pipeline = joblib.load(config.model_path)
    gbm_pred = gbm_pipeline.predict(X_test)
    gbm_result = score(y_test_gbm.values, gbm_pred, dates=gbm_test_df.index)

    # --- LSTM (retrain fresh here for a self-contained comparison) ---
    raw = load_raw_data(config)
    daily = reindex_and_interpolate(raw, config)
    X, y, dates = build_sequences(daily, config, seq_len=SEQUENCE_LENGTH)
    (X_train, y_train, _), (X_val, y_val, _), (X_test_l, y_test_lstm, test_dates) = chronological_split_sequences(
        X, y, dates, config
    )
    X_train, X_val, X_test_l = normalize_X(X_train, X_val, X_test_l)
    y_train_n, y_val_n, _, y_mean, y_std = normalize_y(y_train, y_val, y_test_lstm)
    model, _ = train_lstm(X_train, y_train_n, X_val, y_val_n)
    model.eval()
    with torch.no_grad():
        lstm_pred_n = model(torch.from_numpy(X_test_l)).numpy()
    lstm_pred = lstm_pred_n * y_std + y_mean
    lstm_result = score(y_test_lstm, lstm_pred, dates=test_dates)

    print(f"GBM test rows: {len(y_test_gbm)}  (date range: {gbm_test_df.index.min().date()} to {gbm_test_df.index.max().date()})")
    print(f"LSTM test rows: {len(y_test_lstm)}  (date range: {test_dates.min().date()} to {test_dates.max().date()})")
    print("(Row counts may differ slightly -- different feature pipelines drop different boundary rows.)")

    print(f"\n{'Metric':<20} {'GBM':>10} {'LSTM':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    print(f"{'Test RMSE':<20} {gbm_result['rmse']:>10.2f} {lstm_result['rmse']:>10.2f}")
    print(f"{'Test MAE':<20} {gbm_result['mae']:>10.2f} {lstm_result['mae']:>10.2f}")
    print(f"{'Tail MAE (>250)':<20} {gbm_result['tail_mae']:>10.2f} {lstm_result['tail_mae']:>10.2f}")
    print(f"{'  (n tail rows)':<20} {gbm_result['n_tail']:>10} {lstm_result['n_tail']:>10}")
    print(f"{'Volatile-day MAE':<20} {gbm_result['volatile_mae']:>10.2f} {lstm_result['volatile_mae']:>10.2f}")
    print(f"{'  (n volatile rows)':<20} {gbm_result['n_volatile']:>10} {lstm_result['n_volatile']:>10}")


if __name__ == "__main__":
    main()