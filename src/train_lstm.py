"""LSTM experiment for the transition-lag limitation.

NOTE: this was written and code-reviewed but NOT executed as part of
the verification pass for the rest of this project -- installing
PyTorch in that environment pulled the full CUDA-bundled Linux wheel
(~3GB) and exhausted available disk, and the CPU-only wheel index
wasn't reachable from that environment's network allowlist. It was
first run on the actual dev machine, which surfaced a real bug: the
first version of this script normalized input features but left the
regression target (raw AQI, scale ~30-500) unnormalized. That produced
a test RMSE of 104.81 -- suspiciously close to what a "predict the
mean" baseline would score, which is the signature of an undertrained
network, not a genuine architecture-vs-GBM comparison. This version
normalizes the target too (train-set mean/std only, inverse-transformed
before reporting any metric) -- the standard fix, omitted the first
time around.

WHY THIS EXPERIMENT: transition_experiment.py found that hand-crafted
momentum/delta features made the GBM baseline WORSE, not better -- the
transition-lag limitation isn't fixable by feature engineering on a
tree model. This tests whether an architecture with actual temporal
recurrence does better.

HONEST FRAMING GOING IN: with ~901 usable training sequences, this is
a genuine risk, not a guaranteed win. Report whatever this actually
produces.

Run as:
    python -m src.train_lstm
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.data.load_data import load_raw_data
from src.data.preprocess import reindex_and_interpolate

SEQUENCE_LENGTH = 14
HIDDEN_SIZE = 32
NUM_EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


class AQILSTM(nn.Module):
    """A small single-layer LSTM for next-day AQI regression (normalized target)."""

    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out).squeeze(-1)


def build_sequences(daily: pd.DataFrame, config: Config, seq_len: int = SEQUENCE_LENGTH):
    """Build (X, y, dates) sliding-window sequences from continuous daily data."""
    feature_cols = [config.target_column] + config.weather_columns
    clean = daily.dropna(subset=feature_cols)
    values = clean[feature_cols].values.astype(np.float32)
    dates = clean.index

    X, y, target_dates = [], [], []
    for i in range(len(clean) - seq_len):
        window_dates = dates[i : i + seq_len + 1]
        expected_span = pd.date_range(window_dates[0], window_dates[-1], freq="D")
        if len(window_dates) != len(expected_span):
            continue

        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len, 0])
        target_dates.append(dates[i + seq_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(target_dates)


def chronological_split_sequences(X, y, dates, config: Config):
    """Time-ordered train/val/test split, consistent with preprocess.py."""
    n = len(X)
    n_test = int(n * config.test_frac)
    n_val = int(n * config.val_frac)
    n_train = n - n_test - n_val
    return (
        (X[:n_train], y[:n_train], dates[:n_train]),
        (X[n_train : n_train + n_val], y[n_train : n_train + n_val], dates[n_train : n_train + n_val]),
        (X[n_train + n_val :], y[n_train + n_val :], dates[n_train + n_val :]),
    )


def normalize_X(X_train, X_val, X_test):
    """Standardize input features using train-set statistics only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def normalize_y(y_train, y_val, y_test):
    """Standardize the regression target using train-set statistics only.

    This was the fix for the first run's 104.81 RMSE: without this,
    the network has to learn to shift its whole output range from
    near-zero (default init) up into AQI's actual 100s-300s scale,
    which a small model may not fully do within a capped epoch budget.
    Predictions are inverse-transformed back to real AQI units before
    any metric is computed -- reported metrics are always in true AQI
    points, never normalized units.
    """
    mean = y_train.mean()
    std = y_train.std() + 1e-8
    return (y_train - mean) / std, (y_val - mean) / std, (y_test - mean) / std, mean, std


def train_lstm(X_train, y_train, X_val, y_val):
    """Train with early stopping on val loss (normalized space), return best model."""
    torch.manual_seed(42)
    model = AQILSTM(n_features=X_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    else:
        print(f"Ran full {NUM_EPOCHS} epochs without triggering early stopping (still improving -- worth noting)")

    model.load_state_dict(best_state)
    return model, best_val_loss


def main() -> None:
    """Run the full LSTM experiment and report honestly against the GBM baseline."""
    config = Config()
    raw = load_raw_data(config)
    daily = reindex_and_interpolate(raw, config)

    X, y, dates = build_sequences(daily, config)
    print(f"Built {len(X)} sequences of length {SEQUENCE_LENGTH}")

    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, test_dates) = chronological_split_sequences(
        X, y, dates, config
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    X_train, X_val, X_test = normalize_X(X_train, X_val, X_test)
    y_train_n, y_val_n, y_test_n, y_mean, y_std = normalize_y(y_train, y_val, y_test)

    model, best_val_loss_n = train_lstm(X_train, y_train_n, X_val, y_val_n)
    print(f"Best val RMSE (real AQI units): {(best_val_loss_n ** 0.5) * y_std:.2f}")

    model.eval()
    with torch.no_grad():
        test_pred_n = model(torch.from_numpy(X_test)).numpy()
    test_pred = test_pred_n * y_std + y_mean  # inverse-transform to real AQI units

    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))
    test_mae = float(np.mean(np.abs(test_pred - y_test)))

    tail_mask = y_test > 250
    tail_mae = float(np.mean(np.abs(test_pred[tail_mask] - y_test[tail_mask]))) if tail_mask.sum() > 0 else None

    date_diffs = test_dates[1:] - test_dates[:-1]
    truly_consecutive = date_diffs == pd.Timedelta(days=1)
    day_over_day = np.abs(y_test[1:] - y_test[:-1])
    volatile_mask = np.concatenate([[False], (day_over_day > 40) & truly_consecutive])
    volatile_mae = (
        float(np.mean(np.abs(test_pred[volatile_mask] - y_test[volatile_mask]))) if volatile_mask.sum() > 0 else None
    )

    print("\n=== LSTM Test Results (target-normalization fix applied) ===")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE:  {test_mae:.2f}")
    print(f"Tail MAE (AQI>250, n={int(tail_mask.sum())}): {tail_mae}")
    print(f"Volatile-day MAE (n={int(volatile_mask.sum())}): {volatile_mae}")
    print(f"\nSanity check -- target std in test set: {float(np.std(y_test)):.2f}")
    print("(if test RMSE is still close to this number, the model still isn't learning much")
    print(" beyond the mean, and that's a real finding worth reporting as-is, not a bug to keep chasing)")

    print("\nRe-run python -m src.evaluate for the current GBM's test RMSE/MAE to compare directly.")


if __name__ == "__main__":
    main()