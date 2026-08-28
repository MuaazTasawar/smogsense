# Model Card — SmogSense

## Intended Use

Forecasts next-day AQI (PM2.5) for Lahore, Pakistan, from a recent
window of daily AQI and weather history, to give residents,
particularly those with respiratory conditions, advance warning
before hazardous air quality days. Intended as a portfolio/research
demonstration of a time-series forecasting pipeline for a real public
health problem — **not** a validated public health tool, and not a
substitute for official air quality monitoring or medical guidance.

## Training Data

- **Source:** [Lahore Air Quality Index](https://www.kaggle.com/datasets/shabbirchinioti/lahore-air-quality-index)
  (Kaggle), date-wise AQI (PM2.5) and weather data for Lahore.
- **Coverage:** 2019-06-01 to 2023-11-30, 1,644 raw daily rows out of
  1,801 possible calendar days (~8.7% of calendar days entirely
  absent from the source).
- **Missingness:** an additional 221 rows had a null `aqi_pm2.5`
  value even where a row existed. Combined, roughly 21% of the target
  series required either short-gap interpolation (gaps ≤ 3 days,
  linearly filled) or row exclusion (longer gaps, dropped rather than
  synthetically filled) before feature construction. After lag/rolling
  feature construction and gap-drop, 1,285 usable rows remained.
- **Known bias:** this is a single-city dataset; the model has no
  exposure to other cities' pollution dynamics (different traffic
  patterns, industrial mix, geography) and should not be assumed to
  generalize beyond Lahore.

## Model Architecture / Algorithm

`scikit-learn` Pipeline: `StandardScaler` → `GradientBoostingRegressor`
(`n_estimators=300`, `learning_rate=0.05`, `max_depth=3`). Chosen as a
classical baseline over an LSTM/Transformer given the dataset's size
(~1,285 usable rows) — see Limitations and Future Work for the
DL upgrade path this baseline motivates.

Features (18 total): 5 lagged AQI values (1/2/3/7/14 days), rolling
7/14/3-day mean and std of AQI (computed on shifted history — no
same-day leakage), and 5 same-day weather readings (temperature, dew
point, humidity, wind speed, pressure). Target: next-day AQI.

## Evaluation Metrics & Results

Chronological split (never shuffled): train on 2019–2022, validate on
2022–2023, test on the most recent unseen window (2023-05 to
2023-11).

| Split | RMSE  | MAE   |
|-------|-------|-------|
| Train | 24.67 | 18.67 |
| Val   | 50.41 | 35.60 |
| Test  | 35.82 | 25.56 |

Test RMSE/MAE fall between train and val, indicating the train/val
gap reflects genuine period-to-period difficulty rather than val
being an unrepresentative outlier.

Diagnostic figures (`reports/figures/`): `eval_actual_vs_predicted.png`,
`eval_residuals_over_time.png`, `eval_regression_diagnostics.png`,
`train_feature_importance.png`.

## Limitations

1. **Underestimates extreme tail values.** The target distribution is
   right-skewed (bulk of values 100–200, tail to ~500). The MSE-trained
   model systematically underpredicts actual AQI above ~300 (visible
   in `eval_regression_diagnostics.png`) — the exact regime where
   accurate warning matters most.
2. **Lags during rapid AQI transitions.** The model leans heavily on
   `aqi_lag_1` and `aqi_roll_mean_7` (see `train_feature_importance.png`),
   so it tracks slow-moving trends well but is measurably slower to
   react during sharp smog-onset swings (visible in
   `eval_actual_vs_predicted.png` around Oct–Nov 2023).
3. **Mild overfitting.** Train RMSE (24.67) is meaningfully lower than
   held-out RMSE (35.82 test, 50.41 val) at current hyperparameters.
4. **Weather features are same-day observed values, not forecasts.**
   A real deployment would need next-day weather *forecasts* as input,
   not next-day weather *observations* (which aren't available at
   prediction time) — the ~26-36 point test error already includes
   this simplification's effect and would need re-evaluation with
   real forecast inputs before any operational use.
5. **Single-city, single-pollutant.** Trained only on Lahore PM2.5-
   derived AQI; not validated for other cities or composite
   multi-pollutant AQI.
6. **Uncertainty band is naive.** `predict.py`'s uncertainty band is
   forecast ± test RMSE (a rough symmetric heuristic), not a
   calibrated prediction interval (e.g. quantile regression) — it
   should not be read as a statistically rigorous confidence interval.

## Future Work

- LSTM/Temporal Fusion Transformer upgrade, directly motivated by
  Limitation #2 (transition-lag) — worth revisiting once more data is
  available, since the classical baseline's main failure mode is
  exactly what sequence models tend to handle better.
- Quantile regression for calibrated (non-naive) uncertainty bands.
- Multi-city extension (e.g. adding the Islamabad Kaggle dataset) to
  test generalization.
- Swap same-day weather for a real weather-forecast API input.

## Ethical Considerations

This tool forecasts environmental risk that disproportionately harms
already-vulnerable groups (children, the elderly, people with
respiratory conditions, outdoor workers). A false "safe" prediction
during an actual hazardous-AQI day is a more costly error than a false
alarm — Limitation #1 (tail underestimation) means this baseline's
current error pattern skews toward the *more costly* direction, and
should be corrected (via Future Work) before any real-world advisory
use, not treated as an acceptable trade-off.