# Model Card — SmogSense

## Intended Use

Forecasts next-day AQI (PM2.5) for Lahore, Pakistan, from a recent window of daily AQI and weather history, to give residents, particularly those with respiratory conditions, advance warning before hazardous air quality days. Intended as a portfolio/research demonstration of a time-series forecasting pipeline for a real public health problem -- not a validated public health tool, and not a substitute for official air quality monitoring or medical guidance.

## Training Data

- Source: Lahore Air Quality Index (Kaggle: shabbirchinioti/lahore-air-quality-index), date-wise AQI (PM2.5) and weather data for Lahore.
- Coverage: 2019-06-01 to 2023-11-30 -- 1,644 rows for 1,644 calendar days in that range. Calendar coverage is complete (0 missing days). An earlier draft of this document reported 157 missing calendar days -- an artifact of a date-parsing bug (fixed in load_data.py), corrected here.
- Missingness: 221 of the 1,644 rows have a null aqi_pm2.5 value (~13.4% of rows). Short runs (<= 3 consecutive days) are linearly interpolated; longer runs are dropped rather than synthetically filled. After lag/rolling feature construction, 1,285 usable rows remained for the GBM pipeline (359 dropped total); the LSTM's sequence-based pipeline yields 1,294 usable 14-day sequences via a slightly different construction.
- Known bias: single-city dataset; not validated to generalize beyond Lahore.

## Production Model

**GBM remains the production model**, after a direct, honest comparison against an LSTM (see Model Iterations #4 below). scikit-learn Pipeline: StandardScaler -> GradientBoostingRegressor, `n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.85, min_samples_leaf=10`, trained with `sample_weight=y_train` (tail sample-weighting). Hyperparameters selected by `src/tune.py`'s grid search against the chronological val split.

Quantile models (`src/train_quantiles.py`): same tree structure, `loss="quantile"` at alpha=0.1/0.9, unweighted.

Features (18 total): 5 lagged AQI values, rolling 7/14/3-day mean and std, 5 same-day weather readings. Target: next-day AQI.

## Evaluation Metrics and Results

Chronological split: train 2019-2022, val 2022-2023, test 2023-05 to 2023-11.

**Baseline (v1)** -- original hyperparameters, unweighted:

| Split | RMSE  | MAE   |
|-------|-------|-------|
| Train | 24.67 | 18.67 |
| Val   | 50.41 | 35.60 |
| Test  | 35.82 | 25.56 |

**Production (v2)** -- tuned hyperparameters + tail sample-weighting:

| Split | RMSE  | MAE   |
|-------|-------|-------|
| Train | 33.83 | 25.19 |
| Val   | 48.75 | 34.24 |
| Test  | 36.14 | 26.60 |

**Tail-specific results (test set, actual AQI > 250):**

| Model | Tail MAE | 
|-------|----------|
| v1 (unweighted) | 41.41 |
| v2 (tail-weighted, production) | 40.14 |

## Uncertainty Quantification

`predict.py` returns a real quantile-regression-derived band. Empirical coverage on val: **73.4%** against an 80% target -- reasonably close, reported as measured. Asymmetric by construction (wider on the high side, matching the right-skewed target).

## Model Iterations (Post-Baseline Work)

Four limitations addressed directly, in increasing order of effort, every change tested honestly:

**1. Overfitting (`src/tune.py`).** Grid search shrank the train/val RMSE gap substantially; test-set improvement was modest, not dramatic.

**2. Tail underestimation (`src/tail_experiment.py`).** sample_weight=y_train: real trade-off -- tail MAE improved (41.41 -> 40.14 on test), overall test RMSE/MAE ticked up slightly (35.82 -> 36.14, 25.56 -> 26.60). Adopted, given the project's own stated ethics (see below).

**3. Naive uncertainty band (`src/train_quantiles.py`).** Replaced with real quantile regression -- 73.4% empirical coverage against an 80% target.

**4. Transition-lag (`src/transition_experiment.py`, `src/train_lstm.py`).** Two attempts, both run for real and reported honestly:
   - Hand-engineered momentum/delta features: **made things worse** (val RMSE 48.00 -> 48.36, volatile-day MAE 52.75 -> 54.32). Not adopted.
   - An LSTM (14-day sequences, single-layer, 32 hidden units, early-stopped) was built, and its first run produced a test RMSE of 104.81 -- close enough to the test set's own standard deviation (77.58) to indicate the network hadn't meaningfully learned anything, not that LSTMs are inherently worse here. Root cause: the regression target was left unnormalized while input features were standardized, likely preventing the small network from converging within the epoch budget. Fixed by normalizing the target too (train-set statistics, inverse-transformed before any metric is computed) -- test RMSE dropped to 32.36.

   **Direct, same-test-set comparison (`src/compare_models.py`):**

   | Metric | GBM (production) | LSTM |
   |---|---|---|
   | Test RMSE | 36.14 | **32.36** |
   | Test MAE | 26.60 | **23.12** |
   | Tail MAE (>250) | 40.14 | **37.41** |
   | Volatile-day MAE | **48.43** | 48.95 |

   **Honest read:** the LSTM wins on overall accuracy and tail behavior by a real, meaningful margin. But on volatile-day MAE -- the specific metric this whole experiment was built to fix -- the two are within noise of each other (0.52 points on n=40), with GBM marginally ahead. The LSTM is a genuinely better general forecaster; it did not clearly solve the specific transition-lag problem that motivated building it.

   **Decision: GBM stays production.** Promoting the LSTM would require real, unbuilt engineering -- torch as a hard dependency, a new uncertainty-quantification approach (no quantile-LSTM equivalent exists yet), an API rework, and loss of the interpretability the GBM's feature-importance plot provides -- in exchange for a win that is real in aggregate but doesn't clearly address the motivating problem. The LSTM code, results, and this reasoning are kept here as validated future work, not discarded.

## Limitations

1. **Tail underestimation, improved but not solved.** Real bias remains even with tail sample-weighting.
2. **Transition-lag, genuinely tested, not clearly solved by either approach tried.** Feature engineering made it worse; the LSTM roughly ties the GBM on this specific axis despite winning elsewhere.
3. **Mild overfitting persists**, reduced but not eliminated by tuning.
4. **Weather features are same-day observed, not forecast.**
5. **Single-city, single-pollutant.**
6. **Uncertainty band is real but imperfectly calibrated** (73.4% vs 80% target).

## Future Work

- Promote the LSTM to production IF a calibrated uncertainty approach for it is built first (e.g. MC-dropout or a quantile-loss LSTM) -- the accuracy case is there; the supporting infrastructure isn't yet.
- Multi-city extension to increase effective training data, which may be what genuinely resolves the transition-lag tie either way.
- Isotonic recalibration for the quantile bands.
- Real weather-forecast input instead of same-day observed weather.

## Ethical Considerations

This tool forecasts environmental risk that disproportionately harms already-vulnerable groups. A false "safe" prediction on a hazardous day is a more costly error than a false alarm. This was applied concretely in tail sample-weighting (Model Iterations #2) and was a real factor in weighing whether the LSTM's mixed result on the safety-relevant transition-lag metric was enough to justify a production swap -- it wasn't, given the specific axis that matters most didn't clearly improve.