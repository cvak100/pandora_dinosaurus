# Part 2 – Forecasting evaluation

- Train weeks: **63** (2025-01-27 .. 2026-04-06)
- Val weeks: **16** (2026-04-13 .. 2026-07-27)
- LightGBM best_iteration: **70**

## Metrics
- Train MAE / RMSE: **1.11** / **1.43**
- Val MAE / RMSE / R2: **5.55** / **6.48** / **0.506**
- Seasonal baseline val MAE / RMSE: **7.92** / **9.57**
- Mean validation score: **0.536**

## Top features
- `eggs_lag1`: 70
- `temp_avg_lag1`: 51
- `humidity_avg_lag1`: 47
- `precipitation_lag1`: 46
- `temp_min_lag1`: 40
- `eggs_lag4`: 38
- `eggs_per_hen_day_lag1`: 37
- `daylight_minutes_lag1`: 37
- `eggs_lag2`: 36
- `eggs_lag3`: 28

Model saved to `models/weekly_lgbm.joblib`.
