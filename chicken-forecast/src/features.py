"""Feature engineering helpers (EDA + Part 2 weekly forecasting)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features known / lagged at prediction time (no same-week target leakage).
# Contemporaneous: flock size + calendar only.
# Weather / feed / events: previous week (lag 1) — known before predicting this week.
FEATURE_COLUMNS: list[str] = [
    "hens_mean",
    "hens_min",
    "hens_max",
    "roosters_mean",
    "month",
    "week_of_year",
    "year",
    "n_days",
    # lags of target
    "eggs_lag1",
    "eggs_lag2",
    "eggs_lag3",
    "eggs_lag4",
    "eggs_per_hen_day_lag1",
    "eggs_per_hen_day_lag2",
    # lagged environment / feed
    "feed_kg_lag1",
    "feed_per_bird_day_kg_lag1",
    "feed_per_hen_day_kg_lag1",  # alias / back-compat with older models
    "temp_avg_lag1",
    "temp_min_lag1",
    "temp_max_lag1",
    "precipitation_lag1",
    "humidity_avg_lag1",
    "daylight_minutes_lag1",
    # lagged events
    "event_days_lag1",
    "event_death_lag1",
    "event_released_lag1",
    "event_added_lag1",
    "event_feed_change_lag1",
    "event_egg_anomaly_lag1",
]

TARGET_COL = "eggs"


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "eggs",
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Causal rolling means/std (shifted so current day is excluded)."""
    out = df.copy()
    series = out[value_col]
    for w in windows:
        out[f"{value_col}_roll{w}_mean"] = (
            series.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        out[f"{value_col}_roll{w}_std"] = (
            series.shift(1).rolling(w, min_periods=max(1, w // 2)).std()
        )
    return out


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = "eggs",
    lags: tuple[int, ...] = (1, 7, 14, 28),
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{value_col}_lag{lag}"] = out[value_col].shift(lag)
    return out


def flag_anomalies_zscore(
    df: pd.DataFrame,
    value_col: str = "eggs_per_hen",
    window: int = 28,
    z_thresh: float = 2.5,
) -> pd.DataFrame:
    """Flag unusual days vs recent baseline (causal rolling z-score)."""
    out = df.copy()
    base = out[value_col]
    roll_mean = base.shift(1).rolling(window, min_periods=max(7, window // 2)).mean()
    roll_std = base.shift(1).rolling(window, min_periods=max(7, window // 2)).std()
    z = (base - roll_mean) / roll_std.replace(0, np.nan)
    out[f"{value_col}_z{window}"] = z
    out["is_anomaly_low"] = z < -z_thresh
    out["is_anomaly_high"] = z > z_thresh
    return out


def event_impact_windows(
    df: pd.DataFrame,
    event_col: str,
    metric_col: str = "eggs_per_hen",
    pre: int = 7,
    post: int = 14,
) -> pd.DataFrame:
    """Build before/after windows around days where event_col is True."""
    event_dates = df.loc[df[event_col].fillna(False), "date"]
    rows: list[dict] = []
    indexed = df.set_index("date")
    for event_date in event_dates:
        for offset in range(-pre, post + 1):
            day = event_date + pd.Timedelta(days=offset)
            if day not in indexed.index:
                continue
            rows.append(
                {
                    "event_date": event_date,
                    "event_col": event_col,
                    "day_offset": offset,
                    metric_col: indexed.loc[day, metric_col],
                }
            )
    return pd.DataFrame(rows)


def build_weekly_model_frame(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal lag features for weekly forecasting.

    Target `eggs` is the same week's total. Features that would leak
    (same-week weather/feed/events) are only available as lags.
    """
    out = weekly.sort_values("week_start").reset_index(drop=True).copy()

    if "roosters_mean" not in out.columns:
        out["roosters_mean"] = 0.0

    lag_sources = [
        "eggs",
        "eggs_per_hen_day",
        "feed_kg",
        "feed_per_bird_day_kg",
        "feed_per_hen_day_kg",
        "temp_avg",
        "temp_min",
        "temp_max",
        "precipitation",
        "humidity_avg",
        "daylight_minutes",
        "event_days",
        "event_death",
        "event_released",
        "event_added",
        "event_feed_change",
        "event_egg_anomaly",
    ]
    for col in lag_sources:
        if col not in out.columns:
            out[col] = np.nan
        out[f"{col}_lag1"] = out[col].shift(1)

    out["eggs_lag2"] = out["eggs"].shift(2)
    out["eggs_lag3"] = out["eggs"].shift(3)
    out["eggs_lag4"] = out["eggs"].shift(4)
    out["eggs_per_hen_day_lag2"] = out["eggs_per_hen_day"].shift(2)

    # Need at least 4 prior weeks of history for lag4.
    out = out.dropna(subset=["eggs", "eggs_lag4"]).reset_index(drop=True)
    # Keep full weeks primarily; allow n_days>=5 for edge cases.
    out = out[out["n_days"] >= 5].reset_index(drop=True)
    return out


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def fit_seasonal_baseline(train_weekly: pd.DataFrame) -> dict:
    """
    Seasonal eggs-per-hen-day rates from training weeks only.

    Prefers week_of_year; falls back to month; then global mean.
    """
    eph = train_weekly["eggs_per_hen_day"].dropna()
    global_mean = float(eph.mean()) if len(eph) else 0.5

    by_woy = (
        train_weekly.dropna(subset=["eggs_per_hen_day"])
        .groupby("week_of_year")["eggs_per_hen_day"]
        .mean()
        .to_dict()
    )
    by_month = (
        train_weekly.dropna(subset=["eggs_per_hen_day"])
        .groupby("month")["eggs_per_hen_day"]
        .mean()
        .to_dict()
    )

    # Scale for score: std of (actual - baseline) on train.
    baselines = []
    actuals = []
    for _, row in train_weekly.dropna(subset=["eggs", "eggs_per_hen_day"]).iterrows():
        rate = by_woy.get(int(row["week_of_year"]))
        if rate is None:
            rate = by_month.get(int(row["month"]), global_mean)
        baseline_eggs = float(rate) * float(row["hens_mean"]) * float(row["n_days"])
        baselines.append(baseline_eggs)
        actuals.append(float(row["eggs"]))

    residuals = np.asarray(actuals) - np.asarray(baselines)
    residual_std = float(np.std(residuals)) if len(residuals) else 10.0

    return {
        "by_week_of_year": {int(k): float(v) for k, v in by_woy.items()},
        "by_month": {int(k): float(v) for k, v in by_month.items()},
        "global_mean_eph": global_mean,
        "residual_std": max(residual_std, 1.0),
    }


def seasonal_baseline_eggs(row: pd.Series, baseline: dict) -> float:
    """Expected weekly eggs given hens and season (no model)."""
    woy = int(row["week_of_year"])
    month = int(row["month"])
    rate = baseline["by_week_of_year"].get(woy)
    if rate is None:
        rate = baseline["by_month"].get(month, baseline["global_mean_eph"])
    return float(rate) * float(row["hens_mean"]) * float(row.get("n_days", 7))


def performance_score(
    predicted_eggs: float,
    baseline_eggs: float,
    residual_std: float,
) -> float:
    """
    Normalize (prediction - seasonal baseline) to [-1, +1].

    +1 = well above seasonal expectation; -1 = well below.
    Uses ~2 residual_std as the saturation scale.
    """
    scale = max(2.0 * residual_std, 0.2 * abs(baseline_eggs), 5.0)
    return float(np.clip((predicted_eggs - baseline_eggs) / scale, -1.0, 1.0))
