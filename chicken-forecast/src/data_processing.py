"""Loading, cleaning and basic transforms for daily flock data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DAILY_CLEAN_CSV, DAILY_RAW_CSV, PROCESSED_DIR, RAW_DIR

REQUIRED_COLUMNS = [
    "date",
    "eggs",
    "hens",
    "feed_kg",
    "temp_min",
    "temp_max",
    "precipitation",
]


def load_raw_daily(path: Path | None = None) -> pd.DataFrame:
    """Load raw daily CSV produced by build_daily_dataset."""
    path = path or (RAW_DIR / DAILY_RAW_CSV)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - ensure chronological order
    - coerce numeric columns
    - keep missing eggs as NaN (do not force to 0)
    - empty events -> NaN-friendly empty string
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").reset_index(drop=True)

    numeric_cols = [
        "eggs",
        "hens",
        "roosters",
        "feed_g",
        "feed_kg",
        "temp_min",
        "temp_max",
        "temp_avg",
        "precipitation",
        "humidity_avg",
        "snow_depth_cm",
        "daylight_minutes",
        "pressure_hpa",
        "wind_speed_avg",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "events" in out.columns:
        out["events"] = out["events"].fillna("").astype(str)
    else:
        out["events"] = ""

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return out


def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eggs/hen uses hens only (rooster does not lay).
    Feed metrics use birds = hens + roosters (everyone eats).
    """
    out = df.copy()
    hens = out["hens"].replace(0, np.nan)
    roosters = out["roosters"].fillna(0) if "roosters" in out.columns else 0
    out["birds"] = out["hens"].fillna(0) + roosters
    birds = out["birds"].replace(0, np.nan)
    eggs = out["eggs"]
    feed = out["feed_kg"]

    out["eggs_per_hen"] = eggs / hens
    out["feed_per_bird_kg"] = feed / birds
    out["feed_per_hen_kg"] = out["feed_per_bird_kg"]  # back-compat alias
    out["feed_per_egg_kg"] = feed / eggs.replace(0, np.nan)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["day_of_year"] = out["date"].dt.dayofyear
    out["day_of_week"] = out["date"].dt.dayofweek  # Mon=0
    out["year_month"] = out["date"].dt.to_period("M").astype(str)
    return out


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags for common event categories."""
    out = df.copy()
    ev = out["events"].fillna("").str.lower()
    out["has_event"] = ev.str.len() > 0
    out["event_death"] = ev.str.contains("chicken_died|flock.chicken_died", regex=True)
    out["event_released"] = ev.str.contains(
        "chicken_released|flock.chicken_released", regex=True
    )
    out["event_added"] = ev.str.contains(
        "chicken_added|batch_started|flock.chicken_added|flock.batch_started",
        regex=True,
    )
    out["event_feed_change"] = ev.str.contains(
        r"feed\.(?:new_type|increased|reduced|returned|quality)", regex=True
    )
    out["event_egg_anomaly"] = ev.str.contains(
        r"flock\.eggs_(?:zero|high|low|change)", regex=True
    )
    out["event_water"] = ev.str.contains(r"water\.", regex=True)
    out["event_coop"] = ev.str.contains(r"coop\.", regex=True)
    return out


def prepare_daily(path: Path | None = None) -> pd.DataFrame:
    """Full Part-1 daily pipeline: load → clean → metrics → calendar → events."""
    df = load_raw_daily(path)
    df = clean_daily(df)
    df = add_efficiency_metrics(df)
    df = add_calendar_features(df)
    df = add_event_flags(df)
    return df


def save_processed(df: pd.DataFrame, name: str = DAILY_CLEAN_CSV) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily rows to ISO weeks (Mon-Sun).

    Incomplete trailing weeks (n_days < 7) are kept but flagged via n_days.
    """
    work = df.copy()
    work["week_start"] = work["date"] - pd.to_timedelta(work["date"].dt.dayofweek, unit="D")

    agg_map: dict[str, tuple[str, str]] = {
        "week_end": ("date", "max"),
        "eggs": ("eggs", "sum"),
        "eggs_mean": ("eggs", "mean"),
        "hens_mean": ("hens", "mean"),
        "hens_min": ("hens", "min"),
        "hens_max": ("hens", "max"),
        "feed_kg": ("feed_kg", "sum"),
        "temp_avg": ("temp_avg", "mean"),
        "temp_min": ("temp_min", "min"),
        "temp_max": ("temp_max", "max"),
        "precipitation": ("precipitation", "sum"),
        "humidity_avg": ("humidity_avg", "mean"),
        "daylight_minutes": ("daylight_minutes", "mean"),
        "n_days": ("date", "count"),
        "event_days": ("has_event", "sum"),
    }
    if "roosters" in work.columns:
        agg_map["roosters_mean"] = ("roosters", "mean")
    for col in (
        "event_death",
        "event_released",
        "event_added",
        "event_feed_change",
        "event_egg_anomaly",
        "event_water",
        "event_coop",
    ):
        if col in work.columns:
            agg_map[col] = (col, "sum")

    agg = work.groupby("week_start", as_index=False).agg(
        **{name: pd.NamedAgg(column=src, aggfunc=fn) for name, (src, fn) in agg_map.items()}
    )
    if "roosters_mean" not in agg.columns:
        agg["roosters_mean"] = 0.0
    agg["eggs_per_hen_day"] = agg["eggs"] / (agg["hens_mean"] * agg["n_days"]).replace(
        0, np.nan
    )
    # Feed per bird (hens + roosters) — rooster eats but does not lay.
    birds_mean = agg["hens_mean"] + agg["roosters_mean"].fillna(0)
    agg["birds_mean"] = birds_mean
    agg["feed_per_egg_kg"] = agg["feed_kg"] / agg["eggs"].replace(0, np.nan)
    agg["feed_per_bird_day_kg"] = agg["feed_kg"] / (birds_mean * agg["n_days"]).replace(
        0, np.nan
    )
    agg["feed_per_hen_day_kg"] = agg["feed_per_bird_day_kg"]  # back-compat alias
    agg["month"] = agg["week_start"].dt.month
    agg["week_of_year"] = agg["week_start"].dt.isocalendar().week.astype(int)
    agg["year"] = agg["week_start"].dt.year
    return agg.sort_values("week_start").reset_index(drop=True)