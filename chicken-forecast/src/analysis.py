"""Load summary stats for the analysis dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing import aggregate_weekly, prepare_daily
from src.features import event_impact_windows, flag_anomalies_zscore

MONTHS_SL = {
    1: "januar",
    2: "februar",
    3: "marec",
    4: "april",
    5: "maj",
    6: "junij",
    7: "julij",
    8: "avgust",
    9: "september",
    10: "oktober",
    11: "november",
    12: "december",
}

EVENT_LABELS = {
    "event_death": "Pogin kokoši",
    "event_released": "Izpust / odstranitev",
    "event_added": "Dodana žival / jata",
    "event_feed_change": "Sprememba hrane",
    "event_egg_anomaly": "Anomalija jajc",
}


def build_analysis_context() -> dict:
    df = prepare_daily()
    df = flag_anomalies_zscore(df, value_col="eggs_per_hen", window=28, z_thresh=2.5)
    weekly = aggregate_weekly(df)

    start = df["date"].min()
    end = df["date"].max()
    days = len(df)
    years = (end - start).days / 365.25

    monthly = (
        df.dropna(subset=["eggs_per_hen"])
        .groupby("month", as_index=False)
        .agg(
            eggs_per_hen=("eggs_per_hen", "mean"),
            eggs=("eggs", "mean"),
            n=("eggs", "count"),
        )
        .sort_values("month")
    )
    monthly["month_name"] = monthly["month"].map(MONTHS_SL)
    best = monthly.loc[monthly["eggs_per_hen"].idxmax()]
    worst = monthly.loc[monthly["eggs_per_hen"].idxmin()]

    corr = {}
    plot = df.dropna(subset=["eggs_per_hen", "temp_avg"])
    if len(plot) > 10:
        corr["temp_avg"] = float(plot["eggs_per_hen"].corr(plot["temp_avg"]))
        corr["precipitation"] = float(plot["eggs_per_hen"].corr(plot["precipitation"]))
        if plot["daylight_minutes"].notna().sum() > 30:
            ddf = plot.dropna(subset=["daylight_minutes"])
            corr["daylight_minutes"] = float(
                ddf["eggs_per_hen"].corr(ddf["daylight_minutes"])
            )

    cold = hot = mid = None
    if len(plot) > 20:
        q10, q90 = plot["temp_avg"].quantile(0.1), plot["temp_avg"].quantile(0.9)
        cold = float(plot.loc[plot["temp_avg"] <= q10, "eggs_per_hen"].mean())
        hot = float(plot.loc[plot["temp_avg"] >= q90, "eggs_per_hen"].mean())
        mid = float(
            plot.loc[
                (plot["temp_avg"] > q10) & (plot["temp_avg"] < q90), "eggs_per_hen"
            ].mean()
        )

    events = []
    for col, label in EVENT_LABELS.items():
        n = int(df[col].sum())
        item = {"key": col, "label": label, "n": n, "pre": None, "post": None, "delta": None}
        if n > 0:
            win = event_impact_windows(df, col, metric_col="eggs_per_hen", pre=7, post=14)
            if not win.empty:
                pre = float(win.loc[win["day_offset"] < 0, "eggs_per_hen"].mean())
                post = float(win.loc[win["day_offset"] > 0, "eggs_per_hen"].mean())
                item.update(
                    {
                        "pre": round(pre, 3),
                        "post": round(post, 3),
                        "delta": round(post - pre, 3),
                    }
                )
        events.append(item)

    feed_col = (
        "feed_per_bird_kg" if "feed_per_bird_kg" in df.columns else "feed_per_hen_kg"
    )

    return {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "n_days": days,
        "n_years": round(years, 2),
        "missing_eggs": int(df["eggs"].isna().sum()),
        "total_eggs": int(df["eggs"].sum(skipna=True)),
        "mean_eggs": round(float(df["eggs"].mean()), 2),
        "median_eggs": round(float(df["eggs"].median()), 1),
        "max_eggs": int(df["eggs"].max()),
        "mean_hens": round(float(df["hens"].mean()), 2),
        "min_hens": int(df["hens"].min()),
        "max_hens": int(df["hens"].max()),
        "mean_roosters": round(float(df["roosters"].mean()), 2)
        if "roosters" in df.columns
        else 0.0,
        "eggs_per_hen": round(float(df["eggs_per_hen"].mean()), 3),
        "feed_kg_day": round(float(df["feed_kg"].mean()), 3),
        "feed_g_per_bird": round(float(df[feed_col].mean() * 1000), 0),
        "feed_per_egg_kg": round(float(df["feed_per_egg_kg"].mean()), 3),
        "n_weeks": int(len(weekly)),
        "weekly_eggs_mean": round(float(weekly["eggs"].mean()), 1),
        "best_month": f"{int(best['month'])} ({best['month_name']})",
        "best_eph": round(float(best["eggs_per_hen"]), 3),
        "worst_month": f"{int(worst['month'])} ({worst['month_name']})",
        "worst_eph": round(float(worst["eggs_per_hen"]), 3),
        "monthly": [
            {
                "month": int(r.month),
                "name": r.month_name,
                "eggs_per_hen": round(float(r.eggs_per_hen), 3),
                "eggs": round(float(r.eggs), 2),
                "n": int(r.n),
            }
            for r in monthly.itertuples()
        ],
        "corr": {k: round(v, 3) for k, v in corr.items()},
        "extreme_temp": {
            "cold": round(cold, 3) if cold is not None else None,
            "hot": round(hot, 3) if hot is not None else None,
            "mid": round(mid, 3) if mid is not None else None,
        },
        "events": events,
        "anomalies_low": int(df["is_anomaly_low"].sum()),
        "anomalies_high": int(df["is_anomaly_high"].sum()),
        "takeaways": [
            "Produkcija je močno sezonska — jajca/kokoš se po mesecih jasno razlikujejo.",
            "Spremembe velikosti jate (smrti, izpusti, dodajanja) vplivajo na absolutna jajca; za primerjave uporabljaj eggs/hen.",
            "Petelin se šteje pri hrani, ne pri jajcih/kokoš.",
            "Vreme (temp / padavine) ima šibko linearno korelacijo; skrajnosti so vseeno drugačne od sredine.",
            "Največji padec eggs/hen po dogodku: dodajanje nove jate (mlade kokoši še ne nesejo polno).",
        ],
    }
