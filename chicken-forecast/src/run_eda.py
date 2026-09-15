"""
Part 1 EDA runner: generates plots in reports/ and a written summary.

Run from project root:
  venv\\Scripts\\python.exe -m src.run_eda
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_USERNAME, WEEKLY_CSV  # noqa: E402
from src.data_processing import (  # noqa: E402
    aggregate_weekly,
    prepare_daily,
    save_processed,
)
from src.features import (  # noqa: E402
    event_impact_windows,
    flag_anomalies_zscore,
)

REPORTS = ROOT / "reports"
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run() -> None:
    df = prepare_daily()
    df = flag_anomalies_zscore(df, value_col="eggs_per_hen", window=28, z_thresh=2.5)
    processed_path = save_processed(df)
    weekly = aggregate_weekly(df)
    weekly.to_csv(ROOT / "data" / "processed" / WEEKLY_CSV, index=False)

    findings: list[str] = []
    findings.append(f"# Part 1 – EDA Summary ({DEFAULT_USERNAME})")
    findings.append("")
    findings.append(
        f"- Period: **{df['date'].min().date()} -> {df['date'].max().date()}** "
        f"({len(df)} days; missing egg records: {int(df['eggs'].isna().sum())})"
    )
    findings.append(
        f"- Mean daily eggs: **{df['eggs'].mean():.2f}** "
        f"(median {df['eggs'].median():.1f}; max {df['eggs'].max():.0f})"
    )
    findings.append(
        f"- Mean hens: **{df['hens'].mean():.2f}** "
        f"(range {df['hens'].min():.0f}–{df['hens'].max():.0f})"
    )
    findings.append(
        f"- Mean eggs/hen/day: **{df['eggs_per_hen'].mean():.3f}**"
    )
    findings.append(
        f"- Mean feed: **{df['feed_kg'].mean():.3f} kg/day** "
        f"({df['feed_per_hen_kg'].mean()*1000:.0f} g/hen)"
    )
    findings.append(
        f"- Feed conversion: **{df['feed_per_egg_kg'].mean():.3f} kg feed / egg**"
    )

    # 1) Time series
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df["date"], df["eggs"], color="#2c7fb8", lw=1)
    axes[0].set_ylabel("Eggs")
    axes[0].set_title("Daily egg production")
    axes[1].plot(df["date"], df["eggs_per_hen"], color="#41b6c4", lw=1)
    axes[1].set_ylabel("Eggs / hen")
    axes[2].plot(df["date"], df["feed_kg"], color="#8c6bb1", lw=1)
    axes[2].set_ylabel("Feed kg")
    axes[3].plot(df["date"], df["hens"], color="#4d9221", lw=1)
    axes[3].set_ylabel("Hens")
    axes[3].set_xlabel("Date")
    _save(fig, "01_timeseries_production.png")

    # 2) Weekly eggs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(weekly["week_start"], weekly["eggs"], width=5, color="#2c7fb8", alpha=0.85)
    ax.set_title("Weekly total eggs")
    ax.set_ylabel("Eggs / week")
    _save(fig, "02_weekly_eggs.png")

    # 3) Seasonality by month
    monthly = (
        df.dropna(subset=["eggs_per_hen"])
        .groupby("month", as_index=False)
        .agg(eggs_per_hen=("eggs_per_hen", "mean"), eggs=("eggs", "mean"), n=("eggs", "count"))
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=monthly, x="month", y="eggs_per_hen", color="#41b6c4", ax=ax)
    ax.set_title("Average eggs per hen by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Eggs / hen / day")
    _save(fig, "03_seasonality_month.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df.dropna(subset=["eggs"]), x="month", y="eggs", color="#a6bddb", ax=ax)
    ax.set_title("Daily eggs distribution by month")
    _save(fig, "04_boxplot_eggs_month.png")

    best_m = monthly.loc[monthly["eggs_per_hen"].idxmax()]
    worst_m = monthly.loc[monthly["eggs_per_hen"].idxmin()]
    findings.append(
        f"- Seasonality (eggs/hen): best month **{int(best_m['month'])}** "
        f"({best_m['eggs_per_hen']:.3f}), worst **{int(worst_m['month'])}** "
        f"({worst_m['eggs_per_hen']:.3f})"
    )

    # 4) Daylight / temperature vs production
    plot_df = df.dropna(subset=["eggs_per_hen", "temp_avg"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(plot_df["temp_avg"], plot_df["eggs_per_hen"], alpha=0.35, s=12, c="#2c7fb8")
    axes[0].set_xlabel("Temp avg °C")
    axes[0].set_ylabel("Eggs / hen")
    axes[0].set_title("Temperature vs eggs/hen")
    if plot_df["daylight_minutes"].notna().sum() > 30:
        ddf = plot_df.dropna(subset=["daylight_minutes"])
        axes[1].scatter(
            ddf["daylight_minutes"], ddf["eggs_per_hen"], alpha=0.35, s=12, c="#41b6c4"
        )
        axes[1].set_xlabel("Daylight minutes")
    else:
        axes[1].scatter(
            plot_df["precipitation"], plot_df["eggs_per_hen"], alpha=0.35, s=12, c="#8c6bb1"
        )
        axes[1].set_xlabel("Precipitation mm")
    axes[1].set_ylabel("Eggs / hen")
    axes[1].set_title("Environment vs eggs/hen")
    _save(fig, "05_weather_scatter.png")

    # 5) Correlation heatmap
    corr_cols = [
        "eggs",
        "eggs_per_hen",
        "hens",
        "feed_kg",
        "feed_per_hen_kg",
        "feed_per_egg_kg",
        "temp_avg",
        "temp_min",
        "temp_max",
        "precipitation",
        "humidity_avg",
        "daylight_minutes",
        "month",
        "day_of_year",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation heatmap")
    _save(fig, "06_correlation_heatmap.png")

    if "temp_avg" in corr.columns:
        findings.append(
            f"- Corr(eggs_per_hen, temp_avg) = **{corr.loc['eggs_per_hen', 'temp_avg']:.3f}**"
        )
    if "daylight_minutes" in corr.columns and corr["daylight_minutes"].notna().any():
        findings.append(
            f"- Corr(eggs_per_hen, daylight_minutes) = "
            f"**{corr.loc['eggs_per_hen', 'daylight_minutes']:.3f}**"
        )
    if "precipitation" in corr.columns:
        findings.append(
            f"- Corr(eggs_per_hen, precipitation) = "
            f"**{corr.loc['eggs_per_hen', 'precipitation']:.3f}**"
        )

    # Extreme weather
    if plot_df["temp_avg"].notna().any():
        cold = plot_df[plot_df["temp_avg"] <= plot_df["temp_avg"].quantile(0.1)]
        hot = plot_df[plot_df["temp_avg"] >= plot_df["temp_avg"].quantile(0.9)]
        mid = plot_df[
            (plot_df["temp_avg"] > plot_df["temp_avg"].quantile(0.1))
            & (plot_df["temp_avg"] < plot_df["temp_avg"].quantile(0.9))
        ]
        findings.append(
            f"- Extreme temp: coldest 10% eggs/hen **{cold['eggs_per_hen'].mean():.3f}**, "
            f"hottest 10% **{hot['eggs_per_hen'].mean():.3f}**, "
            f"mid **{mid['eggs_per_hen'].mean():.3f}**"
        )

    # 6) Event impact
    event_cols = [
        "event_death",
        "event_released",
        "event_added",
        "event_feed_change",
        "event_egg_anomaly",
    ]
    impact_rows = []
    for col in event_cols:
        n = int(df[col].sum())
        if n == 0:
            continue
        win = event_impact_windows(df, col, metric_col="eggs_per_hen", pre=7, post=14)
        if win.empty:
            continue
        curve = win.groupby("day_offset", as_index=False)["eggs_per_hen"].mean()
        curve["event"] = col
        impact_rows.append(curve)
        pre = win.loc[win["day_offset"] < 0, "eggs_per_hen"].mean()
        post = win.loc[win["day_offset"] > 0, "eggs_per_hen"].mean()
        findings.append(
            f"- Event `{col}` (n={n}): pre7 eggs/hen **{pre:.3f}** -> "
            f"post14 **{post:.3f}** (delta {post - pre:+.3f})"
        )

    if impact_rows:
        impact = pd.concat(impact_rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        for name, part in impact.groupby("event"):
            ax.plot(part["day_offset"], part["eggs_per_hen"], marker="o", ms=3, label=name)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlabel("Days relative to event")
        ax.set_ylabel("Mean eggs / hen")
        ax.set_title("Event impact windows (−7 … +14 days)")
        ax.legend(fontsize=8)
        _save(fig, "07_event_impact.png")

    # Event count summary plot
    counts = {c: int(df[c].sum()) for c in event_cols}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color="#7bccc4")
    ax.set_title("Event counts in dataset")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "08_event_counts.png")

    # 7) Anomalies
    anom = df[df["is_anomaly_low"] | df["is_anomaly_high"]].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["eggs_per_hen"], color="#9ebcda", lw=1, label="eggs/hen")
    ax.scatter(
        anom.loc[anom["is_anomaly_low"], "date"],
        anom.loc[anom["is_anomaly_low"], "eggs_per_hen"],
        c="#d7301f",
        s=28,
        label="low anomaly",
        zorder=3,
    )
    ax.scatter(
        anom.loc[anom["is_anomaly_high"], "date"],
        anom.loc[anom["is_anomaly_high"], "eggs_per_hen"],
        c="#238b45",
        s=28,
        label="high anomaly",
        zorder=3,
    )
    ax.set_title("Anomalies vs 28-day causal baseline (|z| > 2.5)")
    ax.legend()
    _save(fig, "09_anomalies.png")
    findings.append(
        f"- Anomalies (|z|>2.5 on eggs/hen): low **{int(df['is_anomaly_low'].sum())}**, "
        f"high **{int(df['is_anomaly_high'].sum())}**"
    )

    # 8) Feed efficiency over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["feed_per_egg_kg"], color="#8c6bb1", lw=1)
    ax.set_title("Feed conversion (kg feed / egg)")
    ax.set_ylabel("kg / egg")
    _save(fig, "10_feed_conversion.png")

    # Written summary
    findings.append("")
    findings.append("## Assumptions")
    findings.append(f"- User filter: **{DEFAULT_USERNAME} only**.")
    findings.append(
        "- Feed logs are date ranges; quantity is **grams/day** for each day in the range "
        "(overlap for same food type → later `from_date` wins)."
    )
    findings.append("- Weather from `data/raw/2025.json` + `2026.json`.")
    findings.append(
        f"- Processed table: `{processed_path.relative_to(ROOT).as_posix()}`"
    )
    findings.append("")
    findings.append("## Main takeaways")
    findings.append(
        "1. Production is strongly seasonal — eggs/hen varies materially by month."
    )
    findings.append(
        "2. Flock size changes (deaths / releases / additions) shift absolute egg counts; "
        "always normalize by hens for comparisons."
    )
    findings.append(
        "3. Weather (temperature / daylight when available) correlates with laying rate; "
        "extreme cold/hot tails differ from mid-range days."
    )
    findings.append(
        "4. Feed is relatively stable over long ranges; conversion (kg/egg) is a useful "
        "efficiency KPI alongside eggs/hen."
    )
    findings.append(
        "5. Event windows (especially mortality and feed changes) are worth modeling "
        "as features in Part 2 weekly forecasting."
    )

    summary_path = REPORTS / "eda_summary.md"
    summary_path.write_text("\n".join(findings) + "\n", encoding="utf-8")
    print(f"Processed: {processed_path}")
    print(f"Summary:   {summary_path}")
    print(f"Plots:     {REPORTS}")
    print("\n".join(findings))


if __name__ == "__main__":
    run()
