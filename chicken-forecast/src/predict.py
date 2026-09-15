"""Predict next week's eggs and performance score (-1 .. +1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_USERNAME  # noqa: E402
from src.data_processing import aggregate_weekly, prepare_daily  # noqa: E402
from src.features import (  # noqa: E402
    performance_score,
    seasonal_baseline_eggs,
)

MODELS = ROOT / "models"
DEFAULT_MODEL = MODELS / "weekly_lgbm.joblib"

LAG_COLS = [
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


def load_artifact(path: Path = DEFAULT_MODEL) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run: python -m src.train"
        )
    return joblib.load(path)


def build_next_week_row(weekly: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Construct a feature row for the week after the last completed week.

    Returns (next_week_row, last_completed_week_row).
    """
    hist = weekly.sort_values("week_start").reset_index(drop=True)
    full = hist[hist["n_days"] >= 7]
    if full.empty:
        full = hist
    last = full.iloc[-1]
    next_start = pd.Timestamp(last["week_start"]) + pd.Timedelta(days=7)

    stub = last.copy()
    stub["week_start"] = next_start
    stub["week_end"] = next_start + pd.Timedelta(days=6)
    stub["eggs"] = float("nan")
    stub["eggs_mean"] = float("nan")
    stub["eggs_per_hen_day"] = float("nan")
    stub["n_days"] = 7
    stub["month"] = int(next_start.month)
    stub["week_of_year"] = int(next_start.isocalendar().week)
    stub["year"] = int(next_start.year)

    for col in (
        "feed_kg",
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
        "feed_per_egg_kg",
        "feed_per_hen_day_kg",
        "feed_per_bird_day_kg",
    ):
        if col in stub.index:
            stub[col] = float("nan")

    extended = pd.concat([hist, stub.to_frame().T], ignore_index=True)
    extended = extended.sort_values("week_start").reset_index(drop=True)
    for col in LAG_COLS:
        if col not in extended.columns:
            extended[col] = float("nan")
        extended[f"{col}_lag1"] = extended[col].shift(1)
    extended["eggs_lag2"] = extended["eggs"].shift(2)
    extended["eggs_lag3"] = extended["eggs"].shift(3)
    extended["eggs_lag4"] = extended["eggs"].shift(4)
    extended["eggs_per_hen_day_lag2"] = extended["eggs_per_hen_day"].shift(2)

    row = extended.iloc[-1]
    if pd.isna(row["eggs_lag4"]):
        raise RuntimeError("Not enough history to build lag features for next week.")
    return row, last


def _f(value, digits: int = 2):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), digits)


def predict_next_week(model_path: Path = DEFAULT_MODEL) -> dict:
    artifact = load_artifact(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    baseline = artifact["baseline"]

    daily = prepare_daily()
    weekly = aggregate_weekly(daily)
    row, last = build_next_week_row(weekly)

    X = pd.DataFrame([{c: row.get(c) for c in feature_cols}])
    pred = float(model.predict(X)[0])
    base = seasonal_baseline_eggs(row, baseline)
    score = performance_score(pred, base, baseline["residual_std"])
    delta = pred - base
    hens = float(row["hens_mean"])
    roosters = float(row.get("roosters_mean") or 0.0)
    birds = hens + roosters
    n_days = float(row.get("n_days") or 7)
    pred_per_hen_day = pred / (hens * n_days) if hens else None
    base_per_hen_day = base / (hens * n_days) if hens else None

    woy = int(row["week_of_year"])
    month = int(row["month"])
    rate_woy = baseline["by_week_of_year"].get(woy)
    rate_month = baseline["by_month"].get(month)
    rate_used = rate_woy if rate_woy is not None else rate_month
    rate_source = (
        f"week_of_year={woy}"
        if rate_woy is not None
        else f"month={month} (fallback)"
        if rate_month is not None
        else "global_mean"
    )
    if rate_used is None:
        rate_used = baseline["global_mean_eph"]
        rate_source = "global_mean"

    scale = max(2.0 * baseline["residual_std"], 0.2 * abs(base), 5.0)

    if score > 0.15:
        meaning = "above expected"
        meaning_sl = "nad pričakovanim"
    elif score < -0.15:
        meaning = "below expected"
        meaning_sl = "pod pričakovanim"
    else:
        meaning = "near normal"
        meaning_sl = "blizu normalnega"

    feature_values = []
    for c in feature_cols:
        v = row.get(c)
        try:
            v = None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
        except (TypeError, ValueError):
            v = str(v)
        feature_values.append({"feature": c, "value": _f(v, 4) if isinstance(v, float) else v})

    detailed_method = [
        "Kako pride model do napovedanega števila jajc (korak za korakom):",
        "",
        "1) Podatki: dnevni zapisi jate (jajca, kokoši, hrana, vreme, dogodki) "
        "se agregirajo v ISO tedne (ponedeljek–nedelja).",
        "2) Tarča modela: skupno število jajc v tednu (ne dnevno).",
        "3) Za napoved naslednjega tedna zgradimo feature vektor iz že zaključenih tednov — "
        "brez 'gledanja v prihodnost'. Vreme/hrana/dogodki za napovedani teden niso znani, "
        "zato uporabimo njihove lag1 vrednosti (prejšnji teden). Jajca vstopijo kot lag1–lag4.",
        f"4) Trenutna jata za napovedani teden: {hens:.2f} kokoši (+ {roosters:.0f} petelin). "
        "Petelin se NE šteje pri eggs/hen in NE v sezonskem baseline-u jajc; šteje se le pri hrani.",
        f"5) LightGBM Regressor (drevesni ansambel) prejme {len(feature_cols)} featurejev "
        f"in vrne napoved: {pred:.2f} jajc. Model se je učil na zgodovinskih tednih s strogim "
        "časovnim splitom (zadnji tedni = validacija, brez naključnega mešanja).",
        f"6) Najmočnejši signal so običajno pretekla jajca. Zadnji 4 tedni: "
        f"{_f(row.get('eggs_lag1'),1)}, {_f(row.get('eggs_lag2'),1)}, "
        f"{_f(row.get('eggs_lag3'),1)}, {_f(row.get('eggs_lag4'),1)}. "
        f"Zadnji zaključeni teden ({pd.Timestamp(last['week_start']).date()}–"
        f"{pd.Timestamp(last['week_end']).date()}) je imel {_f(last.get('eggs'),1)} jajc — "
        "zato je napoved blizu te ravni (rahlo prilagojena z vremenom/sezono/jate).",
        f"7) Sezonski baseline (primerjava, ni LightGBM): zgodovinska povprečna eggs/hen/dan "
        f"za {rate_source} = {float(rate_used):.4f}. "
        f"Baseline jajc = {float(rate_used):.4f} × {hens:.2f} kokoši × {n_days:.0f} dni "
        f"= {base:.2f} jajc.",
        f"8) Razlika model − baseline = {pred:.2f} − {base:.2f} = {delta:+.2f} jajc "
        f"({(delta/base*100) if base else 0:+.1f} %).",
        f"9) Score (−1…+1) = clip( (napoved − baseline) / skala , −1, 1), "
        f"kjer skala = max(2×residual_std, 0.2×|baseline|, 5) = "
        f"max(2×{baseline['residual_std']:.2f}, 0.2×{abs(base):.2f}, 5) = {scale:.2f}. "
        f"Torej score = clip({delta:.2f}/{scale:.2f}, −1, 1) = {score:+.3f} ({meaning_sl}).",
        "10) Interpretacija: score meri, ali model pričakuje boljšo/slabšo proizvodnjo "
        "od sezonske norme pri trenutnem številu kokoši — ni napaka modela, ampak "
        "indikator pričakovane uspešnosti.",
    ]

    result = {
        "week_start": str(pd.Timestamp(row["week_start"]).date()),
        "week_end": str(pd.Timestamp(row["week_end"]).date()),
        "week_of_year": woy,
        "month": month,
        "year": int(row["year"]),
        "n_days": int(n_days),
        "hens_mean": _f(hens, 2),
        "roosters_mean": _f(roosters, 2),
        "birds_mean": _f(birds, 2),
        "note_flock": (
            "Jajca / baseline: samo kokoši. Hrana: kokoši + petelin. "
            "Petelin se ne šteje v eggs/hen."
        ),
        "predicted_eggs": _f(pred, 2),
        "seasonal_baseline_eggs": _f(base, 2),
        "delta_eggs": _f(delta, 2),
        "delta_pct": _f((delta / base * 100.0) if base else None, 1),
        "predicted_per_hen_day": _f(pred_per_hen_day, 3),
        "baseline_per_hen_day": _f(base_per_hen_day, 3),
        "predicted_per_day": _f(pred / n_days, 2),
        "score": _f(score, 3),
        "score_meaning": meaning,
        "score_meaning_sl": meaning_sl,
        "score_scale": _f(scale, 2),
        "residual_std": _f(baseline["residual_std"], 2),
        "baseline_rate_eph": _f(float(rate_used), 4),
        "baseline_rate_source": rate_source,
        "last_week_start": str(pd.Timestamp(last["week_start"]).date()),
        "last_week_end": str(pd.Timestamp(last["week_end"]).date()),
        "last_week_eggs": _f(last.get("eggs"), 1),
        "last_week_hens": _f(last.get("hens_mean"), 2),
        "last_week_feed_kg": _f(last.get("feed_kg"), 2),
        "eggs_lag1": _f(row.get("eggs_lag1"), 1),
        "eggs_lag2": _f(row.get("eggs_lag2"), 1),
        "eggs_lag3": _f(row.get("eggs_lag3"), 1),
        "eggs_lag4": _f(row.get("eggs_lag4"), 1),
        "temp_avg_lag1": _f(row.get("temp_avg_lag1"), 2),
        "precipitation_lag1": _f(row.get("precipitation_lag1"), 1),
        "feature_values": feature_values,
        "n_features": len(feature_cols),
        "best_iteration": int(
            getattr(model, "best_iteration_", 0) or 0
        ),
        "detailed_method": detailed_method,
        "explanation": (
            f"Model napove {pred:.1f} jajc za teden "
            f"{pd.Timestamp(row['week_start']).date()}–{pd.Timestamp(row['week_end']).date()}. "
            f"Sezonski baseline (eggs/hen za {rate_source} × {hens:.1f} kokoši × {n_days:.0f} dni) "
            f"je {base:.1f}. Razlika {delta:+.1f} jajc → score {score:+.2f} "
            f"({meaning_sl}). Petelin ({roosters:.0f}) ni v imenovalcu jajc."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next week egg production")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = predict_next_week(args.model)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Next-week forecast ({DEFAULT_USERNAME})")
        print(f"  Week:       {result['week_start']} .. {result['week_end']}")
        print(
            f"  Flock:      {result['hens_mean']:.1f} hens + "
            f"{result['roosters_mean']:.1f} rooster "
            f"(birds={result['birds_mean']:.1f})"
        )
        print(f"  Predicted:  {result['predicted_eggs']:.1f} eggs")
        print(f"  Baseline:   {result['seasonal_baseline_eggs']:.1f} eggs")
        print(
            f"  Delta:      {result['delta_eggs']:+.1f} "
            f"({result['delta_pct']:+.1f}%)"
        )
        print(f"  Score:      {result['score']:+.3f}  ({result['score_meaning_sl']})")
        print(f"  {result['explanation']}")


if __name__ == "__main__":
    main()
