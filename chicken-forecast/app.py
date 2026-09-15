"""Simple Flask dashboard for weekly egg forecast results."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, send_file, send_from_directory, url_for

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.predict import predict_next_week  # noqa: E402
from src.analysis import build_analysis_context  # noqa: E402
from src.advanced_analysis import SECTIONS, load_summary  # noqa: E402
from src.config import (  # noqa: E402
    ANALYSIS_PDF,
    APP_BRAND,
    APP_NAME,
    APP_SECRET_KEY,
    DEFAULT_USERNAME,
    FORECAST_PDF,
)
from src.export_pdf_report import build_pdf  # noqa: E402
from src.export_pdf_forecast import build_forecast_pdf  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
)
app.secret_key = APP_SECRET_KEY

REPORTS = ROOT / "reports"
ADVANCED_REPORTS = ROOT / "reports" / "advanced"
MODELS = ROOT / "models"
PYTHON = Path(sys.executable)


@app.context_processor
def inject_helpers():
    """Always expose help dicts / branding to templates."""
    return {
        "feature_help": FEATURE_HELP,
        "app_name": APP_NAME,
        "app_brand": APP_BRAND,
        "flock_user": DEFAULT_USERNAME,
    }

FORECAST_CHARTS = [
    (
        "12_val_predictions.png",
        "Validacija: dejansko vs napoved",
        "Za vsak teden v validacijskem obdobju: modra = dejanska jajca, oranžna = LightGBM napoved, "
        "zelena črtkana = sezonski baseline. Manjša razlika med modro in oranžno = boljši model.",
    ),
    (
        "13_val_scores.png",
        "Validacijski score (−1 … +1)",
        "Za vsak val teden: kako daleč je napoved modela od sezonskega baseline-a. "
        "+1 = bistveno nad pričakovanim, 0 = normalno, −1 = bistveno pod. Ni napaka modela — "
        "je indikator pričakovane uspešnosti glede na sezono.",
    ),
    (
        "11_feature_importance.png",
        "Pomembnost featurejev",
        "Kateri vhodi LightGBM najbolj vplivajo na napoved. Višja vrstica = pogosteje uporabljen "
        "pri delitvah dreves. Tipično dominirajo pretekla jajca (lagi) in vreme prejšnjega tedna.",
    ),
]

ANALYSIS_CHARTS = [
    (
        "01_timeseries_production.png",
        "Dnevna produkcija",
        "Časovna vrsta: dnevna jajca, jajca/kokoš, kg hrane in število kokoši. "
        "Pokaže dolgoročni trend, padce po dogodkih in spremembe velikosti jate.",
    ),
    (
        "02_weekly_eggs.png",
        "Tedenska jajca",
        "Seštevek jajc po ISO tednih (pon–ned). Stabilnejši signal kot dnevni podatki — "
        "to je tudi tarča napovednega modela.",
    ),
    (
        "03_seasonality_month.png",
        "Sezonalnost po mesecih",
        "Povprečna jajca na kokoš na dan po mesecu. Visoko = dobra sezona nesnosti, "
        "nizko = šibkejša (npr. jesen). Baseline score uporablja podobno sezonsko logiko.",
    ),
    (
        "04_boxplot_eggs_month.png",
        "Porazdelitev jajc po mesecih",
        "Boxplot dnevnih jajc po mesecu: mediana, kvartili in outlierji. "
        "Pokaže, kako stabilna je produkcija znotraj meseca.",
    ),
    (
        "05_weather_scatter.png",
        "Vreme vs jajca/kokoš",
        "Točkovni diagram: temperatura (in svetloba/padavine) proti eggs/hen. "
        "Vsaka točka = en dan.",
    ),
    (
        "06_correlation_heatmap.png",
        "Korelacijska matrika",
        "Linearna povezanost med spremenljivkami (−1 do +1). Rdeče = pozitivna, modro = negativna. "
        "Pomaga videti, kaj gre skupaj z jajci/kokoš (temp, hrana, dan v letu …).",
    ),
    (
        "07_event_impact.png",
        "Vpliv dogodkov (−7 … +14 dni)",
        "Povprečna eggs/hen okoli dogodkov. Dan 0 = dogodek. Padec po dodajanju jate je pričakovan "
        "(mlade kokoši).",
    ),
    (
        "08_event_counts.png",
        "Število dogodkov",
        "Kolikokrat se je vsaka kategorija dogodka pojavila v obdobju.",
    ),
    (
        "09_anomalies.png",
        "Anomalije",
        "Dnevi, kjer je eggs/hen močno odstopal od 28-dnevnega baseline-a (|z| > 2.5). "
        "Rdeče = nenavadno nizko, zeleno = nenavadno visoko.",
    ),
    (
        "10_feed_conversion.png",
        "Konverzija hrane",
        "kg hrane na jajce skozi čas. Nižje = učinkovitejša jata (hrana vključuje petelina).",
    ),
]

# back-compat alias used by older snippets
CHARTS = FORECAST_CHARTS + ANALYSIS_CHARTS

FEATURE_HELP = {
    "eggs_lag1": "Jajca prejšnji teden — najmočnejši kratek signal.",
    "eggs_lag2": "Jajca pred 2 tednoma.",
    "eggs_lag3": "Jajca pred 3 tedni.",
    "eggs_lag4": "Jajca pred 4 tedni (mesečni ritm).",
    "eggs_per_hen_day_lag1": "Jajca/kokoš/dan prejšnji teden (normalizirano na jato).",
    "eggs_per_hen_day_lag2": "Jajca/kokoš/dan pred 2 tednoma.",
    "precipitation_lag1": "Padavine prejšnji teden (mm).",
    "humidity_avg_lag1": "Povprečna vlaga prejšnji teden.",
    "temp_avg_lag1": "Povprečna temperatura prejšnji teden.",
    "temp_min_lag1": "Minimalna temperatura prejšnji teden.",
    "temp_max_lag1": "Maksimalna temperatura prejšnji teden.",
    "daylight_minutes_lag1": "Dnevna svetloba (min) prejšnji teden.",
    "feed_kg_lag1": "Skupna hrana (kg) prejšnji teden.",
    "feed_per_hen_day_kg_lag1": "Hrana na žival/dan prejšnji teden.",
    "feed_per_bird_day_kg_lag1": "Hrana na žival/dan prejšnji teden.",
    "hens_mean": "Povprečno število kokoši v tednu (brez petelina).",
    "hens_min": "Najmanj kokoši v tednu.",
    "hens_max": "Največ kokoši v tednu.",
    "roosters_mean": "Petelini (samo kontekst / hrana, ne jajca).",
    "month": "Mesec (sezonalnost).",
    "week_of_year": "Številka tedna v letu.",
    "year": "Leto.",
    "n_days": "Število dni v tednu (običajno 7).",
    "event_days_lag1": "Število dni z dogodki prejšnji teden.",
    "event_death_lag1": "Smrti v jati prejšnji teden.",
    "event_added_lag1": "Dodane živali prejšnji teden.",
    "event_feed_change_lag1": "Spremembe hrane prejšnji teden.",
    "event_egg_anomaly_lag1": "Označene anomalije jajc prejšnji teden.",
}

ACTIONS = {
    "rebuild": ("src.build_daily_dataset", "Dataset obnovljen iz DB + vremena"),
    "eda": ("src.run_eda", "EDA končana (grafi + summary)"),
    "train": ("src.train", "Model treniran"),
    "advanced": ("src.advanced_analysis", "Napredne analize (7 metod) izračunane"),
}


def _load_metrics() -> dict:
    path = MODELS / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_module(module: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [str(PYTHON), "-m", module],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(out.strip().splitlines()[-12:]) if out.strip() else "(no output)"
        if proc.returncode != 0:
            return False, f"{module} failed (code {proc.returncode}):\n{tail}"
        return True, tail
    except subprocess.TimeoutExpired:
        return False, f"{module} timeout (>10 min)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{module} error: {exc}"


def _charts(spec: list[tuple[str, str, str]]) -> list[dict]:
    return [
        {"name": name, "title": title, "help": help_text}
        for name, title, help_text in spec
        if (REPORTS / name).exists()
    ]


@app.route("/napoved/pdf")
def napoved_pdf():
    """Generate forecast PDF with all dashboard fields + detailed method."""
    try:
        path = build_forecast_pdf()
    except Exception as exc:  # noqa: BLE001
        flash(f"PDF napaka: {exc}", "error")
        return redirect(url_for("index"))
    return send_file(
        path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=FORECAST_PDF,
    )


@app.route("/")
def index():
    error = None
    forecast = None
    try:
        forecast = predict_next_week()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    metrics = _load_metrics()
    return render_template(
        "index.html",
        forecast=forecast,
        metrics=metrics,
        charts=_charts(FORECAST_CHARTS),
        feature_help=FEATURE_HELP,
        error=error,
    )


@app.route("/analiza")
def analiza():
    error = None
    summary = None
    try:
        summary = load_summary()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return render_template(
        "analiza_hub.html",
        sections=SECTIONS,
        summary=summary,
        error=error,
    )


@app.route("/analiza/pregled")
def analiza_overview():
    error = None
    analysis = None
    try:
        analysis = build_analysis_context()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return render_template(
        "analiza.html",
        analysis=analysis,
        charts=_charts(ANALYSIS_CHARTS),
        error=error,
    )


@app.route("/analiza/pdf")
def analiza_pdf():
    """Generate (or regenerate) the full analysis PDF and download it."""
    from flask import request

    refresh = request.args.get("refresh", "0") in {"1", "true", "yes"}
    try:
        path = build_pdf(refresh_advanced=refresh)
    except Exception as exc:  # noqa: BLE001
        flash(f"PDF napaka: {exc}", "error")
        return redirect(url_for("analiza"))
    return send_file(
        path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=ANALYSIS_PDF,
    )


@app.route("/analiza/<slug>")
def analiza_section(slug: str):
    meta = next((s for s in SECTIONS if s["slug"] == slug), None)
    if meta is None:
        abort(404)

    error = None
    section = None
    summary = None
    try:
        summary = load_summary()
        section = summary.get("sections", {}).get(slug)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    idx = next(i for i, s in enumerate(SECTIONS) if s["slug"] == slug)
    prev_s = SECTIONS[idx - 1] if idx > 0 else None
    next_s = SECTIONS[idx + 1] if idx < len(SECTIONS) - 1 else None

    return render_template(
        "analiza_section.html",
        meta=meta,
        section=section,
        sections=SECTIONS,
        prev=prev_s,
        next_section=next_s,
        error=error,
    )


@app.post("/run/<action>")
def run_action(action: str):
    if action == "refresh":
        flash("Napoved osvežena.", "ok")
        return redirect(url_for("index"))

    redirect_map = {
        "eda": "analiza_overview",
        "advanced": "analiza",
        "rebuild": "index",
        "train": "index",
        "all": "index",
    }

    if action == "all":
        steps = ["rebuild", "eda", "train", "advanced"]
        messages = []
        for step in steps:
            module, label = ACTIONS[step]
            ok, detail = _run_module(module)
            if not ok:
                flash(f"{label} — NAPAKA\n{detail}", "error")
                return redirect(url_for("index"))
            messages.append(f"✓ {label}")
        flash("Vse koraki OK:\n" + "\n".join(messages), "ok")
        return redirect(url_for("index"))

    if action not in ACTIONS:
        abort(404)

    module, label = ACTIONS[action]
    ok, detail = _run_module(module)
    if ok:
        flash(f"{label}\n{detail}", "ok")
    else:
        flash(detail, "error")
    return redirect(url_for(redirect_map.get(action, "index")))


@app.route("/reports/<path:filename>")
def report_file(filename: str):
    path = REPORTS / filename
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(REPORTS, filename)


@app.route("/reports/advanced/<path:filename>")
def advanced_file(filename: str):
    path = ADVANCED_REPORTS / filename
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(ADVANCED_REPORTS, filename)


def main() -> None:
    # use_reloader=False avoids double-spawn issues with long train jobs from buttons
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
