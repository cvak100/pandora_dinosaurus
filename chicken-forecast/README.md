# Chicken Forecast

End-to-end analysis and **weekly egg-production forecasting** for a backyard flock, built on farm logs (SQLite) and local weather.

Includes a small **Flask dashboard** (Slovenian UI): next-week forecast, EDA charts, advanced analytics, and PDF export.

### Related: Pandora / Dino (data source)

This project reads the **SQLite farm database** produced by the Pandora **Dino** module — day-to-day flock ops (eggs, feed, events, weather sync, etc.).

- Public Dino / Pandora showcase (schema, models, prediction engine):  
  **[cvak100/pandora_public](https://github.com/cvak100/pandora_public)**
- Export or copy that app’s DB locally as `chicken-forecast/data/raw/farm.db` (never commit real dumps).

Chicken Forecast is a separate ML / analytics layer on top of that data (LightGBM weekly forecast + EDA dashboard), not a fork of Dino’s heuristic engine.

---

## What it does

| Area | Description |
|------|-------------|
| **Dataset** | Builds a daily table from `farm.db` + yearly weather JSON (eggs, hens, feed, events, weather) |
| **EDA** | Trends, seasonality, feed conversion, weather correlations, event windows, anomalies → plots + markdown summary |
| **Forecast** | LightGBM predicts **total eggs next ISO week**, plus a **score in [−1, +1]** vs a seasonal baseline |
| **Dashboard** | Local web app: forecast, analysis hub, one-click rebuild / EDA / train, PDF downloads |
| **Advanced analysis** | Change points, event impact, STL residuals, week clustering, efficiency frontier, interactions, Isolation Forest / LOF anomalies |

**Validation snapshot** (time split, ~63 train / 16 val weeks): LightGBM val MAE ≈ **5.6** eggs/week vs seasonal baseline ≈ **7.9**.

---

## Quick start

**Requirements:** Python 3.10+ (tested on 3.13), Windows / macOS / Linux.

```bash
cd chicken-forecast
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### 1. Put raw data in place

```text
chicken-forecast/data/raw/
├── farm.db         # SQLite flock export (gitignored)
├── 2025.json       # daily weather
└── 2026.json
```

Defaults (override with env vars):

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLOCK_USERNAME` | `default` | Username row in the DB to analyse |
| `FLOCK_DB` | `farm.db` | Filename under `data/raw/` |
| `FLASK_SECRET_KEY` | `chicken-forecast-local` | Flask session secret |

Example (Windows PowerShell):

```powershell
$env:FLOCK_USERNAME = "default"
$env:FLOCK_DB = "farm.db"
```

### 2. Build → explore → train → predict

```bash
python -m src.build_daily_dataset   # data/raw → data/processed
python -m src.run_eda               # reports/*.png + eda_summary.md
python -m src.train                 # models/weekly_lgbm.joblib + metrics
python -m src.predict               # next-week eggs + score
python -m src.advanced_analysis     # reports/advanced/*
```

### 3. Open the dashboard

```bash
python app.py
# http://127.0.0.1:5001          → napoved (forecast)
# http://127.0.0.1:5001/analiza  → analysis hub
```

From the UI you can rebuild the dataset, re-run EDA/training, and download:

- `napoved_complete.pdf` — forecast report  
- `analiza_complete.pdf` — full analysis report  

Optional notebooks: `notebooks/01_eda.ipynb`, `notebooks/02_forecasting.ipynb`.

---

## How forecasting works

- **Target:** sum of eggs in the next ISO week (Mon–Sun).
- **Features:** hen counts, calendar (month / week / year), egg & weather **lags 1–4 weeks** (no same-week weather leak), feed and event lags.
- **Score:** `(prediction − seasonal_baseline) / scale`, clipped to **[−1, +1]**  
  (`+1` = well above seasonal expectation, `0` ≈ normal, `−1` = well below).
- **Artifacts:** `models/weekly_lgbm.joblib`, `models/metrics.json`, `reports/forecast_eval.md`.

Eggs are attributed to **hens only**. Feed includes hens **and** roosters.

---

## Data assumptions

Documented in code (`src/build_daily_dataset.py`); summary:

- Pipeline is scoped to one flock user (`FLOCK_USERNAME`, default `default`).
- `feeding_logs.quantity` = **grams per day** for every day in `[from_date, to_date]`. Same food-type overlap → later `from_date` wins. `feed_kg = grams / 1000`.
- A hen is present on day *d* if `type=hen`, `date_bought ≤ d`, and `date_left` is null or `≥ d`.
- Weather prefers yearly JSON files over incomplete DB weather tables.

---

## Project layout

```text
chicken-forecast/
├── app.py                 # Flask dashboard
├── requirements.txt
├── data/
│   ├── raw/               # farm.db, weather JSON (local)
│   └── processed/         # daily / weekly CSVs
├── models/                # trained model + metrics.json
├── notebooks/
├── reports/               # EDA & forecast plots, PDFs, summaries
│   └── advanced/          # advanced analysis outputs
├── src/
│   ├── config.py          # defaults + branding
│   ├── build_daily_dataset.py
│   ├── run_eda.py
│   ├── train.py
│   ├── predict.py
│   ├── features.py
│   ├── analysis.py
│   ├── advanced_analysis.py
│   ├── export_pdf_report.py
│   └── export_pdf_forecast.py
└── web/
    ├── templates/
    └── static/
```

---

## Stack

Python · pandas · LightGBM · scikit-learn · statsmodels · ruptures · matplotlib / seaborn · Flask · ReportLab

---

## Privacy before a public release

- Do **not** commit `farm.db` / SQLite dumps, personal names, or full farm exports.
- Prefer anonymized / sample CSVs in `data/` if you ship demo data.
- Set `FLASK_SECRET_KEY` if you ever expose the app beyond localhost.
- Treat weather and flock IDs as potentially identifying if combined with a location.

---

## License

Add a license before publishing (e.g. MIT). Until then, all rights reserved by the author.
