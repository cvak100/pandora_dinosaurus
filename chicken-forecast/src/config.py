"""Shared defaults for flock pipeline + dashboard branding."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Override locally without hardcoding personal names:
#   set FLOCK_USERNAME=myuser
#   set FLOCK_DB=my_export.db
DEFAULT_USERNAME = os.environ.get("FLOCK_USERNAME", "default")
DEFAULT_DB_FILENAME = os.environ.get("FLOCK_DB", "farm.db")

APP_NAME = "Chicken Forecast"
APP_BRAND = "Chicken Forecast"
APP_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "chicken-forecast-local")

# Output file stems (no personal names)
DAILY_RAW_CSV = "daily_flock.csv"
DAILY_CLEAN_CSV = "daily_flock_clean.csv"
WEEKLY_CSV = "weekly_flock.csv"
WEEKLY_MODEL_CSV = "weekly_flock_model.csv"
FORECAST_PDF = "napoved_complete.pdf"
ANALYSIS_PDF = "analiza_complete.pdf"


def db_path(raw_dir: Path | None = None) -> Path:
    return (raw_dir or RAW_DIR) / DEFAULT_DB_FILENAME
