"""
Build a daily analysis table for one flock user from farm.db + weather JSON.

Assumptions (documented for EDA / forecasting):
- Single username from config (default: `default`, override with FLOCK_USERNAME).
- `feeding_logs.quantity` is grams per day for every day in [from_date, to_date].
- If two ranges of the same food type overlap, the range with the later from_date wins.
- `feed_kg` = sum of daily grams / 1000.
- `hens` = hens present that day (type=hen, bought<=date, date_left is null or >= date).
- Roosters are counted separately; they affect feed (birds), never eggs/hen.
- Weather prefers yearly JSON files over incomplete DB daily_data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DAILY_RAW_CSV,
    DEFAULT_USERNAME,
    PROCESSED_DIR,
    RAW_DIR,
    db_path as default_db_path,
)

USERNAME = DEFAULT_USERNAME


def _parse_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def load_weather(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate yearly weather JSON files."""
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("20*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            continue
        frame = pd.DataFrame(records)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No weather JSON found in {raw_dir}")
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return weather.reset_index(drop=True)


def _user_id(conn: sqlite3.Connection, username: str = USERNAME) -> int:
    row = conn.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row is None:
        raise ValueError(f"User {username!r} not found")
    return int(row[0])


def _hens_by_day(
    conn: sqlite3.Connection, user_id: int, days: list[date]
) -> tuple[dict[date, int], dict[date, int]]:
    chickens = conn.execute(
        """
        SELECT type, date_bought, date_left
        FROM chickens
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()

    hens: dict[date, int] = {}
    roosters: dict[date, int] = {}
    for day in days:
        n_hens = 0
        n_roosters = 0
        for bird_type, bought, left in chickens:
            bought_d = _parse_date(bought)
            left_d = _parse_date(left)
            if bought_d is None or bought_d > day:
                continue
            if left_d is not None and left_d < day:
                continue
            if bird_type == "hen":
                n_hens += 1
            elif bird_type == "rooster":
                n_roosters += 1
        hens[day] = n_hens
        roosters[day] = n_roosters
    return hens, roosters


def _feed_grams_by_day(
    conn: sqlite3.Connection, user_id: int, days: list[date]
) -> dict[date, float]:
    """
    Expand feeding_logs ranges to daily grams.

    Overlap rule: per food_type_id, keep the covering log with the latest from_date.
    """
    logs = conn.execute(
        """
        SELECT food_type_id, quantity, from_date, to_date, updated_at
        FROM feeding_logs
        WHERE user_id = ?
        ORDER BY from_date
        """,
        (user_id,),
    ).fetchall()

    parsed = [
        {
            "food_type_id": int(food_type_id),
            "quantity": float(quantity or 0.0),
            "from_date": _parse_date(from_date),
            "to_date": _parse_date(to_date),
            "updated_at": str(updated_at or ""),
        }
        for food_type_id, quantity, from_date, to_date, updated_at in logs
        if from_date and to_date
    ]

    totals: dict[date, float] = {day: 0.0 for day in days}
    for day in days:
        by_food: dict[int, dict[str, Any]] = {}
        for log in parsed:
            if log["from_date"] <= day <= log["to_date"]:
                prev = by_food.get(log["food_type_id"])
                if prev is None or log["from_date"] > prev["from_date"]:
                    by_food[log["food_type_id"]] = log
                elif (
                    log["from_date"] == prev["from_date"]
                    and log["updated_at"] > prev["updated_at"]
                ):
                    by_food[log["food_type_id"]] = log
        totals[day] = sum(item["quantity"] for item in by_food.values())
    return totals


def _events_by_day(
    conn: sqlite3.Connection, user_id: int, days: list[date]
) -> dict[date, str]:
    rows = conn.execute(
        """
        SELECT date(ae.ts_start) AS d, aet.key, ae.title, ae.severity
        FROM activity_events ae
        JOIN activity_event_types aet ON aet.id = ae.event_type_id
        WHERE ae.user_id = ?
        ORDER BY ae.ts_start
        """,
        (user_id,),
    ).fetchall()

    day_set = set(days)
    events: dict[date, list[str]] = {day: [] for day in days}
    for d_str, key, title, severity in rows:
        d = _parse_date(d_str)
        if d is None or d not in day_set:
            continue
        label = key or "event"
        if title:
            label = f"{label}:{title}"
        if severity and severity not in ("info",):
            label = f"{label}[{severity}]"
        events[d].append(label)

    return {d: "|".join(parts) if parts else "" for d, parts in events.items()}


def build_daily_dataframe(
    db_path: Path | None = None,
    raw_dir: Path = RAW_DIR,
    username: str = USERNAME,
) -> pd.DataFrame:
    db_path = db_path or default_db_path(raw_dir)
    conn = sqlite3.connect(db_path)
    try:
        user_id = _user_id(conn, username)
        egg_rows = conn.execute(
            """
            SELECT date, total_eggs
            FROM eggs
            WHERE user_id = ?
            ORDER BY date
            """,
            (user_id,),
        ).fetchall()
        if not egg_rows:
            raise ValueError(f"No eggs for user {username!r}")

        egg_map = {_parse_date(d): int(eggs) for d, eggs in egg_rows}
        start = min(egg_map)
        end = max(egg_map)
        assert start is not None and end is not None
        days = _daterange(start, end)

        hens_map, rooster_map = _hens_by_day(conn, user_id, days)
        feed_map = _feed_grams_by_day(conn, user_id, days)
        events_map = _events_by_day(conn, user_id, days)
    finally:
        conn.close()

    weather = load_weather(raw_dir)
    weather_idx = weather.set_index("date")

    records: list[dict[str, Any]] = []
    for day in days:
        ts = pd.Timestamp(day)
        w = weather_idx.loc[ts] if ts in weather_idx.index else None
        records.append(
            {
                "date": ts,
                "eggs": egg_map.get(day),  # None if missing record
                "hens": hens_map[day],
                "roosters": rooster_map[day],
                "feed_g": feed_map[day],
                "feed_kg": feed_map[day] / 1000.0,
                "temp_min": None if w is None else w.get("temperature_min"),
                "temp_max": None if w is None else w.get("temperature_max"),
                "temp_avg": None if w is None else w.get("temperature_avg"),
                "precipitation": None if w is None else w.get("precipitation_mm"),
                "humidity_avg": None if w is None else w.get("humidity_avg"),
                "snow_depth_cm": None if w is None else w.get("snow_depth_cm"),
                "daylight_minutes": None if w is None else w.get("daylight_minutes"),
                "pressure_hpa": None if w is None else w.get("pressure_hpa"),
                "wind_speed_avg": None if w is None else w.get("wind_speed_avg"),
                "events": events_map[day],
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = build_daily_dataframe()
    raw_out = RAW_DIR / DAILY_RAW_CSV
    df.to_csv(raw_out, index=False)

    print(f"Wrote {raw_out} ({len(df)} rows)")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Missing eggs: {df['eggs'].isna().sum()}")
    print(f"Missing weather temp_avg: {df['temp_avg'].isna().sum()}")
    print(df[["eggs", "hens", "feed_kg", "temp_avg", "precipitation"]].describe())


if __name__ == "__main__":
    main()
