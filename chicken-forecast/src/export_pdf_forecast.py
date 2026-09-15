"""PDF report for the forecast dashboard page (all on-page data + method)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import APP_BRAND, APP_NAME, FORECAST_PDF  # noqa: E402
from src.predict import predict_next_week  # noqa: E402

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
OUT_PDF = REPORTS / FORECAST_PDF

FORECAST_CHARTS = [
    ("12_val_predictions.png", "Validacija: dejansko vs napoved"),
    ("13_val_scores.png", "Validacijski score (−1 … +1)"),
    ("11_feature_importance.png", "Pomembnost featurejev"),
]


def _register_fonts() -> tuple[str, str]:
    candidates = [
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf"), "Arial", "Arial-Bold"),
        (Path(r"C:\Windows\Fonts\calibri.ttf"), Path(r"C:\Windows\Fonts\calibrib.ttf"), "Calibri", "Calibri-Bold"),
    ]
    for regular, bold, name, bold_name in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont(name, str(regular)))
            pdfmetrics.registerFont(TTFont(bold_name, str(bold)))
            return name, bold_name
    return "Helvetica", "Helvetica-Bold"


def _styles(font: str, font_bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FT", parent=base["Title"], fontName=font_bold, fontSize=18, leading=22,
            alignment=TA_CENTER, spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "FH1", parent=base["Heading1"], fontName=font_bold, fontSize=13, leading=16,
            spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1c2418"),
        ),
        "h2": ParagraphStyle(
            "FH2", parent=base["Heading2"], fontName=font_bold, fontSize=11, leading=14,
            spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2f6b3a"),
        ),
        "body": ParagraphStyle(
            "FB", parent=base["BodyText"], fontName=font, fontSize=9, leading=12, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "FS", parent=base["BodyText"], fontName=font, fontSize=8, leading=10,
            textColor=colors.HexColor("#5c6b52"), spaceAfter=3,
        ),
        "center": ParagraphStyle(
            "FC", parent=base["BodyText"], fontName=font, fontSize=10, leading=13,
            alignment=TA_CENTER,
        ),
    }


def _esc(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _kv_table(rows: list[tuple[str, Any]], font: str, font_bold: str) -> Table:
    data = [
        [
            Paragraph(f"<b>{_esc(k)}</b>", ParagraphStyle("k", fontName=font_bold, fontSize=8, leading=10)),
            Paragraph(_esc(v), ParagraphStyle("v", fontName=font, fontSize=8, leading=10)),
        ]
        for k, v in rows
    ]
    t = Table(data, colWidths=[6.2 * cm, 10.8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f7f3ea")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5cdb8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _dict_table(rows: list[dict], font: str) -> Table | Paragraph:
    if not rows:
        return Paragraph("(prazno)", ParagraphStyle("e", fontName=font, fontSize=8))
    cols = list(rows[0].keys())
    header = [
        Paragraph(f"<b>{_esc(c)}</b>", ParagraphStyle("th", fontName=font, fontSize=7, leading=9))
        for c in cols
    ]
    body = [
        [
            Paragraph(_esc(row.get(c, "")), ParagraphStyle("td", fontName=font, fontSize=7, leading=9))
            for c in cols
        ]
        for row in rows
    ]
    width = 17.0 * cm
    col_w = width / max(len(cols), 1)
    t = Table([header] + body, colWidths=[col_w] * len(cols), repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcead9")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5cdb8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faf7f0")]),
            ]
        )
    )
    return t


def _add_image(story: list, path: Path, caption: str, styles: dict) -> None:
    if not path.exists():
        story.append(Paragraph(f"[Manjka graf: {_esc(path.name)}]", styles["small"]))
        return
    max_w, max_h = 17 * cm, 10 * cm
    img = Image(str(path))
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    story.append(KeepTogether([img, Paragraph(_esc(caption), styles["small"]), Spacer(1, 0.2 * cm)]))


def _bullets(items: list[str], styles: dict) -> ListFlowable:
    return ListFlowable(
        [
            ListItem(
                Paragraph(_esc(i), styles["body"]),
                leftIndent=8,
                bulletColor=colors.HexColor("#2f6b3a"),
            )
            for i in items
            if i.strip()
        ],
        bulletType="bullet",
        start="•",
    )


def _chunked_pre(text: str, styles: dict, chunk: int = 3500) -> list:
    parts = []
    for i in range(0, len(text), chunk):
        parts.append(
            Paragraph(
                f"<font face='Courier' size='5.5'>{_esc(text[i:i+chunk])}</font>",
                styles["small"],
            )
        )
    return parts


def _load_metrics() -> dict:
    path = MODELS / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_forecast_pdf(out_path: Path | None = None) -> Path:
    out_path = out_path or OUT_PDF
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font, font_bold = _register_fonts()
    styles = _styles(font, font_bold)
    forecast = predict_next_week()
    metrics = _load_metrics()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=f"{APP_NAME} – napoved jajc",
        author="chicken-forecast",
    )
    story: list = []

    # Cover
    story.append(Paragraph(APP_BRAND, styles["center"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Tedenska napoved jajc – poročilo", styles["title"]))
    story.append(
        Paragraph(
            f"Napovedani teden: {_esc(forecast['week_start'])} – {_esc(forecast['week_end'])}",
            styles["center"],
        )
    )
    story.append(
        Paragraph(
            f"<b>{_esc(forecast['predicted_eggs'])} jajc</b> · baseline "
            f"{_esc(forecast['seasonal_baseline_eggs'])} · score "
            f"{forecast['score']:+.3f} ({_esc(forecast['score_meaning_sl'])})",
            styles["center"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(_esc(forecast["explanation"]), styles["body"]))
    story.append(PageBreak())

    # Detailed method
    story.append(Paragraph("1. Kako smo prišli do napovedi", styles["h1"]))
    story.append(
        Paragraph(
            "Spodaj je natančen opis pipeline-a od podatkov do številke na dashboardu.",
            styles["body"],
        )
    )
    for line in forecast.get("detailed_method", []):
        if not line:
            story.append(Spacer(1, 0.15 * cm))
        else:
            story.append(Paragraph(_esc(line), styles["body"]))

    # Forecast block (all fields from page)
    story.append(Paragraph("2. Napoved – vsa polja s strani", styles["h1"]))
    story.append(
        _kv_table(
            [
                ("Teden od", forecast["week_start"]),
                ("Teden do", forecast["week_end"]),
                ("Week of year", forecast["week_of_year"]),
                ("Mesec / leto", f"{forecast['month']} / {forecast['year']}"),
                ("n_days", forecast["n_days"]),
                ("Napovedana jajca", forecast["predicted_eggs"]),
                ("Napoved / dan", forecast["predicted_per_day"]),
                ("Napoved eggs/hen/dan", forecast["predicted_per_hen_day"]),
                ("Sezonski baseline jajc", forecast["seasonal_baseline_eggs"]),
                ("Baseline eggs/hen/dan", forecast["baseline_per_hen_day"]),
                ("Baseline rate (eph)", forecast.get("baseline_rate_eph")),
                ("Baseline rate vir", forecast.get("baseline_rate_source")),
                ("Δ vs baseline (jajca)", forecast["delta_eggs"]),
                ("Δ vs baseline (%)", forecast["delta_pct"]),
                ("Score", forecast["score"]),
                ("Score pomen", forecast["score_meaning_sl"]),
                ("Score skala", forecast["score_scale"]),
                ("residual_std (train)", forecast["residual_std"]),
                ("Kokoši", forecast["hens_mean"]),
                ("Petelin", forecast["roosters_mean"]),
                ("Živali (hrana)", forecast["birds_mean"]),
                ("Opomba jata", forecast["note_flock"]),
                ("Zadnji teden", f"{forecast['last_week_start']} – {forecast['last_week_end']}"),
                ("Zadnji teden jajca", forecast["last_week_eggs"]),
                ("Zadnji teden kokoši", forecast["last_week_hens"]),
                ("Zadnji teden hrana kg", forecast["last_week_feed_kg"]),
                ("eggs_lag1", forecast["eggs_lag1"]),
                ("eggs_lag2", forecast["eggs_lag2"]),
                ("eggs_lag3", forecast["eggs_lag3"]),
                ("eggs_lag4", forecast["eggs_lag4"]),
                ("temp_avg_lag1", forecast["temp_avg_lag1"]),
                ("precipitation_lag1", forecast["precipitation_lag1"]),
                ("Št. featurejev modela", forecast.get("n_features")),
                ("LightGBM best_iteration", forecast.get("best_iteration")),
            ],
            font,
            font_bold,
        )
    )

    story.append(Paragraph("2.1 Vsi featureji poslani v LightGBM", styles["h2"]))
    story.append(_dict_table(forecast.get("feature_values") or [], font))

    # Metrics
    story.append(Paragraph("3. Validacija modela (vsa polja)", styles["h1"]))
    if not metrics:
        story.append(Paragraph("metrics.json ni na voljo — poženi train.", styles["body"]))
    else:
        story.append(
            Paragraph(
                "Model se uči na starejših tednih, nato preverimo zadnje tedne s časovnim splitom "
                "(brez naključnega mešanja). Manjši MAE/RMSE = boljša napoved. Če je LightGBM MAE "
                "manjši od Baseline MAE, model premaga golo sezono.",
                styles["body"],
            )
        )
        story.append(
            _kv_table(
                [
                    ("Train MAE", metrics.get("train", {}).get("mae")),
                    ("Train RMSE", metrics.get("train", {}).get("rmse")),
                    ("Train R²", metrics.get("train", {}).get("r2")),
                    ("Train n", metrics.get("train", {}).get("n")),
                    ("Val MAE", metrics.get("val", {}).get("mae")),
                    ("Val RMSE", metrics.get("val", {}).get("rmse")),
                    ("Val R²", metrics.get("val", {}).get("r2")),
                    ("Val n", metrics.get("val", {}).get("n")),
                    ("Baseline Val MAE", metrics.get("val_seasonal_baseline", {}).get("mae")),
                    ("Baseline Val RMSE", metrics.get("val_seasonal_baseline", {}).get("rmse")),
                    ("Baseline Val R²", metrics.get("val_seasonal_baseline", {}).get("r2")),
                    ("Mean val score", metrics.get("val_score_mean")),
                    ("Train tedni", metrics.get("train_weeks")),
                    ("Val tedni", metrics.get("val_weeks")),
                    ("Train od–do", f"{metrics.get('train_start')} → {metrics.get('train_end')}"),
                    ("Val od–do", f"{metrics.get('val_start')} → {metrics.get('val_end')}"),
                    ("best_iteration", metrics.get("best_iteration")),
                ],
                font,
                font_bold,
            )
        )
        story.append(Paragraph("3.1 Top featureji (importance)", styles["h2"]))
        story.append(_dict_table(metrics.get("top_features") or [], font))
        story.append(
            Paragraph(
                "Številka importance = kako pogosto LightGBM uporabi feature pri delitvah dreves "
                "(ni enota jajc).",
                styles["small"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("4. Grafi napovedi / validacije", styles["h1"]))
    for name, title in FORECAST_CHARTS:
        _add_image(story, REPORTS / name, title, styles)

    story.append(Paragraph("5. Celoten forecast JSON (dobesedno)", styles["h1"]))
    # drop detailed_method duplication noise is ok - include everything
    story.extend(_chunked_pre(json.dumps(forecast, ensure_ascii=False, indent=2), styles))

    story.append(Paragraph("6. Celoten metrics.json (dobesedno)", styles["h1"]))
    story.extend(_chunked_pre(json.dumps(metrics, ensure_ascii=False, indent=2), styles))

    doc.build(story)
    return out_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = build_forecast_pdf()
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
