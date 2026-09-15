"""
Build a full PDF report: historical overview + all 7 advanced analyses.

Literally includes all KPIs, tables, findings, notes and charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

from src.advanced_analysis import (  # noqa: E402
    SECTIONS,
    load_summary,
)
from src.analysis import build_analysis_context  # noqa: E402
from src.config import ANALYSIS_PDF, APP_BRAND, APP_NAME, DEFAULT_USERNAME  # noqa: E402

REPORTS = ROOT / "reports"
ADVANCED = REPORTS / "advanced"
OUT_PDF = REPORTS / ANALYSIS_PDF

# EDA chart specs (same as Flask ANALYSIS_CHARTS filenames)
EDA_CHARTS = [
    ("01_timeseries_production.png", "Dnevna produkcija"),
    ("02_weekly_eggs.png", "Tedenska jajca"),
    ("03_seasonality_month.png", "Sezonalnost po mesecih"),
    ("04_boxplot_eggs_month.png", "Porazdelitev jajc po mesecih"),
    ("05_weather_scatter.png", "Vreme vs jajca/kokoš"),
    ("06_correlation_heatmap.png", "Korelacijska matrika"),
    ("07_event_impact.png", "Vpliv dogodkov"),
    ("08_event_counts.png", "Število dogodkov"),
    ("09_anomalies.png", "Anomalije (z-score)"),
    ("10_feed_conversion.png", "Konverzija hrane"),
]


def _register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            "Arial",
            "Arial-Bold",
        ),
        (
            Path(r"C:\Windows\Fonts\calibri.ttf"),
            Path(r"C:\Windows\Fonts\calibrib.ttf"),
            "Calibri",
            "Calibri-Bold",
        ),
        (
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
            "DejaVu",
            "DejaVu-Bold",
        ),
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
            "T",
            parent=base["Title"],
            fontName=font_bold,
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=font_bold,
            fontSize=14,
            leading=18,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1c2418"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=font_bold,
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2f6b3a"),
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9,
            leading=12,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "S",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#5c6b52"),
            spaceAfter=3,
        ),
        "center": ParagraphStyle(
            "C",
            parent=base["BodyText"],
            fontName=font,
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
        ),
    }


def _esc(text: Any) -> str:
    s = "" if text is None else str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _chunked_pre(text: str, styles: dict, chunk: int = 3500) -> list:
    """Split huge JSON into multiple paragraphs (ReportLab limit-friendly)."""
    parts = []
    for i in range(0, len(text), chunk):
        parts.append(
            Paragraph(
                f"<font face='Courier' size='5.5'>{_esc(text[i:i+chunk])}</font>",
                styles["small"],
            )
        )
    return parts


def _kv_table(rows: list[tuple[str, Any]], font: str, font_bold: str) -> Table:
    data = [[Paragraph(f"<b>{_esc(k)}</b>", ParagraphStyle("k", fontName=font_bold, fontSize=8, leading=10)),
             Paragraph(_esc(v), ParagraphStyle("v", fontName=font, fontSize=8, leading=10))]
            for k, v in rows]
    t = Table(data, colWidths=[6.5 * cm, 10.5 * cm])
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


def _dict_table(rows: list[dict], font: str, styles: dict) -> Table | Paragraph:
    if not rows:
        return Paragraph("(prazno)", styles["small"])
    cols = list(rows[0].keys())
    header = [
        Paragraph(f"<b>{_esc(c)}</b>", ParagraphStyle("th", fontName=font, fontSize=7, leading=9))
        for c in cols
    ]
    body = []
    for row in rows:
        body.append(
            [
                Paragraph(_esc(row.get(c, "")), ParagraphStyle("td", fontName=font, fontSize=7, leading=9))
                for c in cols
            ]
        )
    data = [header] + body
    # fit width
    width = 17.0 * cm
    col_w = width / max(len(cols), 1)
    t = Table(data, colWidths=[col_w] * len(cols), repeatRows=1)
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
    # Fit to page width
    max_w = 17 * cm
    max_h = 10 * cm
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_w / iw, max_h / ih, 1.0)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    story.append(KeepTogether([img, Paragraph(_esc(caption), styles["small"]), Spacer(1, 0.25 * cm)]))


def _bullet_list(items: list[str], styles: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_esc(i), styles["body"]), leftIndent=8, bulletColor=colors.HexColor("#2f6b3a"))
         for i in items],
        bulletType="bullet",
        start="•",
    )


def build_pdf(out_path: Path | None = None, refresh_advanced: bool = False) -> Path:
    out_path = out_path or OUT_PDF
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if refresh_advanced:
        from src.advanced_analysis import run_all

        run_all()

    font, font_bold = _register_fonts()
    styles = _styles(font, font_bold)
    overview = build_analysis_context()
    advanced = load_summary()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=f"{APP_NAME} – popolno analitično poročilo",
        author="chicken-forecast",
    )
    story: list = []

    # ----- Cover -----
    story.append(Paragraph(APP_BRAND, styles["center"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Popolno analitično poročilo", styles["title"]))
    story.append(
        Paragraph(
            f"Obdobje: {_esc(overview['period_start'])} → {_esc(overview['period_end'])} "
            f"({overview['n_days']} dni ≈ {overview['n_years']} let)",
            styles["center"],
        )
    )
    story.append(
        Paragraph(
            "Vsebina: Zgodovinska analiza (pregled) + 7 naprednih analiz "
            "(change points, event impact, decomposition, clustering, efficiency, "
            "interactions, anomalies). Poročilo vsebuje DOBESEDNO vse KPI, tabele, "
            "ugotovitve, metode in grafe.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            f"Generirano: {_esc(advanced.get('generated_at', 'n/a'))}",
            styles["small"],
        )
    )
    story.append(PageBreak())

    # ----- Overview -----
    story.append(Paragraph("1. Zgodovinska analiza (pregled)", styles["h1"]))
    story.append(
        Paragraph(
            f"Povzetek celotnega obdobja za uporabnika {DEFAULT_USERNAME}. "
            "Jajca/kokoš: samo kokoši. Hrana: kokoši + petelin.",
            styles["body"],
        )
    )

    story.append(Paragraph("1.1 Ključni kazalniki", styles["h2"]))
    kv = [
        ("Obdobje od", overview["period_start"]),
        ("Obdobje do", overview["period_end"]),
        ("Št. dni", overview["n_days"]),
        ("Št. let", overview["n_years"]),
        ("Manjkajoči egg zapisi", overview["missing_eggs"]),
        ("Skupaj jajc", overview["total_eggs"]),
        ("Povprečje jajc/dan", overview["mean_eggs"]),
        ("Mediana jajc/dan", overview["median_eggs"]),
        ("Max jajc/dan", overview["max_eggs"]),
        ("Povprečje kokoši", overview["mean_hens"]),
        ("Min–max kokoši", f"{overview['min_hens']}–{overview['max_hens']}"),
        ("Povprečje petelinov", overview["mean_roosters"]),
        ("Eggs / kokoš / dan", overview["eggs_per_hen"]),
        ("Hrana kg/dan", overview["feed_kg_day"]),
        ("Hrana g/žival/dan", overview["feed_g_per_bird"]),
        ("kg hrane / jajce", overview["feed_per_egg_kg"]),
        ("Št. tednov", overview["n_weeks"]),
        ("Povprečje jajc/teden", overview["weekly_eggs_mean"]),
        ("Najboljši mesec", f"{overview['best_month']} ({overview['best_eph']})"),
        ("Najslabši mesec", f"{overview['worst_month']} ({overview['worst_eph']})"),
        ("Anomalije nizko", overview["anomalies_low"]),
        ("Anomalije visoko", overview["anomalies_high"]),
    ]
    story.append(_kv_table(kv, font, font_bold))

    story.append(Paragraph("1.2 Sezonalnost po mesecih (vse vrstice)", styles["h2"]))
    story.append(_dict_table(overview["monthly"], font, styles))

    story.append(Paragraph("1.3 Vreme – korelacije in ekstremi", styles["h2"]))
    corr_rows = [(f"corr({k})", v) for k, v in overview.get("corr", {}).items()]
    ext = overview.get("extreme_temp", {})
    corr_rows += [
        ("eggs/hen hladnih 10%", ext.get("cold")),
        ("eggs/hen sredina", ext.get("mid")),
        ("eggs/hen toplih 10%", ext.get("hot")),
    ]
    story.append(_kv_table(corr_rows, font, font_bold))

    story.append(Paragraph("1.4 Vpliv dogodkov (pred 7d / po 14d) – vse vrstice", styles["h2"]))
    story.append(_dict_table(overview["events"], font, styles))

    story.append(Paragraph("1.5 Glavni zaključki", styles["h2"]))
    story.append(_bullet_list(overview["takeaways"], styles))

    story.append(Paragraph("1.6 Celoten overview JSON (dobesedno)", styles["h2"]))
    story.extend(
        _chunked_pre(json.dumps(overview, ensure_ascii=False, indent=2), styles)
    )

    story.append(Paragraph("1.7 EDA grafi", styles["h2"]))
    for name, title in EDA_CHARTS:
        _add_image(story, REPORTS / name, title, styles)

    story.append(PageBreak())

    # ----- 7 advanced sections -----
    story.append(Paragraph("2. Napredne analize (7 metod)", styles["h1"]))
    story.append(
        Paragraph(
            f"Advanced summary obdobje: {_esc(advanced.get('period_start'))} → "
            f"{_esc(advanced.get('period_end'))}. "
            f"generated_at={_esc(advanced.get('generated_at'))}",
            styles["body"],
        )
    )

    for i, meta in enumerate(SECTIONS, start=1):
        slug = meta["slug"]
        sec = advanced.get("sections", {}).get(slug, {})
        story.append(Paragraph(f"2.{i} {meta['title']}", styles["h1"]))
        story.append(Paragraph(_esc(meta["blurb"]), styles["body"]))

        if not sec:
            story.append(Paragraph("Ni podatkov za to sekcijo.", styles["body"]))
            continue

        story.append(Paragraph("Ugotovitve", styles["h2"]))
        story.append(_bullet_list(sec.get("findings") or ["(ni)"], styles))

        story.append(Paragraph("Metode / predpostavke", styles["h2"]))
        story.append(_bullet_list(sec.get("notes") or ["(ni)"], styles))

        tables = sec.get("tables") or {}
        for tname, rows in tables.items():
            story.append(Paragraph(f"Tabela: {_esc(tname)} (vse vrstice)", styles["h2"]))
            if isinstance(rows, list):
                story.append(_dict_table(rows, font, styles))
            else:
                story.append(
                    Paragraph(
                        f"<font size='6'>{_esc(json.dumps(rows, ensure_ascii=False, indent=2))}</font>",
                        styles["small"],
                    )
                )

        story.append(Paragraph("Grafi", styles["h2"]))
        for chart in sec.get("charts") or []:
            caption = f"{chart.get('title', '')} — {chart.get('help', '')}"
            _add_image(story, ADVANCED / chart["name"], caption, styles)

        story.append(Paragraph("Celoten JSON sekcije (dobesedno)", styles["h2"]))
        story.extend(
            _chunked_pre(json.dumps(sec, ensure_ascii=False, indent=2), styles)
        )
        story.append(PageBreak())

    # ----- Appendix: full advanced JSON -----
    story.append(Paragraph("Dodatek A – celoten advanced_summary.json", styles["h1"]))
    story.append(
        Paragraph(
            "Spodaj je celoten JSON naprednih analiz (brez vdelanih PNG bajtov).",
            styles["body"],
        )
    )
    story.extend(
        _chunked_pre(json.dumps(advanced, ensure_ascii=False, indent=2), styles, chunk=3000)
    )

    doc.build(story)
    return out_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = build_pdf()
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
