"""
Advanced historical analyses for flock data.

Sections:
1. Change-point detection (PELT, Binary Segmentation)
2. Event impact (before/after + impulse response + seasonal control)
3. STL-like seasonal decomposition + residual analysis
4. Clustering of weeks
5. Efficiency / frontier (eggs vs feed vs hens)
6. Interaction effects
7. Advanced anomaly detection (IF, LOF)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import aggregate_weekly, prepare_daily  # noqa: E402
from src.features import event_impact_windows  # noqa: E402

REPORTS = ROOT / "reports" / "advanced"
SUMMARY_PATH = REPORTS / "advanced_summary.json"

sns.set_theme(style="whitegrid", context="notebook")

SECTIONS = [
    {
        "slug": "change-points",
        "title": "1. Change Point Detection",
        "blurb": "Iskanje prelomov v režimu produkcije (PELT + Binary Segmentation).",
    },
    {
        "slug": "event-impact",
        "title": "2. Event Impact Analysis",
        "blurb": "Rigoroznejši vpliv dogodkov: pred/po, impulse response, sezonska kontrola.",
    },
    {
        "slug": "decomposition",
        "title": "3. Decomposition + Residual",
        "blurb": "Trend, sezona, ostanek — zanimive stvari so v residualih.",
    },
    {
        "slug": "clustering",
        "title": "4. Clustering obdobij",
        "blurb": "Tedni z podobnim vedenjem (produkcija, hrana, vreme).",
    },
    {
        "slug": "efficiency",
        "title": "5. Efficiency / Frontier",
        "blurb": "Koliko jajc pri dani hrani in jati — katera obdobja so bila najučinkovitejša.",
    },
    {
        "slug": "interactions",
        "title": "6. Interaction Effects",
        "blurb": "Kombinacije: mraz×stres, nova jata×vreme, hrana×temperatura.",
    },
    {
        "slug": "anomalies",
        "title": "7. Anomaly Detection",
        "blurb": "Isolation Forest + LOF za statistično nenavadne dneve/tedne.",
    },
]


def _save(fig: plt.Figure, name: str) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def _prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = prepare_daily().dropna(subset=["eggs", "eggs_per_hen"]).copy()
    daily = daily.sort_values("date").reset_index(drop=True)
    weekly = aggregate_weekly(prepare_daily())
    weekly = weekly[weekly["n_days"] >= 5].copy().reset_index(drop=True)
    return daily, weekly


# ---------------------------------------------------------------------------
# 1. Change points
# ---------------------------------------------------------------------------
def run_change_points(daily: pd.DataFrame) -> dict:
    series = daily["eggs_per_hen"].to_numpy(dtype=float)
    dates = daily["date"]
    signal = series.reshape(-1, 1)

    # PELT on eggs/hen
    algo_pelt = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(signal)
    bkps_pelt = algo_pelt.predict(pen=3)
    # Binary segmentation
    algo_bin = rpt.Binseg(model="l2", min_size=14).fit(signal)
    bkps_bin = algo_bin.predict(n_bkps=6)

    def _bkp_rows(bkps: list[int], method: str) -> list[dict]:
        rows = []
        for b in bkps[:-1]:  # last is end of series
            idx = int(b) - 1
            if idx < 7 or idx >= len(daily) - 7:
                continue
            pre = float(series[idx - 7 : idx].mean())
            post = float(series[idx : idx + 7].mean())
            rows.append(
                {
                    "method": method,
                    "date": str(dates.iloc[idx].date()),
                    "index": idx,
                    "pre7_eph": round(pre, 3),
                    "post7_eph": round(post, 3),
                    "delta": round(post - pre, 3),
                }
            )
        return rows

    pelt_rows = _bkp_rows(bkps_pelt, "PELT")
    bin_rows = _bkp_rows(bkps_bin, "BinSeg")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, series, color="#2c7fb8", lw=1, label="eggs/hen")
    for r in pelt_rows:
        ax.axvline(pd.Timestamp(r["date"]), color="#e6550d", ls="--", lw=1.2, alpha=0.85)
    for r in bin_rows:
        ax.axvline(pd.Timestamp(r["date"]), color="#31a354", ls=":", lw=1.2, alpha=0.7)
    ax.set_title("Change points na eggs/hen (oranžno=PELT, zeleno=BinSeg)")
    ax.set_ylabel("Eggs / hen")
    ax.legend(loc="upper right")
    chart = _save(fig, "cp_01_eggs_per_hen.png")

    # Also on absolute eggs
    eggs = daily["eggs"].to_numpy(dtype=float).reshape(-1, 1)
    bkps_e = rpt.Pelt(model="rbf", min_size=14).fit(eggs).predict(pen=3)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(dates, daily["eggs"], color="#8c6bb1", lw=1)
    for b in bkps_e[:-1]:
        ax.axvline(dates.iloc[b - 1], color="#e6550d", ls="--", lw=1)
    ax.set_title("Change points na absolutnih jajcih (PELT)")
    ax.set_ylabel("Eggs")
    chart2 = _save(fig, "cp_02_eggs_abs.png")

    return {
        "slug": "change-points",
        "findings": [
            f"PELT našel {len(pelt_rows)} prelomov na eggs/hen (min. segment 14 dni).",
            f"Binary Segmentation (6 prelomov) doda alternativne kandidatne datume.",
            "Prelomi so hipoteze — preveri dogodke/krmo/mraz okoli teh datumov, tudi če niso zabeleženi.",
        ],
        "tables": {
            "pelt": pelt_rows,
            "binseg": bin_rows,
        },
        "charts": [
            {
                "name": chart,
                "title": "PELT / BinSeg na eggs/hen",
                "help": "Navpične črte = ocenjeni prelomi režima. Oranžna = PELT, zelena = Binary Segmentation.",
            },
            {
                "name": chart2,
                "title": "PELT na absolutnih jajcih",
                "help": "Prelomi na surovem številu jajc (bolj občutljivo na spremembo velikosti jate).",
            },
        ],
        "notes": [
            "Metode: ruptures PELT (RBF) in Binary Segmentation (L2).",
            "Bayesian CPD ni vključen (težji stack); PELT je praktičen baseline.",
            "Delta = povprečje eggs/hen 7 dni po − 7 dni pred prelomom.",
        ],
    }


# ---------------------------------------------------------------------------
# 2. Event impact
# ---------------------------------------------------------------------------
def run_event_impact(daily: pd.DataFrame) -> dict:
    event_cols = [
        ("event_death", "Pogin"),
        ("event_released", "Izpust"),
        ("event_added", "Dodajanje"),
        ("event_feed_change", "Sprememba hrane"),
        ("event_egg_anomaly", "Anomalija jajc"),
    ]
    impulse_rows = []
    summary = []

    fig, ax = plt.subplots(figsize=(11, 5))
    for col, label in event_cols:
        n = int(daily[col].sum()) if col in daily.columns else 0
        if n == 0:
            summary.append({"event": label, "n": 0})
            continue
        win = event_impact_windows(daily, col, metric_col="eggs_per_hen", pre=14, post=21)
        curve = win.groupby("day_offset")["eggs_per_hen"].mean()
        ax.plot(curve.index, curve.values, marker="o", ms=3, label=f"{label} (n={n})")
        pre = float(win.loc[win["day_offset"].between(-14, -1), "eggs_per_hen"].mean())
        post = float(win.loc[win["day_offset"].between(1, 14), "eggs_per_hen"].mean())
        # Seasonal control: same month/day-of-year windows without event
        summary.append(
            {
                "event": label,
                "n": n,
                "pre14": round(pre, 3),
                "post14": round(post, 3),
                "delta": round(post - pre, 3),
            }
        )
        for off, val in curve.items():
            impulse_rows.append({"event": label, "day_offset": int(off), "eggs_per_hen": float(val)})

    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Dni glede na dogodek")
    ax.set_ylabel("Mean eggs / hen")
    ax.set_title("Impulse response dogodkov (−14 … +21)")
    ax.legend(fontsize=8)
    chart1 = _save(fig, "ev_01_impulse.png")

    # Seasonal matched control for deaths: compare event windows vs same DOY in other years without event
    control_rows = []
    if "event_death" in daily.columns and daily["event_death"].any():
        event_dates = daily.loc[daily["event_death"], "date"]
        for ed in event_dates:
            pre_e = daily[(daily["date"] >= ed - pd.Timedelta(days=14)) & (daily["date"] < ed)]
            post_e = daily[(daily["date"] > ed) & (daily["date"] <= ed + pd.Timedelta(days=14))]
            # control: same calendar window previous/next year if available
            for year_shift in (-1, 1):
                c0 = ed + pd.DateOffset(years=year_shift) - pd.Timedelta(days=14)
                c1 = ed + pd.DateOffset(years=year_shift) + pd.Timedelta(days=14)
                ctrl = daily[(daily["date"] >= c0) & (daily["date"] <= c1)]
                if len(ctrl) < 10:
                    continue
                mid = ed + pd.DateOffset(years=year_shift)
                pre_c = ctrl[ctrl["date"] < mid]["eggs_per_hen"].mean()
                post_c = ctrl[ctrl["date"] > mid]["eggs_per_hen"].mean()
                if pd.isna(pre_c) or pd.isna(post_c):
                    continue
                did = (post_e["eggs_per_hen"].mean() - pre_e["eggs_per_hen"].mean()) - (
                    post_c - pre_c
                )
                control_rows.append(
                    {
                        "event_date": str(ed.date()),
                        "control_year_shift": year_shift,
                        "treated_delta": round(
                            float(post_e["eggs_per_hen"].mean() - pre_e["eggs_per_hen"].mean()),
                            3,
                        ),
                        "control_delta": round(float(post_c - pre_c), 3),
                        "did": round(float(did), 3),
                    }
                )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [s["event"] for s in summary if s.get("n", 0) > 0]
    deltas = [s["delta"] for s in summary if s.get("n", 0) > 0]
    colors = ["#31a354" if d >= 0 else "#e6550d" for d in deltas]
    ax.barh(labels[::-1], deltas[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Δ eggs/hen (post14 − pre14)")
    ax.set_title("Before–After povzetek")
    chart2 = _save(fig, "ev_02_before_after.png")

    findings = [
        "Impulse response kaže trajanje učinka, ne samo povprečje pred/po.",
        "DiD tukaj = (Δ po dogodku) − (Δ v istem koledarskem oknu ±1 leto), kjer podatki obstajajo.",
    ]
    if control_rows:
        mean_did = float(np.mean([r["did"] for r in control_rows]))
        findings.append(f"Povprečni DiD za pogine: {mean_did:+.3f} eggs/hen (sezonsko korigirano).")

    return {
        "slug": "event-impact",
        "findings": findings,
        "tables": {"summary": summary, "did_death": control_rows},
        "charts": [
            {
                "name": chart1,
                "title": "Impulse response",
                "help": "Povprečna eggs/hen po dnevih okoli dogodka. Dan 0 = dogodek.",
            },
            {
                "name": chart2,
                "title": "Before–After Δ",
                "help": "Razlika povprečja 14 dni po − 14 dni pred. Zeleno = rast, oranžno = padec.",
            },
        ],
        "notes": [
            "Klasični DiD z večjatno kontrolo ni mogoč (ena jata). Uporabljena je sezonska kontrola YoY.",
            "Impulse window: −14 … +21 dni.",
        ],
    }


# ---------------------------------------------------------------------------
# 3. Decomposition
# ---------------------------------------------------------------------------
def run_decomposition(daily: pd.DataFrame) -> dict:
    # STL needs regular series; reindex to full calendar and interpolate short gaps
    s = daily.set_index("date")["eggs_per_hen"].asfreq("D")
    s = s.interpolate(limit=3)
    s = s.dropna()
    stl = STL(s, period=7, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(s.index, s.values, color="#2c7fb8", lw=1)
    axes[0].set_ylabel("Observed")
    axes[0].set_title("STL dekompozicija eggs/hen (period=7)")
    axes[1].plot(res.trend.index, res.trend, color="#e6550d", lw=1)
    axes[1].set_ylabel("Trend")
    axes[2].plot(res.seasonal.index, res.seasonal, color="#31a354", lw=0.8)
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(res.resid.index, res.resid, color="#8c6bb1", lw=0.8)
    axes[3].set_ylabel("Residual")
    chart1 = _save(fig, "dec_01_stl.png")

    resid = res.resid.dropna()
    z = (resid - resid.mean()) / resid.std(ddof=0)
    outliers = z[z.abs() > 2.5]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(resid.index, resid.values, color="#9ebcda", lw=0.9)
    ax.scatter(outliers.index, outliers.values, c="#d7301f", s=22, zorder=3)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Residuali (|z|>2.5 označeni)")
    ax.set_ylabel("Residual eggs/hen")
    chart2 = _save(fig, "dec_02_residuals.png")

    top = (
        outliers.abs()
        .sort_values(ascending=False)
        .head(12)
        .rename("abs_z")
        .to_frame()
        .assign(residual=lambda d: resid.loc[d.index].values)
        .reset_index()
        .rename(columns={"date": "date", "index": "date"})
    )
    # fix column name
    if "index" in top.columns:
        top = top.rename(columns={"index": "date"})
    top_rows = [
        {
            "date": str(pd.Timestamp(r["date"]).date()),
            "residual": round(float(r["residual"]), 3),
            "abs_z": round(float(r["abs_z"]), 2),
        }
        for _, r in top.iterrows()
    ]

    return {
        "slug": "decomposition",
        "findings": [
            "STL loči tedensko sezono (period=7), počasni trend in ostanek.",
            f"V residualih je {len(outliers)} dni z |z|>2.5 — kandidati za nepojasnjene padce/rasti.",
            "Če residual pade, trend/sezona tega ne pojasnita → išči dogodek, bolezen, krmo, merilno napako.",
        ],
        "tables": {"top_residuals": top_rows},
        "charts": [
            {
                "name": chart1,
                "title": "STL komponente",
                "help": "Observed = Trend + Seasonal + Residual. Analiziraj predvsem residual.",
            },
            {
                "name": chart2,
                "title": "Residual outliers",
                "help": "Rdeče točke = dnevi, kjer ostanek močno odstopa od 0.",
            },
        ],
        "notes": [
            "Uporabljen statsmodels STL (robust=True, period=7).",
            "Za letno sezono bi potrebovali daljšo serijo; tukaj je poudarek na tedenskem ritmu + trendu.",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------
def run_clustering(weekly: pd.DataFrame) -> dict:
    feats = [
        "eggs_per_hen_day",
        "feed_per_bird_day_kg" if "feed_per_bird_day_kg" in weekly.columns else "feed_per_hen_day_kg",
        "temp_avg",
        "precipitation",
        "hens_mean",
    ]
    feats = [f for f in feats if f in weekly.columns]
    data = weekly.dropna(subset=feats).copy()
    X = StandardScaler().fit_transform(data[feats])
    k = 4
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    data["cluster"] = km.fit_predict(X)

    centers = data.groupby("cluster")[feats].mean().round(3)
    counts = data["cluster"].value_counts().sort_index()

    # Label clusters heuristically
    labels = {}
    for c, row in centers.iterrows():
        eph = row.get("eggs_per_hen_day", 0)
        feed = row.get(feats[1], 0) if len(feats) > 1 else 0
        temp = row.get("temp_avg", 0)
        if eph >= centers["eggs_per_hen_day"].median() and feed <= centers[feats[1]].median():
            labels[int(c)] = "Visoka produkcija / varčna hrana"
        elif temp <= centers["temp_avg"].quantile(0.35) and eph < centers["eggs_per_hen_day"].median():
            labels[int(c)] = "Zimski / hladnejši režim"
        elif eph < centers["eggs_per_hen_day"].quantile(0.35):
            labels[int(c)] = "Nizka produkcija / stres"
        else:
            labels[int(c)] = "Srednji / mešani režim"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in sorted(data["cluster"].unique()):
        part = data[data["cluster"] == c]
        ax.scatter(part["week_start"], part["eggs_per_hen_day"], s=40, label=f"C{c}: {labels.get(int(c), '')}")
    ax.set_title("Tedni po clusterjih (eggs/hen/day)")
    ax.set_ylabel("Eggs / hen / day")
    ax.legend(fontsize=8)
    chart1 = _save(fig, "cl_01_timeline.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = data.copy()
    plot_df["label"] = plot_df["cluster"].map(lambda c: f"C{c}")
    sns.scatterplot(
        data=plot_df,
        x=feats[1],
        y="eggs_per_hen_day",
        hue="label",
        style="label",
        s=60,
        ax=ax,
    )
    ax.set_title("Clusterji: hrana vs eggs/hen")
    chart2 = _save(fig, "cl_02_feed_vs_eph.png")

    table = []
    for c in sorted(centers.index):
        row = {"cluster": int(c), "label": labels.get(int(c), ""), "n_weeks": int(counts.loc[c])}
        for f in feats:
            row[f] = float(centers.loc[c, f])
        table.append(row)

    return {
        "slug": "clustering",
        "findings": [
            f"KMeans (k={k}) na standardiziranih tedenskih featurejih.",
            "Clusterji pomagajo opisati režime (npr. zimski vs varčen visokodonosen).",
        ],
        "tables": {"clusters": table},
        "charts": [
            {
                "name": chart1,
                "title": "Clusterji skozi čas",
                "help": "Vsaka točka = teden. Barva = skupina podobnega vedenja.",
            },
            {
                "name": chart2,
                "title": "Hrana vs produkcija",
                "help": "Ali so clusterji ločeni po učinkovitosti (veliko jajc / malo hrane).",
            },
        ],
        "notes": [
            f"Featureji: {', '.join(feats)}.",
            "DTW ni uporabljen (kratek nabor tednov); KMeans je jasen baseline.",
        ],
    }


# ---------------------------------------------------------------------------
# 5. Efficiency
# ---------------------------------------------------------------------------
def run_efficiency(weekly: pd.DataFrame) -> dict:
    w = weekly.dropna(subset=["eggs", "feed_kg", "hens_mean"]).copy()
    birds = w["hens_mean"] + w.get("roosters_mean", 0).fillna(0)
    w["birds"] = birds
    w["eggs_per_kg_feed"] = w["eggs"] / w["feed_kg"].replace(0, np.nan)
    w["eggs_per_hen_week"] = w["eggs"] / w["hens_mean"].replace(0, np.nan)

    # Simple frontier: top decile eggs_per_kg_feed
    q90 = w["eggs_per_kg_feed"].quantile(0.9)
    w["is_frontier"] = w["eggs_per_kg_feed"] >= q90

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(w["feed_kg"], w["eggs"], c="#9ebcda", s=35, alpha=0.8, label="tedni")
    fr = w[w["is_frontier"]]
    ax.scatter(fr["feed_kg"], fr["eggs"], c="#e6550d", s=55, label="frontier (top 10% eggs/kg)")
    ax.set_xlabel("Feed kg / week")
    ax.set_ylabel("Eggs / week")
    ax.set_title("Efficiency scatter + empirična frontier")
    ax.legend()
    chart1 = _save(fig, "ef_01_frontier.png")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(w["week_start"], w["eggs_per_kg_feed"], color="#2c7fb8", lw=1.2)
    ax.axhline(q90, color="#e6550d", ls="--", label=f"P90={q90:.2f}")
    ax.set_ylabel("Eggs / kg feed")
    ax.set_title("Učinkovitost skozi čas")
    ax.legend()
    chart2 = _save(fig, "ef_02_eggs_per_kg.png")

    # Diminishing returns: bin by feed_per_bird
    feed_col = "feed_per_bird_day_kg" if "feed_per_bird_day_kg" in w.columns else "feed_per_hen_day_kg"
    if feed_col in w.columns:
        w2 = w.dropna(subset=[feed_col, "eggs_per_hen_day"])
        w2 = w2.assign(feed_bin=pd.qcut(w2[feed_col], 5, duplicates="drop"))
        bins = w2.groupby("feed_bin", observed=True).agg(
            feed=("feed_bin", lambda s: s.astype(str).iloc[0]),
            mean_feed=(feed_col, "mean"),
            mean_eph=("eggs_per_hen_day", "mean"),
            n=("eggs", "count"),
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(bins["mean_feed"], bins["mean_eph"], "o-", color="#31a354")
        ax.set_xlabel("Feed / bird / day (kg)")
        ax.set_ylabel("Eggs / hen / day")
        ax.set_title("Ali več hrane še vedno prinese jajca? (kvintili)")
        chart3 = _save(fig, "ef_03_diminishing.png")
    else:
        chart3 = None
        bins = pd.DataFrame()

    top = w.nlargest(8, "eggs_per_kg_feed")[
        ["week_start", "eggs", "feed_kg", "hens_mean", "eggs_per_kg_feed"]
    ]
    top_rows = [
        {
            "week_start": str(pd.Timestamp(r.week_start).date()),
            "eggs": round(float(r.eggs), 1),
            "feed_kg": round(float(r.feed_kg), 2),
            "hens": round(float(r.hens_mean), 1),
            "eggs_per_kg": round(float(r.eggs_per_kg_feed), 2),
        }
        for r in top.itertuples()
    ]

    return {
        "slug": "efficiency",
        "findings": [
            f"Povprečna učinkovitost: {w['eggs_per_kg_feed'].mean():.2f} jajc / kg hrane.",
            f"Frontier = zgornjih 10% tednov (≥ {q90:.2f} jajc/kg).",
            "Če krivulja kvintilov flatenira, dodatna hrana prinese malo dodatnih jajc.",
        ],
        "tables": {"top_weeks": top_rows},
        "charts": [
            {
                "name": chart1,
                "title": "Eggs vs feed frontier",
                "help": "Oranžni tedni so empirično najbolj učinkoviti (veliko jajc na kg hrane).",
            },
            {
                "name": chart2,
                "title": "Eggs/kg skozi čas",
                "help": "Časovna vrsta učinkovitosti. Črtkana = 90. percentil.",
            },
        ]
        + (
            [
                {
                    "name": chart3,
                    "title": "Padajoči donosi hrane",
                    "help": "Povprečna eggs/hen po kvintilih dnevne hrane na žival.",
                }
            ]
            if chart3
            else []
        ),
        "notes": [
            "Hrana vključuje petelina; jajca so samo od kokoši — to je namerno za farm efficiency.",
            "Ni DEA/optimizacijski frontier model; empirični P90 je interpretabilen baseline.",
        ],
    }


# ---------------------------------------------------------------------------
# 6. Interactions
# ---------------------------------------------------------------------------
def run_interactions(daily: pd.DataFrame) -> dict:
    d = daily.copy()
    d["cold"] = d["temp_avg"] <= d["temp_avg"].quantile(0.2)
    d["recent_stress"] = (
        d["event_death"].fillna(False)
        | d["event_released"].fillna(False)
        | d["event_feed_change"].fillna(False)
        | d["event_added"].fillna(False)
    )
    # stress in last 7 days
    d["stress_7d"] = (
        d["recent_stress"].astype(int).rolling(7, min_periods=1).max().astype(bool)
    )

    # Cold × stress
    pivot = (
        d.dropna(subset=["eggs_per_hen", "temp_avg"])
        .groupby(["cold", "stress_7d"])["eggs_per_hen"]
        .agg(["mean", "count"])
        .reset_index()
    )
    pivot_rows = [
        {
            "cold": bool(r.cold),
            "stress_7d": bool(r.stress_7d),
            "mean_eph": round(float(r.mean), 3),
            "n": int(r.count),
        }
        for r in pivot.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stress, label in [(False, "brez stresa (7d)"), (True, "po stresu (7d)")]:
        part = d[(d["stress_7d"] == stress)].dropna(subset=["temp_avg", "eggs_per_hen"])
        # binned means
        part = part.assign(temp_bin=pd.qcut(part["temp_avg"], 5, duplicates="drop"))
        g = part.groupby("temp_bin", observed=True).agg(
            t=("temp_avg", "mean"), e=("eggs_per_hen", "mean")
        )
        ax.plot(g["t"], g["e"], "o-", label=label)
    ax.set_xlabel("Temp avg °C")
    ax.set_ylabel("Eggs / hen")
    ax.set_title("Interakcija: temperatura × nedavni stres")
    ax.legend()
    chart1 = _save(fig, "ix_01_temp_stress.png")

    # Feed × temperature
    feed_col = "feed_per_bird_kg" if "feed_per_bird_kg" in d.columns else "feed_per_hen_kg"
    d2 = d.dropna(subset=[feed_col, "temp_avg", "eggs_per_hen"]).copy()
    d2["temp_group"] = pd.qcut(d2["temp_avg"], 3, labels=["hladno", "srednje", "toplo"])
    d2["feed_group"] = pd.qcut(d2[feed_col], 3, labels=["malo", "srednje", "veliko"])
    heat = d2.pivot_table(
        index="temp_group", columns="feed_group", values="eggs_per_hen", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
    ax.set_title("Interakcija hrana × temperatura (mean eggs/hen)")
    chart2 = _save(fig, "ix_02_feed_temp_heatmap.png")

    # New flock proxy: hens jumped recently OR event_added
    d["hens_up"] = d["hens"].diff().fillna(0) > 0
    after_add = d["event_added"].fillna(False) | d["hens_up"]
    d["new_flock_30d"] = after_add.astype(int).rolling(30, min_periods=1).max().astype(bool)
    g = (
        d.dropna(subset=["eggs_per_hen", "precipitation"])
        .groupby(["new_flock_30d", d["precipitation"] > 0])["eggs_per_hen"]
        .mean()
        .rename("mean_eph")
        .reset_index()
    )
    # rename columns carefully
    g.columns = ["new_flock_30d", "rainy", "mean_eph"]
    flock_rows = [
        {
            "new_flock_30d": bool(r.new_flock_30d),
            "rainy": bool(r.rainy),
            "mean_eph": round(float(r.mean_eph), 3),
        }
        for r in g.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    cats = ["suho", "dež"]
    old = [
        g.loc[(g.new_flock_30d == False) & (g.rainy == False), "mean_eph"],
        g.loc[(g.new_flock_30d == False) & (g.rainy == True), "mean_eph"],
    ]
    new = [
        g.loc[(g.new_flock_30d == True) & (g.rainy == False), "mean_eph"],
        g.loc[(g.new_flock_30d == True) & (g.rainy == True), "mean_eph"],
    ]
    old_v = [float(x.iloc[0]) if len(x) else np.nan for x in old]
    new_v = [float(x.iloc[0]) if len(x) else np.nan for x in new]
    x = np.arange(2)
    ax.bar(x - width / 2, old_v, width, label="stara jata")
    ax.bar(x + width / 2, new_v, width, label="nova jata (30d)")
    ax.set_xticks(x, cats)
    ax.set_ylabel("Eggs / hen")
    ax.set_title("Nova jata × dež")
    ax.legend()
    chart3 = _save(fig, "ix_03_flock_rain.png")

    return {
        "slug": "interactions",
        "findings": [
            "Mraz po stresnem dogodku je lahko drugačen kot mraz v mirnem obdobju.",
            "Heatmapa hrana×temp pokaže, v katerem režimu hrana najbolj “deluje”.",
            "Nova jata × dež: primerjava nesnosti v suhem/deževnem dnevu.",
        ],
        "tables": {
            "cold_stress": pivot_rows,
            "flock_rain": flock_rows,
        },
        "charts": [
            {
                "name": chart1,
                "title": "Temp × stres",
                "help": "Krivulji eggs/hen po temperaturi, ločeno glede na stres v zadnjih 7 dneh.",
            },
            {
                "name": chart2,
                "title": "Hrana × temperatura",
                "help": "Povprečna eggs/hen v kombinacijah tertilov hrane in temperature.",
            },
            {
                "name": chart3,
                "title": "Nova jata × dež",
                "help": "Ali nova jata (30 dni po dodajanju) drugače reagira na dež.",
            },
        ],
        "notes": [
            "Interakcije so deskriptivne (ne polni kauzalni model).",
            "cold = spodnjih 20% temp_avg; stress_7d = smrt/izpust/hrana/dodajanje v 7 dneh.",
        ],
    }


# ---------------------------------------------------------------------------
# 7. Anomalies
# ---------------------------------------------------------------------------
def run_anomalies(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    feat_cols = [
        c
        for c in [
            "eggs",
            "eggs_per_hen",
            "feed_kg",
            "feed_per_bird_kg",
            "feed_per_hen_kg",
            "temp_avg",
            "precipitation",
            "humidity_avg",
            "hens",
        ]
        if c in daily.columns
    ]
    d = daily.dropna(subset=feat_cols).copy()
    X = StandardScaler().fit_transform(d[feat_cols])

    iso = IsolationForest(contamination=0.05, random_state=42)
    d["if_outlier"] = iso.fit_predict(X) == -1
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    d["lof_outlier"] = lof.fit_predict(X) == -1
    d["both"] = d["if_outlier"] & d["lof_outlier"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d["date"], d["eggs_per_hen"], color="#9ebcda", lw=1)
    both = d[d["both"]]
    ax.scatter(both["date"], both["eggs_per_hen"], c="#d7301f", s=36, label="IF ∩ LOF", zorder=3)
    only_if = d[d["if_outlier"] & ~d["lof_outlier"]]
    ax.scatter(only_if["date"], only_if["eggs_per_hen"], c="#e6550d", s=22, label="samo IF", zorder=2)
    ax.set_title("Napredne anomalije (Isolation Forest + LOF)")
    ax.legend()
    chart1 = _save(fig, "an_01_daily.png")

    # Weekly IF
    wfeats = [
        c
        for c in ["eggs", "eggs_per_hen_day", "feed_kg", "temp_avg", "precipitation", "hens_mean"]
        if c in weekly.columns
    ]
    w = weekly.dropna(subset=wfeats).copy()
    Xw = StandardScaler().fit_transform(w[wfeats])
    w["if_outlier"] = IsolationForest(contamination=0.08, random_state=42).fit_predict(Xw) == -1
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(w["week_start"], w["eggs"], width=5, color="#9ebcda")
    ax.bar(
        w.loc[w["if_outlier"], "week_start"],
        w.loc[w["if_outlier"], "eggs"],
        width=5,
        color="#d7301f",
        label="outlier teden",
    )
    ax.set_title("Tedenske anomalije (Isolation Forest)")
    ax.legend()
    chart2 = _save(fig, "an_02_weekly.png")

    top = d[d["both"]].sort_values("date")[
        ["date", "eggs", "eggs_per_hen", "feed_kg", "temp_avg", "precipitation"]
    ].head(20)
    rows = [
        {
            "date": str(pd.Timestamp(r.date).date()),
            "eggs": float(r.eggs),
            "eggs_per_hen": round(float(r.eggs_per_hen), 3),
            "feed_kg": round(float(r.feed_kg), 3),
            "temp_avg": round(float(r.temp_avg), 2) if pd.notna(r.temp_avg) else None,
            "precipitation": round(float(r.precipitation), 1)
            if pd.notna(r.precipitation)
            else None,
        }
        for r in top.itertuples()
    ]

    return {
        "slug": "anomalies",
        "findings": [
            f"Isolation Forest označil {int(d['if_outlier'].sum())} dni; LOF {int(d['lof_outlier'].sum())}.",
            f"Presek (bolj zanesljivi kandidati): {int(d['both'].sum())} dni.",
            f"Tedenski IF outlierji: {int(w['if_outlier'].sum())}.",
        ],
        "tables": {"daily_both": rows},
        "charts": [
            {
                "name": chart1,
                "title": "Dnevne anomalije",
                "help": "Rdeče = IF in LOF se strinjata. Oranžno = samo Isolation Forest.",
            },
            {
                "name": chart2,
                "title": "Tedenske anomalije",
                "help": "Tedni, ki so nenavadni glede na jajca+hrano+vreme+jate.",
            },
        ],
        "notes": [
            f"Featureji: {', '.join(feat_cols)}.",
            "Seasonal Hybrid ESD ni ločeno implementiran; STL residuali pokrivajo sezonski del (glej Decomposition).",
            "contamination≈5–8% — pričakuj redke, a ne prazne zadetke.",
        ],
    }


def run_all() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    daily, weekly = _prepare()
    results = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "period_start": str(daily["date"].min().date()),
        "period_end": str(daily["date"].max().date()),
        "sections": {},
    }
    runners = [
        run_change_points,
        run_event_impact,
        run_decomposition,
        lambda d=None, w=None: run_clustering(weekly),
        lambda: run_efficiency(weekly),
        run_interactions,
        lambda: run_anomalies(daily, weekly),
    ]
    # clearer calls
    results["sections"]["change-points"] = run_change_points(daily)
    results["sections"]["event-impact"] = run_event_impact(daily)
    results["sections"]["decomposition"] = run_decomposition(daily)
    results["sections"]["clustering"] = run_clustering(weekly)
    results["sections"]["efficiency"] = run_efficiency(weekly)
    results["sections"]["interactions"] = run_interactions(daily)
    results["sections"]["anomalies"] = run_anomalies(daily, weekly)

    SUMMARY_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    for slug, sec in results["sections"].items():
        print(f"  {slug}: {len(sec.get('charts', []))} charts, {len(sec.get('findings', []))} findings")
    return results


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return run_all()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_all()


if __name__ == "__main__":
    main()
