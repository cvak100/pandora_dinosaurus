"""Train LightGBM weekly egg forecaster."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import WEEKLY_CSV, WEEKLY_MODEL_CSV  # noqa: E402
from src.data_processing import aggregate_weekly, prepare_daily, save_processed  # noqa: E402
from src.features import (  # noqa: E402
    TARGET_COL,
    available_feature_columns,
    build_weekly_model_frame,
    fit_seasonal_baseline,
    performance_score,
    seasonal_baseline_eggs,
)

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

# Hold out the last N weeks for validation (time-based, no shuffle).
VAL_WEEKS = 16
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 400,
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def train() -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    daily = prepare_daily()
    save_processed(daily)
    weekly = aggregate_weekly(daily)
    weekly.to_csv(PROCESSED / WEEKLY_CSV, index=False)

    frame = build_weekly_model_frame(weekly)
    frame.to_csv(PROCESSED / WEEKLY_MODEL_CSV, index=False)

    feature_cols = available_feature_columns(frame)
    if len(frame) <= VAL_WEEKS + 8:
        raise RuntimeError(
            f"Not enough weekly rows ({len(frame)}) for train/val split "
            f"with VAL_WEEKS={VAL_WEEKS}."
        )

    split_idx = len(frame) - VAL_WEEKS
    train_df = frame.iloc[:split_idx].copy()
    val_df = frame.iloc[split_idx:].copy()

    baseline = fit_seasonal_baseline(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics = _metrics(y_train.to_numpy(), train_pred)
    val_metrics = _metrics(y_val.to_numpy(), val_pred)

    # Baseline metrics on validation
    val_base = np.array([seasonal_baseline_eggs(r, baseline) for _, r in val_df.iterrows()])
    baseline_metrics = _metrics(y_val.to_numpy(), val_base)

    scores = [
        performance_score(float(p), float(b), baseline["residual_std"])
        for p, b in zip(val_pred, val_base)
    ]

    # Feature importance plot
    importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#2c7fb8")
    ax.set_title("LightGBM feature importance")
    fig.tight_layout()
    fig.savefig(REPORTS / "11_feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Prediction vs actual
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(val_df["week_start"], y_val, "o-", label="actual", color="#2c7fb8")
    ax.plot(val_df["week_start"], val_pred, "s-", label="LightGBM", color="#e6550d")
    ax.plot(val_df["week_start"], val_base, "--", label="seasonal baseline", color="#31a354")
    ax.set_title("Validation: weekly eggs")
    ax.set_ylabel("Eggs / week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "12_val_predictions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Score over validation
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(val_df["week_start"], scores, width=5, color="#9e9ac8")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Validation performance score (-1 .. +1)")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(REPORTS / "13_val_scores.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "baseline": baseline,
        "val_weeks": VAL_WEEKS,
        "lgb_params": LGB_PARAMS,
        "best_iteration": int(getattr(model, "best_iteration_", LGB_PARAMS["n_estimators"])),
    }
    joblib.dump(artifact, MODELS / "weekly_lgbm.joblib")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "val_seasonal_baseline": baseline_metrics,
        "val_score_mean": float(np.mean(scores)),
        "train_weeks": int(len(train_df)),
        "val_weeks": int(len(val_df)),
        "train_start": str(train_df["week_start"].iloc[0].date()),
        "train_end": str(train_df["week_start"].iloc[-1].date()),
        "val_start": str(val_df["week_start"].iloc[0].date()),
        "val_end": str(val_df["week_start"].iloc[-1].date()),
        "best_iteration": artifact["best_iteration"],
        "top_features": importance.head(10).to_dict(orient="records"),
    }
    (MODELS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    lines = [
        "# Part 2 – Forecasting evaluation",
        "",
        f"- Train weeks: **{metrics['train_weeks']}** "
        f"({metrics['train_start']} .. {metrics['train_end']})",
        f"- Val weeks: **{metrics['val_weeks']}** "
        f"({metrics['val_start']} .. {metrics['val_end']})",
        f"- LightGBM best_iteration: **{metrics['best_iteration']}**",
        "",
        "## Metrics",
        f"- Train MAE / RMSE: **{train_metrics['mae']:.2f}** / **{train_metrics['rmse']:.2f}**",
        f"- Val MAE / RMSE / R2: **{val_metrics['mae']:.2f}** / "
        f"**{val_metrics['rmse']:.2f}** / **{val_metrics['r2']:.3f}**",
        f"- Seasonal baseline val MAE / RMSE: **{baseline_metrics['mae']:.2f}** / "
        f"**{baseline_metrics['rmse']:.2f}**",
        f"- Mean validation score: **{metrics['val_score_mean']:.3f}**",
        "",
        "## Top features",
    ]
    for row in metrics["top_features"]:
        lines.append(f"- `{row['feature']}`: {row['importance']}")
    lines.append("")
    lines.append("Model saved to `models/weekly_lgbm.joblib`.")
    (REPORTS / "forecast_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
