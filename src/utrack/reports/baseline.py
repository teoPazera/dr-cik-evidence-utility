"""U0.4: run, score and report the zero-cost statistical-forecaster baseline.

Condition C0 (no context) on all 199 dev tasks, two forecasters, 100 samples
each, scored under all three scalings (Decision A). These numbers are the
statistical floor for later stages, not a claim about the Dr-CiK paper's
figures (plan_a.md U0.4: different task set, no tuning to match).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utrack.data.loader import Dataset, fill_history_forward
from utrack.forecasters.base import Forecaster, resolve_seasonal_period_steps
from utrack.forecasters.naive import LastValueNaiveForecaster
from utrack.forecasters.seasonal_naive import SeasonalNaiveForecaster
from utrack.scoring.aggregate import aggregate_over_tasks, winsorise
from utrack.scoring.crps import mae_of_median, mean_crps, rmse_of_mean
from utrack.scoring.scaling import (
    scale_a1_future_range,
    scale_a2_seasonal_naive_mae,
    scale_a3_mean_abs_history,
)
from utrack.store.forecast_store import ForecastStore, build_cell_record
from utrack.store.manifest import RunManifest, config_hash, git_commit, new_run_id

logger = logging.getLogger(__name__)

FORECASTERS: dict[str, Forecaster] = {
    "last_value_naive": LastValueNaiveForecaster(),
    "seasonal_naive": SeasonalNaiveForecaster(),
}
SCALINGS = ("a1", "a2", "a3")

PAPER_COMPARISON_NOTE = (
    "plan_a.md U0.4 asks to quote the Dr-CiK paper's no-context naive score next to these numbers "
    "*if the paper states one*. Checked on 2026-09-18 (arXiv:2605.27904): the abstract gives no such "
    "number, and an automated read of the full-text PDF found none stated either. Caveat: that read "
    "was machine extraction, so a number inside a figure or an image-rendered table could have been "
    "missed; a human check of the paper's results tables would settle it. Nothing here was tuned "
    "toward any external figure."
)


def _dev_task_ids(dataset: Dataset) -> list[str]:
    def sort_key(bid: str) -> tuple[int, str]:
        parts = bid.split("_")
        return (int(parts[1]), bid) if len(parts) > 1 and parts[1].isdigit() else (10**9, bid)

    return sorted((bid for bid, t in dataset.tasks.items() if t.labels_public), key=sort_key)


def _cell_seed(base_seed: int, *parts: str) -> int:
    """Deterministic per-cell seed from a config seed + identifying strings
    (plan_a.md 5.2.7: seeded from config, no reliance on Python's per-process hash())."""
    digest = hashlib.sha256("|".join((str(base_seed), *parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_baseline(
    repo_root: Path,
    dataset: Dataset,
    u0_config: dict,
    machine_name: str,
    store_path: Path,
    manifest_path: Path,
) -> RunManifest:
    baseline_cfg = u0_config["baseline"]
    n_samples = int(baseline_cfg["n_samples"])
    base_seed = int(baseline_cfg["seed"])
    condition_id = baseline_cfg["condition_id"]
    forecaster_names = baseline_cfg["forecasters"]
    dataset_revision = u0_config["dataset"]["revision"]

    commit = git_commit(repo_root)
    cfg_hash = config_hash(baseline_cfg)
    started_at = datetime.now(timezone.utc).isoformat()

    store = ForecastStore(store_path)
    dev_task_ids = _dev_task_ids(dataset)

    n_cells = 0
    for bid in dev_task_ids:
        forecast_input = dataset.forecast_input(bid)
        for name in forecaster_names:
            forecaster = FORECASTERS[name]
            seed = _cell_seed(base_seed, bid, condition_id, name)
            output = forecaster.forecast(forecast_input, context=None, n_samples=n_samples, seed=seed)
            record = build_cell_record(
                benchmark_id=bid,
                condition_id=condition_id,
                repeat=0,
                condition_seed=0,
                output=output,
                dataset_revision=dataset_revision,
                code_commit=commit,
                config_hash=cfg_hash,
            )
            store.append(record)
            n_cells += 1

    manifest = RunManifest(
        run_id=new_run_id(),
        machine=machine_name,
        dataset_revision=dataset_revision,
        code_commit=commit,
        config_hash=cfg_hash,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        n_cells=n_cells,
        forecaster_names=list(forecaster_names),
        condition_ids=[condition_id],
    )
    manifest.write(manifest_path)
    return manifest


def score_baseline(dataset: Dataset, store: ForecastStore, u0_config: dict) -> pd.DataFrame:
    fallback_lag = int(u0_config["baseline"]["scaling_a2_fallback_lag_steps"])
    rows: list[dict] = []

    for record in store.read_all():
        bid = record["benchmark_id"]
        task = dataset.tasks[bid]
        samples = ForecastStore.samples_as_array(record)
        target = np.asarray(task.future_values, dtype=float)
        history = fill_history_forward(task.history_values, bid)

        raw_crps = mean_crps(target, samples)
        raw_mae = mae_of_median(target, samples)
        raw_rmse = rmse_of_mean(target, samples)

        m = resolve_seasonal_period_steps(task.seasonal_period, task.frequency, bid) or fallback_lag

        denom = {
            "a1": scale_a1_future_range(target, bid),
            "a2": scale_a2_seasonal_naive_mae(history, m, bid),
            "a3": scale_a3_mean_abs_history(history, bid),
        }

        row: dict = {
            "benchmark_id": bid,
            "forecaster_name": record["forecaster_name"],
            "n_samples": samples.shape[0],
            "horizon": samples.shape[1],
            "season_length_steps": int(record["sampling_params"].get("resolved_season_length_steps", 1)),
            "fell_back_to_naive": any("fell back" in n for n in record.get("notes", [])),
            "raw_crps": raw_crps,
            "raw_mae": raw_mae,
            "raw_rmse": raw_rmse,
        }
        for key in SCALINGS:
            d = denom[key]
            row[f"denom_{key}"] = d
            degenerate = not np.isfinite(d) or d < 1e-9
            row[f"denom_{key}_degenerate"] = degenerate
            scaled_crps = raw_crps / d if not degenerate else float("nan")
            w = winsorise(scaled_crps) if np.isfinite(scaled_crps) else None
            row[f"scaled_crps_{key}"] = w.winsorised if w else float("nan")
            row[f"scaled_crps_{key}_capped"] = w.capped if w else False
            row[f"scaled_mae_{key}"] = raw_mae / d if not degenerate else float("nan")
            row[f"scaled_rmse_{key}"] = raw_rmse / d if not degenerate else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["forecaster_name", "benchmark_id"]).reset_index(drop=True)
    return df


def write_baseline_scores(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _rank_agreement(df_forecaster: pd.DataFrame) -> dict[str, float]:
    cols = {k: f"scaled_crps_{k}" for k in SCALINGS}
    out = {}
    for a, b in (("a1", "a2"), ("a1", "a3"), ("a2", "a3")):
        # Spearman = Pearson on (average-tie) ranks; pandas' method="spearman" would pull in scipy.
        pair = df_forecaster[[cols[a], cols[b]]].dropna()
        out[f"{a}_vs_{b}"] = float(pair[cols[a]].rank().corr(pair[cols[b]].rank()))
    return out


def write_baseline_report(df: pd.DataFrame, out_path: Path, paper_note: str | None = None) -> None:
    lines: list[str] = []
    lines.append("# U0.4 baseline report")
    lines.append("")
    lines.append(
        f"Condition C0 (no context), {df['benchmark_id'].nunique()} dev tasks, "
        f"{sorted(df['forecaster_name'].unique())} forecasters, "
        f"{int(df['n_samples'].iloc[0])} samples each. Generated by `uv run utrack report baseline`."
    )
    lines.append("")
    lines.append(
        "These are the statistical floor for later stages. They are not expected to match "
        "the Dr-CiK paper's figures (a different task set, 240 vs. 279 tasks) and nothing "
        "here was tuned to match them."
    )
    lines.append("")

    lines.append("## 1. Score distributions per scaling")
    lines.append("")
    for forecaster in sorted(df["forecaster_name"].unique()):
        sub = df[df["forecaster_name"] == forecaster]
        lines.append(f"### {forecaster}")
        lines.append("")
        lines.append("| scaling | mean | stderr | median | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for key in SCALINGS:
            col = sub[f"scaled_crps_{key}"].dropna()
            agg = aggregate_over_tasks(col.tolist())
            lines.append(
                f"| {key} | {agg.mean:.4g} | {agg.stderr:.4g} | {col.median():.4g} | "
                f"{col.min():.4g} | {col.max():.4g} |"
            )
        lines.append("")

    lines.append("## 2. Share of tasks at the winsorisation cap (5.0)")
    lines.append("")
    lines.append("| forecaster | scaling | capped | total | share |")
    lines.append("|---|---|---|---|---|")
    for forecaster in sorted(df["forecaster_name"].unique()):
        sub = df[df["forecaster_name"] == forecaster]
        for key in SCALINGS:
            capped = int(sub[f"scaled_crps_{key}_capped"].sum())
            total = len(sub)
            lines.append(f"| {forecaster} | {key} | {capped} | {total} | {capped / total:.1%} |")
    lines.append("")

    lines.append("## 3. Tasks with degenerate denominators")
    lines.append("")
    any_degenerate = False
    for key in SCALINGS:
        degenerate_tasks = sorted(df.loc[df[f"denom_{key}_degenerate"], "benchmark_id"].unique())
        if degenerate_tasks:
            any_degenerate = True
            lines.append(f"- **{key}**: {len(degenerate_tasks)} task(s): {', '.join(degenerate_tasks)}")
    if not any_degenerate:
        lines.append(
            "None. Consistent with U0.2's audit, which found 0 near-constant futures among "
            "the 199 dev tasks (relevant to A1); A2 and A3 also had no near-zero denominators here."
        )
    lines.append("")

    lines.append("## 4. Rank agreement between scalings (Spearman correlation of scaled CRPS)")
    lines.append("")
    lines.append("| forecaster | a1 vs a2 | a1 vs a3 | a2 vs a3 |")
    lines.append("|---|---|---|---|")
    for forecaster in sorted(df["forecaster_name"].unique()):
        sub = df[df["forecaster_name"] == forecaster]
        agreement = _rank_agreement(sub)
        lines.append(
            f"| {forecaster} | {agreement['a1_vs_a2']:.3f} | {agreement['a1_vs_a3']:.3f} | "
            f"{agreement['a2_vs_a3']:.3f} |"
        )
    lines.append("")

    lines.append("### Horizon dependence")
    lines.append("")
    lines.append(
        "Spearman correlation between each scaled CRPS and the forecast horizon. The scalings "
        "disagree on direction, which is one reason their task rankings disagree; A2 divides by a "
        "one-step error, so its score grows with horizon and it is the scaling that hits the cap."
    )
    lines.append("")
    lines.append("| forecaster | a1 vs horizon | a2 vs horizon | a3 vs horizon | median horizon, a2-capped | median horizon, not capped |")
    lines.append("|---|---|---|---|---|---|")
    for forecaster in sorted(df["forecaster_name"].unique()):
        sub = df[df["forecaster_name"] == forecaster]
        corr = {k: float(sub[f"scaled_crps_{k}"].rank().corr(sub["horizon"].rank())) for k in SCALINGS}
        capped_h = sub.loc[sub["scaled_crps_a2_capped"], "horizon"].median()
        other_h = sub.loc[~sub["scaled_crps_a2_capped"], "horizon"].median()
        lines.append(
            f"| {forecaster} | {corr['a1']:.3f} | {corr['a2']:.3f} | {corr['a3']:.3f} | "
            f"{capped_h:g} | {other_h:g} |"
        )
    lines.append("")

    lines.append("## 5. Seasonal-naive: how often a real season was used")
    lines.append("")
    seasonal = df[df["forecaster_name"] == "seasonal_naive"]
    if len(seasonal):
        n_fallback = int(seasonal["fell_back_to_naive"].sum())
        used = seasonal.loc[~seasonal["fell_back_to_naive"], "season_length_steps"].value_counts().sort_index()
        used_text = (
            " Season lengths used where a season was found: "
            + ", ".join(f"{int(m)} steps: {int(c)} tasks" for m, c in used.items())
            + "."
            if len(used)
            else ""
        )
        lines.append(
            f"{n_fallback} of {len(seasonal)} dev tasks ({n_fallback / len(seasonal):.1%}) had no "
            "usable multi-step season and fell back to m=1 (identical to `last_value_naive`). "
            "`seasonal_period` is either an integer step count (usable) or a pandas alias that "
            "re-encodes `frequency` itself (not a season) - see `forecasters/base.py`." + used_text
        )
        lines.append("")
        naive = df[df["forecaster_name"] == "last_value_naive"].set_index("benchmark_id")
        fb = seasonal[seasonal["fell_back_to_naive"]].set_index("benchmark_id")
        common = fb.index.intersection(naive.index)
        if len(common):
            rel = ((fb.loc[common, "raw_crps"] - naive.loc[common, "raw_crps"]).abs() / naive.loc[common, "raw_crps"])
            lines.append(
                "**Sanity check / sampling-noise estimate.** On the fallback tasks the two forecasters "
                "run the same process (m=1) with different seeds, so any score difference is pure "
                f"Monte Carlo noise at {int(seasonal['n_samples'].iloc[0])} samples: median relative "
                f"difference in raw CRPS {rel.median():.2%}, 90th percentile {rel.quantile(0.9):.2%}, "
                f"max {rel.max():.2%} over {len(common)} tasks."
            )
            lines.append("")

    if paper_note:
        lines.append("## 6. Dr-CiK paper comparison")
        lines.append("")
        lines.append(paper_note)
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
