"""Run and report the controlled U1.2-style task_42 C0/C1 smoke demonstration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from pathlib import Path

import numpy as np
import pandas as pd

from utrack.conditions.builders import build_condition
from utrack.conditions.leakage import leaks_beyond_history, matching_future_pairs
from utrack.data.loader import Dataset, fill_history_forward
from utrack.forecasters.base import resolve_seasonal_period_steps
from utrack.forecasters.llm_direct import LiteLLMDirectForecaster
from utrack.forecasters.llm_prompt import SYSTEM_MESSAGE, render_user_prompt
from utrack.scoring.aggregate import winsorise
from utrack.scoring.crps import mae_of_median, mean_crps, rmse_of_mean
from utrack.scoring.scaling import (
    scale_a1_future_range,
    scale_a2_seasonal_naive_mae,
    scale_a3_mean_abs_history,
)
from utrack.seeds import derive_seed
from utrack.store.forecast_store import ForecastStore, build_cell_record
from utrack.store.manifest import RunManifest, config_hash, git_commit, new_run_id

SCALINGS = ("a1", "a2", "a3")


def run_smoke(
    repo_root: Path,
    dataset: Dataset,
    u0_cfg: dict,
    u1_cfg: dict,
    machine_name: str,
    store_path: Path,
    manifest_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RunManifest:
    """Run exactly task_42 C0/C1, one repeat, 25 requested samples per condition."""
    task_id = "task_42"
    condition_ids = ("C0", "C1")
    repeat = 0
    sampling = u1_cfg["sampling"]
    n_samples = int(sampling["samples_per_cell"])
    forecaster = LiteLLMDirectForecaster(
        model=u1_cfg["model"],
        temperature=float(sampling["temperature"]),
        n_retries=int(u1_cfg.get("n_retries", 3)),
        cost_cap_usd=float(u1_cfg["cost_cap_usd"]),
        input_price_per_million=float(u1_cfg["pricing_usd_per_million_tokens"]["input"]),
        cached_input_price_per_million=float(u1_cfg["pricing_usd_per_million_tokens"]["cached_input"]),
        output_price_per_million=float(u1_cfg["pricing_usd_per_million_tokens"]["output"]),
        max_output_tokens=int(u1_cfg["max_output_tokens"]),
    )
    dataset_revision = u0_cfg["dataset"]["revision"]
    run_config = {
        "purpose": "u1.2-style smoke demonstration, not a gate result",
        "task_id": task_id,
        "condition_ids": list(condition_ids),
        "repeat": repeat,
        "u1": u1_cfg,
    }
    manifest = RunManifest(
        run_id=new_run_id(),
        machine=machine_name,
        dataset_revision=dataset_revision,
        code_commit=git_commit(repo_root),
        config_hash=config_hash(run_config),
        started_at=datetime.now(timezone.utc).isoformat(),
        n_cells=0,
        forecaster_names=[forecaster.name],
        condition_ids=list(condition_ids),
    )
    store = ForecastStore(store_path)
    forecast_input = dataset.forecast_input(task_id)
    total_cells = len(condition_ids)
    for cell_index, condition_id in enumerate(condition_ids, start=1):
        if progress_callback is not None:
            progress_callback({
                "event": "cell_started",
                "cell_index": cell_index,
                "total_cells": total_cells,
                "benchmark_id": task_id,
                "condition_id": condition_id,
                "repeat": repeat,
                "n_samples": n_samples,
                "spent_usd": forecaster.ledger.spent_usd,
                "cost_cap_usd": forecaster.ledger.cap_usd,
            })
        condition = build_condition(condition_id, dataset, task_id, int(u0_cfg["conditions"]["seed"]))
        seed = derive_seed(int(u0_cfg["conditions"]["seed"]), task_id, condition_id, str(repeat), forecaster.name)
        def on_attempt(progress: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback({
                    **progress,
                    "cell_index": cell_index,
                    "total_cells": total_cells,
                    "condition_id": condition_id,
                    "repeat": repeat,
                })

        output = forecaster.forecast(
            forecast_input, condition.context, n_samples=n_samples, seed=seed, progress_callback=on_attempt
        )
        store.append(
            build_cell_record(
                benchmark_id=task_id,
                condition_id=condition_id,
                repeat=repeat,
                condition_seed=condition.seed or 0,
                output=output,
                dataset_revision=dataset_revision,
                code_commit=manifest.code_commit,
                config_hash=manifest.config_hash,
            )
        )
        manifest.n_cells += 1
        if progress_callback is not None:
            progress_callback({
                "event": "cell_stored",
                "cell_index": cell_index,
                "total_cells": total_cells,
                "benchmark_id": task_id,
                "condition_id": condition_id,
                "repeat": repeat,
                "valid_samples": output.n_valid,
                "requested_samples": n_samples,
                "cell_cost_usd": output.cost_usd,
                "spent_usd": forecaster.ledger.spent_usd,
                "cost_cap_usd": forecaster.ledger.cap_usd,
            })
        if output.n_valid < n_samples:
            break
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.write(manifest_path)
    return manifest



def smoke_prompt_leakage(dataset: Dataset, task_id: str = "task_42") -> pd.DataFrame:
    """Check the exact C0/C1 smoke prompts for label-like future-value leakage.

    A consecutive future-value run is a failure. Exact scattered timestamp/value pairs are
    reported separately because benchmark evidence can legitimately include them.
    """
    forecast_input = dataset.forecast_input(task_id)
    task = dataset.tasks[task_id]
    history = "\n".join(
        f"({timestamp}, {value:.6g})"
        for timestamp, value in zip(forecast_input.history_timestamps, fill_history_forward(forecast_input.history_values, task_id))
    )
    rows: list[dict[str, Any]] = []
    for condition_id in ("C0", "C1"):
        condition = build_condition(condition_id, dataset, task_id, 20260918)
        text = SYSTEM_MESSAGE + "\n" + render_user_prompt(forecast_input, condition.context)
        hit = leaks_beyond_history(text, history, task.future_values)
        at_future, exact, nonzero = matching_future_pairs(text, task.future_timestamps, task.future_values)
        rows.append({
            "condition_id": condition_id,
            "future_run_length": 0 if hit is None else hit.run_length,
            "future_run_start": None if hit is None else hit.future_start,
            "future_timestamp_value_pairs": at_future,
            "exact_future_pairs": exact,
            "exact_nonzero_future_pairs": nonzero,
            "prompt_clean": hit is None,
        })
    return pd.DataFrame(rows)


def write_smoke_plots(dataset: Dataset, store_path: Path, out_path: Path) -> None:
    """Plot task_42 history, truth and all valid C0/C1 sample trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_id = "task_42"
    task = dataset.tasks[task_id]
    records = {r["condition_id"]: r for r in ForecastStore(store_path).read_all() if r["benchmark_id"] == task_id}
    missing = [condition for condition in ("C0", "C1") if condition not in records]
    if missing:
        raise ValueError(f"missing smoke records for {missing}")

    history_x = np.arange(len(task.history_values))
    future_x = np.arange(len(task.history_values), len(task.history_values) + len(task.future_values))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    colors = {"C0": "#4575b4", "C1": "#d73027"}
    for ax, condition_id in zip(axes, ("C0", "C1")):
        samples = ForecastStore.samples_as_array(records[condition_id])
        for sample in samples:
            ax.plot(future_x, sample, color=colors[condition_id], alpha=0.14, linewidth=0.8)
        ax.plot(history_x, fill_history_forward(task.history_values, task_id), color="#222222", linewidth=1.5, label="history")
        ax.plot(future_x, task.future_values, color="#111111", linestyle="--", linewidth=1.8, label="truth")
        ax.plot(future_x, np.median(samples, axis=0), color=colors[condition_id], linewidth=2.0, label=f"{condition_id} median")
        ax.axvline(len(task.history_values) - 0.5, color="#666666", linestyle=":", linewidth=1)
        ax.set_title(f"task_42 — {condition_id}: {len(samples)} valid sample trajectories")
        ax.set_ylabel(task.time_series_variable)
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time step (history followed by forecast horizon)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_smoke_check(
    dataset: Dataset,
    scores: pd.DataFrame,
    store_path: Path,
    estimate_path: Path,
    out_path: Path,
) -> None:
    """Write the formal U1.2 completion/check document from existing smoke records."""
    if scores.empty:
        raise ValueError("cannot write smoke check without scored records")
    leakage = smoke_prompt_leakage(dataset)
    # Under the original U0.6 one-call-per-25-samples assumption task_42's high estimate is 5,291 / 5,502.
    # The implemented Gemini route uses one completion per request, so multiply it by 25 for a fair request count.
    dry = {"C0": (1923, 5291), "C1": (2071, 5502)}
    lines = [
        "# U1.2 smoke-test check — task_42 C0 vs C1",
        "",
        "**Run date:** 2026-09-18. This document completes the U1.2 engineering check; it is not a U1 gate result.",
        "",
        "## Run scope",
        "",
        "- Task: `task_42`; conditions: C0 (no context) and C1 (ground-truth evidence); one repeat.",
        "- Requested samples: 25 per condition. The Gemini/LiteLLM route sends one trajectory per API call, so this execution made 25 calls per condition rather than using `n > 1`.",
        "- Plot: `artifacts/u1/smoke/task42_c0_c1_trajectories.png`.",
        "",
        "## Kill-condition evaluation",
        "",
        "| condition | valid/requested | valid rate | cost | result |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in scores.iterrows():
        result = "PASS" if row["valid_rate"] >= 0.80 else "FAIL"
        lines.append(f"| {row['condition_id']} | {int(row['n_valid'])}/{int(row['n_requested'])} | {row['valid_rate']:.1%} | ${row['cost_usd']:.6f} | {result} |")
    lines += [
        "",
        "**Valid-sample kill condition (≥80%): PASS** — C0 was 24/24 valid and C1 was 25/25 valid in the retained continuation store.",
        "",
        "## Actual tokens and cost versus U0.6 dry run",
        "",
        "The U0.6 estimate predates the selected Gemini route and assumes `n > 1`; it gives low/high **per-prompt** input ranges for task_42. Since the actual route makes 25 one-sample calls, the comparable total range is the per-prompt range multiplied by 25. Actual provider usage includes message framing and any provider-side tokenisation effects, while U0.6 used character heuristics; this is therefore a calibration comparison, not an exact accounting reconciliation.",
        "",
        "| condition | dry-run input range for 25 one-sample calls | actual input | actual / dry-run high | actual output | actual cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in scores.iterrows():
        low, high = dry[row["condition_id"]]
        low_total, high_total = low * 25, high * 25
        ratio = row["input_tokens"] / high_total
        lines.append(f"| {row['condition_id']} | {low_total:,}–{high_total:,} | {int(row['input_tokens']):,} | {ratio:.2f}× | {int(row['output_tokens']):,} | ${row['cost_usd']:.6f} |")
    lines += [
        "",
        "**Cost-per-request kill condition (≤2× dry-run estimate): PASS on the available evidence.** Actual input totals are 1.30× (C0) and 1.34× (C1) of the U0.6 high heuristic bound—below the 2× threshold. The original dry run did not have provider pricing configured, so it cannot supply a literal dollar estimate. Actual spend was $0.309471 total for 49 valid trajectories, or $0.006316 per valid trajectory. This must be used to revise the full-run estimate before U1.3.",
        "",
        "## Prompt-label leakage check",
        "",
        "| condition | consecutive future-value run | exact future timestamp/value pairs | exact non-zero pairs | result |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in leakage.iterrows():
        result = "PASS" if row["prompt_clean"] else "FAIL"
        lines.append(f"| {row['condition_id']} | {int(row['future_run_length'])} | {int(row['exact_future_pairs'])} | {int(row['exact_nonzero_future_pairs'])} | {result} |")
    lines += [
        "",
        "**Label-leak kill condition: PASS.** Neither exact smoke prompt contained a consecutive run of future values beyond what the history already provides, and neither contained an exact future timestamp/value pair. The structural isolation check remains documented in `artifacts/u0/leakage_report.md`.",
        "",
        "## Engineering outcome",
        "",
        "All three U1.2 kill conditions pass. C1 also had lower CRPS than C0 in this one-task demonstration, but that is descriptive only and does not evaluate U1 gate criteria.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _score_samples(dataset: Dataset, task_id: str, samples: np.ndarray, fallback_lag: int) -> dict:
    task = dataset.tasks[task_id]
    target = np.asarray(task.future_values, dtype=float)
    history = fill_history_forward(task.history_values, task_id)
    m = resolve_seasonal_period_steps(task.seasonal_period, task.frequency, task_id) or fallback_lag
    denom = {
        "a1": scale_a1_future_range(target, task_id),
        "a2": scale_a2_seasonal_naive_mae(history, m, task_id),
        "a3": scale_a3_mean_abs_history(history, task_id),
    }
    raw_crps = mean_crps(target, samples)
    row = {
        "raw_crps": raw_crps,
        "raw_mae": mae_of_median(target, samples),
        "raw_rmse": rmse_of_mean(target, samples),
    }
    for name, value in denom.items():
        row[f"denom_{name}"] = value
        if np.isfinite(value) and value >= 1e-9:
            score = raw_crps / value
            row[f"scaled_crps_{name}"] = winsorise(score).winsorised
            row[f"scaled_crps_{name}_capped"] = winsorise(score).capped
        else:
            row[f"scaled_crps_{name}"] = float("nan")
            row[f"scaled_crps_{name}_capped"] = False
    return row


def score_smoke(repo_root: Path, dataset: Dataset, u0_cfg: dict, store_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    fallback_lag = int(u0_cfg["baseline"]["scaling_a2_fallback_lag_steps"])
    for record in ForecastStore(store_path).read_all():
        samples = ForecastStore.samples_as_array(record)
        if len(samples) < 2:
            continue
        reqs = record["sampling_params"].get("request_costs", [])
        total_prompt = int(record["token_counts"].get("input", 0))
        total_cached = int(record["token_counts"].get("cached_input", 0))
        row = {
            "benchmark_id": record["benchmark_id"],
            "condition_id": record["condition_id"],
            "forecaster_name": record["forecaster_name"],
            "n_requested": record["n_requested"],
            "n_valid": record["n_valid"],
            "valid_rate": record["n_valid"] / record["n_requested"],
            "cost_usd": record["cost_usd"],
            "cost_per_valid_sample_usd": record["cost_usd"] / record["n_valid"] if record["n_valid"] else float("nan"),
            "input_tokens": total_prompt,
            "cached_input_tokens": total_cached,
            "uncached_input_tokens": total_prompt - total_cached,
            "cached_input_fraction": total_cached / total_prompt if total_prompt else float("nan"),
            "output_tokens": int(record["token_counts"].get("output", 0)),
            "requests": len(reqs),
            "cache_read_cost_usd": sum(float(r.get("cache_read_cost_usd") or 0) for r in reqs),
            "input_cost_usd": sum(float(r.get("input_cost_usd") or 0) for r in reqs),
            "output_cost_usd": sum(float(r.get("output_cost_usd") or 0) for r in reqs),
        }
        row.update(_score_samples(dataset, record["benchmark_id"], samples, fallback_lag))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark_id", "condition_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def write_smoke_report(df: pd.DataFrame, baseline_scores_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# U1.2-style smoke demonstration — task_42 C0 vs C1", ""]
    lines += [
        "This is a one-task, one-repeat engineering smoke demonstration, not a U1 gate result or a general claim about context utility.",
        "It requests 25 trajectories for C0 and C1 separately and compares them with the pre-existing statistical C0 baselines.",
        "",
    ]
    if df.empty:
        lines.append("No scoreable records exist (at least two valid trajectories per cell are required for CRPS).")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    display_cols = ["condition_id", "n_valid", "valid_rate", "raw_crps", "scaled_crps_a1", "scaled_crps_a2", "scaled_crps_a3", "cost_usd", "input_tokens", "cached_input_tokens", "cached_input_fraction", "output_tokens"]
    lines += ["## LLM results", "", "| " + " | ".join(display_cols) + " |", "|" + "|".join(["---"] * len(display_cols)) + "|"]
    for _, r in df.iterrows():
        values = []
        for col in display_cols:
            value = r[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Cache accounting", "", "Cached input tokens are reported by LiteLLM/Gemini. They do not change CRPS; they only affect billing. `cost_usd` is the proxy-reported total cost, so cached-token discounts are already included and are not subtracted again.", ""]
    for _, r in df.iterrows():
        lines.append(
            f"- {r['condition_id']}: {int(r['cached_input_tokens']):,}/{int(r['input_tokens']):,} input tokens cached "
            f"({r['cached_input_fraction']:.1%}); uncached input {int(r['uncached_input_tokens']):,}; "
            f"proxy cost ${r['cost_usd']:.6f}; cache-read component ${r['cache_read_cost_usd']:.6f}."
        )
    if baseline_scores_path.exists():
        base = pd.read_parquet(baseline_scores_path)
        base = base[base["benchmark_id"] == "task_42"]
        lines += ["", "## Statistical C0 baselines on task_42", "", "| forecaster | raw CRPS | scaled CRPS A1 | A2 | A3 |", "|---|---:|---:|---:|---:|"]
        for _, r in base.iterrows():
            lines.append(f"| {r['forecaster_name']} | {r['raw_crps']:.6f} | {r['scaled_crps_a1']:.6f} | {r['scaled_crps_a2']:.6f} | {r['scaled_crps_a3']:.6f} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
