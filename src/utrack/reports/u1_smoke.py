"""Run and report the controlled U1.2-style task_42 C0/C1 smoke demonstration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utrack.conditions.builders import build_condition
from utrack.data.loader import Dataset, fill_history_forward
from utrack.forecasters.base import resolve_seasonal_period_steps
from utrack.forecasters.llm_direct import LiteLLMDirectForecaster
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
    for condition_id in condition_ids:
        condition = build_condition(condition_id, dataset, task_id, int(u0_cfg["conditions"]["seed"]))
        seed = derive_seed(int(u0_cfg["conditions"]["seed"]), task_id, condition_id, str(repeat), forecaster.name)
        output = forecaster.forecast(forecast_input, condition.context, n_samples=n_samples, seed=seed)
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
        if output.n_valid < n_samples:
            break
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.write(manifest_path)
    return manifest


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
