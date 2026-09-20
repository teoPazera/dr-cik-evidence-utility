"""Resumable staged U1 execution, scoring, cost ledger, and review artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import json

import numpy as np
import pandas as pd

from utrack.conditions.builders import build_condition
from utrack.data.loader import Dataset, fill_history_forward
from utrack.forecasters.base import resolve_seasonal_period_steps
from utrack.forecasters.llm_direct import CostLedger, LiteLLMDirectForecaster
from utrack.scoring.aggregate import winsorise
from utrack.scoring.crps import mae_of_median, mean_crps, rmse_of_mean
from utrack.scoring.scaling import scale_a1_future_range, scale_a2_seasonal_naive_mae, scale_a3_mean_abs_history
from utrack.scoring.utility import utility_b1_absolute, utility_b2_relative
from utrack.seeds import derive_seed
from utrack.store.forecast_store import ForecastStore, build_cell_record
from utrack.store.manifest import RunManifest, config_hash, git_commit, new_run_id

SCALINGS = ("a1", "a2", "a3")


def _cell_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (record["benchmark_id"], record["condition_id"], record["forecaster_name"], int(record["repeat"]))


def existing_keys(store: ForecastStore) -> set[tuple[str, str, str, int]]:
    return {_cell_key(r) for r in store.read_all()}


def _make_forecaster(u1_cfg: dict[str, Any], ledger: CostLedger) -> LiteLLMDirectForecaster:
    pricing = u1_cfg["pricing_usd_per_million_tokens"]
    sampling = u1_cfg["sampling"]
    return LiteLLMDirectForecaster(
        model=u1_cfg["model"], temperature=float(sampling["temperature"]),
        n_retries=int(u1_cfg.get("n_retries", 3)), cost_cap_usd=float(u1_cfg["cost_cap_usd"]),
        input_price_per_million=float(pricing["input"]),
        cached_input_price_per_million=float(pricing["cached_input"]),
        output_price_per_million=float(pricing["output"]), max_output_tokens=int(u1_cfg["max_output_tokens"]), ledger=ledger,
    )


def write_cost_ledger(store_path: Path, out_path: Path, cap_usd: float) -> dict[str, Any]:
    records = ForecastStore(store_path).read_all()
    rows = []
    for record in records:
        usage = record.get("token_counts", {})
        params = record.get("sampling_params", {})
        request_costs = params.get("request_costs", [])
        rows.append({
            "task": record["benchmark_id"], "condition": record["condition_id"], "repeat": record["repeat"],
            "cost_usd": float(record.get("cost_usd", 0)), "n_requested": int(record.get("n_requested", 0)),
            "n_valid": int(record.get("n_valid", 0)), "input_tokens": int(usage.get("input", 0)),
            "cached_input_tokens": int(usage.get("cached_input", 0)), "output_tokens": int(usage.get("output", 0)),
            "attempts": int(params.get("attempts", 0)), "request_count": len(request_costs),
        })
    frame = pd.DataFrame(rows)
    total = float(frame["cost_usd"].sum()) if not frame.empty else 0.0
    lines = ["# U1 live cost ledger", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "", f"- Cells stored: {len(frame)}", f"- Total recorded spend: **${total:.6f}**", f"- Emergency cap: **${cap_usd:.2f}**", f"- Remaining before cap: **${cap_usd - total:.6f}**", ""]
    if not frame.empty:
        lines += ["## By cell", "", "| task | condition | repeat | valid/requested | attempts | input | cached input | output | cost |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in frame.sort_values(["task", "condition", "repeat"]).iterrows():
            lines.append(f"| {r["task"]} | {r["condition"]} | {int(r["repeat"])} | {int(r["n_valid"] )}/{int(r["n_requested"])} | {int(r["attempts"])} | {int(r["input_tokens"]):,} | {int(r["cached_input_tokens"]):,} | {int(r["output_tokens"]):,} | ${r["cost_usd"]:.6f} |")
        lines += ["", "## By task and condition", "", "| task | condition | cells | valid/requested | cost |", "|---|---|---:|---:|---:|"]
        grouped = frame.groupby(["task", "condition"], as_index=False).agg(cells=("repeat", "count"), valid=("n_valid", "sum"), requested=("n_requested", "sum"), cost=("cost_usd", "sum"))
        for _, r in grouped.sort_values(["task", "condition"]).iterrows():
            lines.append(f"| {r["task"]} | {r["condition"]} | {int(r["cells"])} | {int(r["valid"] )}/{int(r["requested"])} | ${r["cost"]:.6f} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"cells": len(frame), "spent_usd": total, "remaining_usd": cap_usd - total}


def run_u1(
    repo_root: Path, dataset: Dataset, u0_cfg: dict[str, Any], u1_cfg: dict[str, Any], machine_name: str,
    store_path: Path, manifest_path: Path, ledger_path: Path, task_ids: list[str], condition_ids: list[str], repeats: list[int],
    resume: bool = True, progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RunManifest:
    store = ForecastStore(store_path)
    old = store.read_all()
    existing = existing_keys(store)
    previous_spend = sum(float(r.get("cost_usd", 0)) for r in old)
    ledger = CostLedger(cap_usd=float(u1_cfg["cost_cap_usd"]), spent_usd=previous_spend)
    forecaster = _make_forecaster(u1_cfg, ledger)
    sampling = u1_cfg["sampling"]
    n_samples = int(sampling["samples_per_cell"])
    planned = [(task, condition, repeat) for task in task_ids for condition in condition_ids for repeat in repeats]
    run_cfg = {"task_ids": task_ids, "condition_ids": condition_ids, "repeats": repeats, "u1": u1_cfg, "purpose": "staged U1"}
    manifest = RunManifest(new_run_id(), machine_name, u0_cfg["dataset"]["revision"], git_commit(repo_root), config_hash(run_cfg), datetime.now(timezone.utc).isoformat(), forecaster_names=[forecaster.name], condition_ids=condition_ids)
    try:
        for i, (task_id, condition_id, repeat) in enumerate(planned, start=1):
            key = (task_id, condition_id, forecaster.name, repeat)
            if resume and key in existing:
                if progress_callback:
                    progress_callback({"event": "cell_skipped", "cell_index": i, "total_cells": len(planned), "benchmark_id": task_id, "condition_id": condition_id, "repeat": repeat, "spent_usd": ledger.spent_usd, "cost_cap_usd": ledger.cap_usd, "message": "already in append-only store"})
                continue
            condition = build_condition(condition_id, dataset, task_id, int(u0_cfg["conditions"]["seed"]))
            forecast_input = dataset.forecast_input(task_id)
            seed = derive_seed(int(u0_cfg["conditions"]["seed"]), task_id, condition_id, str(repeat), forecaster.name)
            if progress_callback:
                progress_callback({"event": "cell_started", "cell_index": i, "total_cells": len(planned), "benchmark_id": task_id, "condition_id": condition_id, "repeat": repeat, "n_samples": n_samples, "spent_usd": ledger.spent_usd, "cost_cap_usd": ledger.cap_usd})
            def nested(event: dict[str, Any]) -> None:
                if progress_callback:
                    progress_callback({**event, "cell_index": i, "total_cells": len(planned), "condition_id": condition_id, "repeat": repeat})
            output = forecaster.forecast(forecast_input, condition.context, n_samples, seed, progress_callback=nested)
            store.append(build_cell_record(benchmark_id=task_id, condition_id=condition_id, repeat=repeat, condition_seed=condition.seed or 0, output=output, dataset_revision=manifest.dataset_revision, code_commit=manifest.code_commit, config_hash=manifest.config_hash))
            existing.add(key); manifest.n_cells += 1
            write_cost_ledger(store_path, ledger_path, ledger.cap_usd)
            if progress_callback:
                progress_callback({"event": "cell_stored", "cell_index": i, "total_cells": len(planned), "benchmark_id": task_id, "condition_id": condition_id, "repeat": repeat, "valid_samples": output.n_valid, "requested_samples": n_samples, "cell_cost_usd": output.cost_usd, "spent_usd": ledger.spent_usd, "cost_cap_usd": ledger.cap_usd})
            if output.n_valid < n_samples:
                break
    finally:
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        manifest.write(manifest_path)
        write_cost_ledger(store_path, ledger_path, ledger.cap_usd)
    return manifest


def score_u1(dataset: Dataset, u0_cfg: dict[str, Any], store_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fallback = int(u0_cfg["baseline"]["scaling_a2_fallback_lag_steps"])
    for record in ForecastStore(store_path).read_all():
        if int(record.get("n_valid", 0)) < 2:
            continue
        task_id = record["benchmark_id"]; task = dataset.tasks[task_id]
        samples = ForecastStore.samples_as_array(record); history = fill_history_forward(task.history_values, task_id); target = np.asarray(task.future_values, dtype=float)
        seasonal = resolve_seasonal_period_steps(task.seasonal_period, task.frequency, task_id) or fallback
        denoms = {"a1": scale_a1_future_range(target, task_id), "a2": scale_a2_seasonal_naive_mae(history, seasonal, task_id), "a3": scale_a3_mean_abs_history(history, task_id)}
        raw = mean_crps(target, samples)
        row = {"benchmark_id": task_id, "condition_id": record["condition_id"], "repeat": int(record["repeat"]), "forecaster_name": record["forecaster_name"], "n_requested": int(record["n_requested"]), "n_valid": int(record["n_valid"]), "valid_rate": int(record["n_valid"]) / int(record["n_requested"]), "cost_usd": float(record["cost_usd"]), "raw_crps": raw, "raw_mae_median": mae_of_median(target, samples), "raw_rmse_mean": rmse_of_mean(target, samples)}
        for name, denom in denoms.items():
            w = winsorise(raw / denom)
            row[f"denom_{name}"] = denom; row[f"scaled_crps_{name}"] = w.winsorised; row[f"scaled_crps_{name}_capped"] = w.capped
        rows.append(row)
    scores = pd.DataFrame(rows)
    if scores.empty: return scores
    base = scores[scores.condition_id == "C0"][["benchmark_id", "repeat", "forecaster_name", "scaled_crps_a3", "scaled_crps_a3_capped"]].rename(columns={"scaled_crps_a3":"baseline_scaled_crps_a3", "scaled_crps_a3_capped":"baseline_capped"})
    scores = scores.merge(base, on=["benchmark_id", "repeat", "forecaster_name"], how="left")
    scores["utility_b1_a3"] = scores.apply(lambda r: utility_b1_absolute(r.baseline_scaled_crps_a3, r.scaled_crps_a3) if pd.notna(r.baseline_scaled_crps_a3) else np.nan, axis=1)
    scores["utility_b2_a3"] = scores.apply(lambda r: utility_b2_relative(r.baseline_scaled_crps_a3, r.scaled_crps_a3) if pd.notna(r.baseline_scaled_crps_a3) else np.nan, axis=1)
    scores["utility_capped"] = scores["baseline_capped"].fillna(False) | scores["scaled_crps_a3_capped"]
    return scores


def write_u1_review(dataset: Dataset, scores: pd.DataFrame, store_path: Path, out_dir: Path, report_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {(r["benchmark_id"], r["condition_id"], int(r["repeat"])): r for r in ForecastStore(store_path).read_all()}
    for (task_id, repeat), group in scores.groupby(["benchmark_id", "repeat"]):
        task = dataset.tasks[task_id]; fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
        hist = fill_history_forward(task.history_values, task_id); hx=np.arange(len(hist)); fx=np.arange(len(hist), len(hist)+len(task.future_values))
        ax.plot(hx, hist, color="#222", label="history", linewidth=1.5); ax.plot(fx, task.future_values, color="#111", linestyle="--", label="truth", linewidth=1.7)
        for _, row in group.sort_values("condition_id").iterrows():
            record=records[(task_id,row.condition_id,int(repeat))]; samples=ForecastStore.samples_as_array(record)
            q10,q50,q90=np.quantile(samples,[.1,.5,.9],axis=0); color={"C0":"#4575b4","C1":"#d73027","C2":"#7b3294","C3":"#1b9e77"}.get(row.condition_id,"#555")
            # Every valid trajectory is visible in a faint line; the median is the robust central summary.
            for sample in samples:
                ax.plot(fx, sample, color=color, alpha=.075, linewidth=.65, zorder=1)
            ax.fill_between(fx,q10,q90,color=color,alpha=.10, zorder=2)
            ax.plot(fx,q50,color=color,linewidth=2.0,label=f"{row.condition_id} median",zorder=3)
        ax.axvline(len(hist)-.5,color="#666",linestyle=":"); ax.set_title(f"{task_id}, repeat {repeat}: full samples shown by 10–90% bands; line is median"); ax.legend(ncol=3); ax.grid(alpha=.2); ax.set_ylabel(task.time_series_variable); ax.set_xlabel("time step")
        fig.savefig(out_dir / f"{task_id}_repeat{repeat}.png", dpi=180); plt.close(fig)
    lines=["# U1 staged review", "", "Headline: A3 scaled CRPS; B1 utility = baseline minus contextual score (positive helps).", "", "| task | condition | repeat | valid/requested | A3 CRPS | B1 utility | B2 utility | cost |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in scores.sort_values(["benchmark_id","condition_id","repeat"]).iterrows():
        utility = "-" if pd.isna(r.utility_b1_a3) else f"{r.utility_b1_a3:.6f}"
        relative = "-" if pd.isna(r.utility_b2_a3) else f"{r.utility_b2_a3:.2%}"
        lines.append(f"| {r["benchmark_id"]} | {r["condition_id"]} | {int(r["repeat"])} | {int(r["n_valid"] )}/{int(r["n_requested"])} | {r["scaled_crps_a3"]:.6f} | {utility} | {relative} | ${r["cost_usd"]:.6f} |")
    lines += ["", "Plots show every valid trajectory as a faint line, plus 10–90% bands and the robust sample median. No post-outcome sample selection is applied."]
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
