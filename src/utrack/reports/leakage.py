"""U0.5 leakage scan over every dev task and condition, and its report.

This module reads `future_values` on purpose (it is the checker, not a builder); the condition
builders themselves are structurally forbidden to (see conditions/leakage.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from utrack.conditions import builders as builders_mod
from utrack.conditions import placebo as placebo_mod
from utrack.conditions import preview as preview_mod
from utrack.conditions import render as render_mod
from utrack.conditions.builders import build_condition
from utrack.conditions.leakage import (
    DEFAULT_MIN_RUN,
    MIN_DISTINCT,
    REL_TOL,
    find_future_run,
    history_text,
    is_untestable,
    label_fields_on_forecast_input,
    label_references_in_source,
    matching_future_pairs,
    metadata_text,
)
from utrack.data.loader import Dataset
from utrack.forecasters import llm_prompt as llm_prompt_mod

SCAN_SURFACES = ("metadata", "history", "C1", "C2", "C3", "C4")  # C0 adds no text beyond the ForecastInput
BUILDER_MODULES = (builders_mod, placebo_mod, render_mod, preview_mod, llm_prompt_mod)


def scan_dev_tasks(dataset: Dataset, base_seed: int, min_run: int = DEFAULT_MIN_RUN) -> pd.DataFrame:
    rows = []
    for benchmark_id in dataset.dev_task_ids():
        future = dataset.tasks[benchmark_id].future_values
        forecast_input = dataset.forecast_input(benchmark_id)
        texts = {"metadata": metadata_text(forecast_input), "history": history_text(forecast_input)}
        for condition_id in ("C1", "C2", "C3", "C4"):
            texts[condition_id] = build_condition(condition_id, dataset, benchmark_id, base_seed).context
        untestable = is_untestable(future, min_run)
        for surface, text in texts.items():
            hit = None if untestable else find_future_run(text, future, min_run)
            at_future, exact, exact_nonzero = matching_future_pairs(text, forecast_input.future_timestamps, future)
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "surface": surface,
                    "untestable": untestable,
                    "run_length": hit.run_length if hit else 0,
                    "future_start": hit.future_start if hit else -1,
                    "pairs_at_future_ts": at_future,
                    "exact_pairs": exact,
                    "exact_nonzero_pairs": exact_nonzero,
                    "horizon": len(future),
                }
            )
    return pd.DataFrame(rows)


def order_position_summary(dataset: Dataset, base_seed: int) -> dict[str, tuple[float, float]]:
    """Mean normalised position (0 first, 1 last) of supporting documents, averaged over dev tasks,
    under the stored rank order and under the C4 rendering. 0.5 means order carries no role signal."""
    stored, rendered = [], []
    for benchmark_id in dataset.dev_task_ids():
        documents = dataset.documents_by_task[benchmark_id]
        n = len(documents)
        role_by_id = {d.document_id: d.role for d in documents}
        stored.append(np.mean([i / (n - 1) for i, d in enumerate(documents) if d.role == "supporting"]))
        c4 = build_condition("C4", dataset, benchmark_id, base_seed)
        rendered.append(
            np.mean([i / (n - 1) for i, did in enumerate(c4.document_ids) if role_by_id[did] == "supporting"])
        )

    def mean_se(values: list[float]) -> tuple[float, float]:
        return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))

    return {"stored rank order": mean_se(stored), "C4 rendered order": mean_se(rendered)}


def _pair_section(scan: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "### Exact future values given at scattered timestamps",
        "",
        "The run scan needs consecutive values, so it cannot see a text that hands over the true value at some "
        "future timestamps, for example one every third hour. This check reads every `(timestamp, value)` pair "
        "in a text and compares those at a future timestamp with the true value there. It is the benchmark's "
        "intended context (plan_a.md U0.5), so it is reported, not asserted; it matters when reading utility, "
        "because on these tasks the evidence condition is handed part of the answer.",
        "",
        "| surface | tasks with a pair at a future timestamp | tasks with an exact match | most steps matched (share of horizon) |",
        "|---|---|---|---|",
    ]
    for surface in SCAN_SURFACES:
        part = scan[scan["surface"] == surface]
        matched = part[part["exact_pairs"] > 0]
        share = f"{(matched['exact_pairs'] / matched['horizon']).max():.0%}" if len(matched) else "-"
        lines.append(f"| {surface} | {int((part['pairs_at_future_ts'] > 0).sum())} | {len(matched)} | {share} |")
    matched = scan[(scan["exact_pairs"] > 0) & (scan["surface"] != "history")].sort_values(["surface", "benchmark_id"])
    if len(matched):
        lines += [
            "",
            "Zeros are cheap to guess (night-time irradiance is 0 whatever the evidence says), so the non-zero "
            "count is the informative one.",
            "",
            "| task | surface | future steps given exactly | of which non-zero | horizon |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['benchmark_id']} | {r['surface']} | {r['exact_pairs']} | {r['exact_nonzero_pairs']} | {r['horizon']} |"
            for _, r in matched.iterrows()
        ]
    return lines


def write_leakage_report(
    scan: pd.DataFrame, order_summary: dict[str, tuple[float, float]], min_run: int, out_path: Path
) -> None:
    n_tasks = scan["benchmark_id"].nunique()
    label_fields = label_fields_on_forecast_input()
    references = {m.__name__: label_references_in_source(m) for m in BUILDER_MODULES}
    lines = [
        "# U0.5 leakage report",
        "",
        f"Scanned {n_tasks} dev tasks. Generated by `uv run utrack conditions leakage`.",
        "",
        "## 1. Structural checks",
        "",
        f"- Label fields present on `ForecastInput`: {sorted(label_fields) if label_fields else 'none'}.",
    ]
    for module_name, refs in references.items():
        lines.append(f"- `{module_name}` references to `future_values` / `TaskLabels` / `.labels`: {refs if refs else 'none'}.")
    lines += [
        "",
        "## 2. Text scan for runs of future values",
        "",
        f"A hit is a run of at least {min_run} consecutive future values, containing at least {MIN_DISTINCT} "
        f"distinct values, that appear as consecutive numbers in the text (relative tolerance {REL_TOL:g}, "
        "timestamps removed, thousands separators tried both ways).",
        "",
        "- `metadata`: every `ForecastInput` field except the history, as JSON. **Asserted clean.**",
        "- `history`: the history in the CiK prompt's `(timestamp, value)` format. A hit means the future repeats "
        "a stretch of the history exactly. The forecaster sees the history anyway, so this is reported, not a leak.",
        "- C1 to C4: text from the benchmark's own evidence and documents, which may legitimately contain numbers. "
        "Reported, not asserted.",
        "",
        "| surface | tasks scanned | tasks with a hit | longest run found |",
        "|---|---|---|---|",
    ]
    testable = scan[~scan["untestable"]]
    for surface in SCAN_SURFACES:
        part = testable[testable["surface"] == surface]
        hits = part[part["run_length"] > 0]
        longest = int(hits["run_length"].max()) if len(hits) else 0
        lines.append(f"| {surface} | {len(part)} | {len(hits)} | {longest if len(hits) else '-'} |")

    hits = testable[testable["run_length"] > 0]
    lines += ["", "### Tasks with a hit", ""]
    if hits.empty:
        lines.append("None.")
    else:
        lines += ["| task | surface | run length | starts at future step | horizon |", "|---|---|---|---|---|"]
        for _, r in hits.iterrows():
            lines.append(f"| {r['benchmark_id']} | {r['surface']} | {r['run_length']} | {r['future_start']} | {r['horizon']} |")

    lines += _pair_section(scan)

    untestable = sorted(scan.loc[scan["untestable"], "benchmark_id"].unique())
    lines += [
        "",
        "### Tasks the scan cannot judge",
        "",
        f"Tasks whose future has no window of {min_run} values with {MIN_DISTINCT} distinct values: "
        + (", ".join(untestable) if untestable else "none."),
        "",
        "## 3. Does document order still reveal role?",
        "",
        "Mean normalised position of a task's supporting documents (0 = first, 1 = last), averaged over the "
        "dev tasks. 0.5 means position carries no role signal. U0.2 found stored order places every supporting "
        "document before every distractor.",
        "",
        "| order | mean position of supporting documents | standard error |",
        "|---|---|---|",
    ]
    for name, (mean, se) in order_summary.items():
        lines.append(f"| {name} | {mean:.3f} | {se:.3f} |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
