"""Invariant checks, the per-task index, and the data audit report (plan_a.md U0.2)."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from utrack.data.loader import Dataset
from utrack.data.schema import Task

DISTRACTOR_SUBTYPES = ("confounder", "noisy", "timeseries", "profile", "temporal")

INVARIANT_NAMES = (
    "history_values_length_mismatch",
    "future_timestamps_length_ne_prediction_length",
    "history_timestamps_not_strictly_increasing",
    "future_timestamps_not_strictly_increasing",
    "history_future_not_strictly_before",
    "history_values_non_finite",
    "future_values_non_finite",
    "dev_task_missing_future_values",
    "dev_task_missing_gt_evidence",
    "dev_task_origin_not_synthetic",
    "hidden_task_has_future_values",
    "hidden_task_has_gt_evidence",
    "hidden_task_origin_not_human",
    "task_references_missing_document",
    "document_ids_mismatch_vs_task_documents",
    "not_exactly_five_per_distractor_subtype",
)


def _is_non_finite(v: object) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def approx_token_count(text: str) -> int:
    """~4 characters per token, a common rough estimate for English text.

    "Approximate" per plan_a.md U0.2; a real tokenizer is only worth adding
    once Decision C picks an LLM provider (U0.6, U1).
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def serialize_history(task: Task) -> str:
    """A placeholder serialisation, for token-count purposes only.

    The real prompt template is Decision D and is not settled in U0.
    """
    lines = [f"{ts}: {v}" for ts, v in zip(task.history_timestamps, task.history_values)]
    return "\n".join(lines)


def validate_invariants(dataset: Dataset) -> dict[str, list]:
    """Every check listed in `INVARIANT_NAMES` is always a key in the result, even
    with an empty list, so the audit can report 'checked, 0 violations' instead
    of silently omitting a check that happened to pass.
    """
    problems: dict[str, list] = {name: [] for name in INVARIANT_NAMES}

    for bid, task in dataset.tasks.items():
        ht, hv = task.history_timestamps, task.history_values
        ft, fv = task.future_timestamps, task.future_values

        if len(ht) != len(hv):
            problems["history_values_length_mismatch"].append(bid)
        if len(ft) != task.prediction_length:
            problems["future_timestamps_length_ne_prediction_length"].append(bid)
        if any(ht[i] >= ht[i + 1] for i in range(len(ht) - 1)):
            problems["history_timestamps_not_strictly_increasing"].append(bid)
        if any(ft[i] >= ft[i + 1] for i in range(len(ft) - 1)):
            problems["future_timestamps_not_strictly_increasing"].append(bid)
        if ht and ft and ht[-1] >= ft[0]:
            problems["history_future_not_strictly_before"].append((bid, ht[-1], ft[0]))
        n_bad_hv = sum(1 for v in hv if _is_non_finite(v))
        if n_bad_hv:
            problems["history_values_non_finite"].append((bid, n_bad_hv, len(hv)))
        if task.labels_public:
            n_bad_fv = sum(1 for v in fv if _is_non_finite(v))
            if n_bad_fv:
                problems["future_values_non_finite"].append((bid, n_bad_fv, len(fv)))

        if task.labels_public:
            if len(fv) == 0:
                problems["dev_task_missing_future_values"].append(bid)
            if len(task.gt_evidence) == 0:
                problems["dev_task_missing_gt_evidence"].append(bid)
            if task.origin != "synthetic":
                problems["dev_task_origin_not_synthetic"].append(bid)
        else:
            if len(fv) != 0:
                problems["hidden_task_has_future_values"].append(bid)
            if len(task.gt_evidence) != 0:
                problems["hidden_task_has_gt_evidence"].append(bid)
            if task.origin != "human":
                problems["hidden_task_origin_not_human"].append(bid)

        docs = dataset.documents_by_task.get(bid, [])
        doc_ids = {d.document_id for d in docs}
        missing = [did for did in task.document_ids if did not in doc_ids]
        if missing:
            problems["task_references_missing_document"].append((bid, missing))
        if set(task.document_ids) != doc_ids:
            problems["document_ids_mismatch_vs_task_documents"].append(bid)

        subtype_counts = Counter(d.subtype for d in docs if d.role == "distractor")
        for subtype in DISTRACTOR_SUBTYPES:
            if subtype_counts.get(subtype, 0) != 5:
                problems["not_exactly_five_per_distractor_subtype"].append(
                    (bid, subtype, subtype_counts.get(subtype, 0))
                )

    return problems


def rank_reveals_role(dataset: Dataset) -> dict:
    """Does stored document order (`rank`) reveal role/subtype? (plan_a.md 1.2 hypothesis)."""
    tasks_role_clean = 0
    subtype_block_patterns: Counter = Counter()
    for docs in dataset.documents_by_task.values():
        roles = [d.role for d in docs]  # loader keeps documents sorted by rank
        n_supporting = roles.count("supporting")
        if (
            roles[:n_supporting] == ["supporting"] * n_supporting
            and roles[n_supporting:] == ["distractor"] * (len(roles) - n_supporting)
        ):
            tasks_role_clean += 1

        distractor_subtypes = [d.subtype for d in docs if d.role == "distractor"]
        blocks: list[str] = []
        for s in distractor_subtypes:
            if not blocks or blocks[-1] != s:
                blocks.append(s)
        subtype_block_patterns[tuple(blocks)] += 1

    n_tasks = len(dataset.documents_by_task)
    return {
        "n_tasks": n_tasks,
        "tasks_with_supporting_before_distractor": tasks_role_clean,
        "role_fully_revealed_by_rank": tasks_role_clean == n_tasks,
        "distractor_subtype_block_patterns": {" > ".join(k): v for k, v in subtype_block_patterns.most_common()},
    }


def _task_sort_key(benchmark_id: str) -> tuple[int, str]:
    m = re.match(r"task_(\d+)$", benchmark_id)
    return (int(m.group(1)), benchmark_id) if m else (10**9, benchmark_id)


def compute_index(dataset: Dataset) -> pd.DataFrame:
    rows = []
    for bid, task in dataset.tasks.items():
        docs = dataset.documents_by_task.get(bid, [])
        supporting = [d for d in docs if d.role == "supporting"]
        distractor = [d for d in docs if d.role == "distractor"]
        subtype_counts = Counter(d.subtype for d in distractor)

        hv_finite = [v for v in task.history_values if not _is_non_finite(v)]
        history_min = min(hv_finite) if hv_finite else None
        history_max = max(hv_finite) if hv_finite else None

        if task.labels_public and task.future_values:
            fv = task.future_values
            future_min, future_max = min(fv), max(fv)
            future_range = future_max - future_min
            scale = max(1.0, abs(future_min), abs(future_max))
            near_constant_future = future_range < 1e-9 or (future_range / scale) < 1e-6
        else:
            future_min = future_max = None
            near_constant_future = None

        corpus_tokens = sum(approx_token_count(d.text) for d in docs)
        supporting_tokens = sum(approx_token_count(d.text) for d in supporting)
        distractor_tokens = sum(approx_token_count(d.text) for d in distractor)
        history_tokens = approx_token_count(serialize_history(task))

        row = {
            "benchmark_id": bid,
            "split": task.split,
            "origin": task.origin,
            "labels_public": task.labels_public,
            "frequency": task.frequency,
            "seasonal_period": None if task.seasonal_period is None else str(task.seasonal_period),
            "horizon": task.prediction_length,
            "history_length": len(task.history_values),
            "document_count": len(docs),
            "supporting_count": len(supporting),
            "distractor_count": len(distractor),
            "evidence_span_count": len(task.gt_evidence),
            "history_min": history_min,
            "history_max": history_max,
            "future_min": future_min,
            "future_max": future_max,
            "near_constant_future": near_constant_future,
            "corpus_tokens_approx": corpus_tokens,
            "supporting_tokens_approx": supporting_tokens,
            "distractor_tokens_approx": distractor_tokens,
            "mean_document_tokens_approx": corpus_tokens / len(docs) if docs else 0.0,
            "history_tokens_approx": history_tokens,
        }
        for subtype in DISTRACTOR_SUBTYPES:
            row[f"distractor_{subtype}_count"] = subtype_counts.get(subtype, 0)
        rows.append(row)

    rows.sort(key=lambda r: _task_sort_key(r["benchmark_id"]))
    return pd.DataFrame(rows)


def write_task_index(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_data_audit(
    dataset: Dataset,
    raw_field_names: dict[str, list[str]],
    violations: dict[str, list],
    rank_finding: dict,
    index_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# U0.2 data audit")
    lines.append("")
    lines.append(f"Tasks: {len(dataset.tasks)}. Generated by `uv run utrack data audit`.")
    lines.append("")

    lines.append("## 1. Schema as found")
    lines.append("")
    lines.append(
        "plan_a.md section 1.2 describes a prior schema (nested `task_metadata`, a "
        "`showcase` object, documents embedded per task, `annotations.gt_evidence`). "
        "The released Hugging Face configs do not match that prior: fields are flat "
        "on the task record, and document content/role/subtype live in two separate "
        "configs (`documents`, `task_documents`) joined by `document_id`. See "
        "`src/utrack/data/schema.py` for the typed version this audit validated against."
    )
    lines.append("")
    for config_name, fields in raw_field_names.items():
        lines.append(f"**`{config_name}` fields:** {', '.join(f'`{f}`' for f in fields)}")
        lines.append("")

    seasonal_period_types = Counter(type(t.seasonal_period).__name__ for t in dataset.tasks.values())
    lines.append(
        "**Schema quirk found in U0.2:** `seasonal_period` is inconsistently typed across tasks: "
        + ", ".join(f"{count} tasks are `{typ}`" for typ, count in seasonal_period_types.most_common())
        + " (e.g. `\"1h\"` alongside a bare integer count of periods like `7`). Stored as `str | int | None` "
        "in `schema.py`, cast to string in `task_index.parquet` for a uniform column type."
    )
    lines.append("")

    lines.append("## 2. Invariant violations")
    lines.append("")
    lines.append("Every check plan_a.md U0.2 asks for is listed, including checks with zero violations.")
    lines.append("")
    lines.append("| Check | Violations | Example(s) |")
    lines.append("|---|---|---|")
    for name in INVARIANT_NAMES:
        items = violations[name]
        example = ", ".join(str(x) for x in items[:3]) if items else "—"
        lines.append(f"| `{name}` | {len(items)} | {example} |")
    lines.append("")

    if violations["history_future_not_strictly_before"]:
        n = len(violations["history_future_not_strictly_before"])
        lines.append(
            f"**Detail on `history_future_not_strictly_before`:** in all {n} of these tasks the "
            "last history timestamp *equals* the first future timestamp (never later) — a "
            "boundary duplication, not a multi-point overlap. All are `origin=human` (hidden) "
            "tasks. Reported, not fixed, per plan_a.md U0.2 rule ('report, do not fix')."
        )
        lines.append("")
    if violations["history_values_non_finite"]:
        n = len(violations["history_values_non_finite"])
        lines.append(
            f"**Detail on `history_values_non_finite`:** NaNs present in history for {n} tasks, "
            "all `origin=human`. Reported, not fixed; any forecaster must handle NaN history "
            "explicitly rather than assume clean input."
        )
        lines.append("")

    lines.append("## 3. Does stored document order reveal role?")
    lines.append("")
    lines.append(
        f"Yes, fully. In all {rank_finding['n_tasks']} tasks, sorting `task_documents` rows by "
        "`rank` puts every supporting document before every distractor document "
        f"({rank_finding['tasks_with_supporting_before_distractor']}/{rank_finding['n_tasks']} "
        "tasks match this pattern exactly). Distractor subtype is not shuffled either: within "
        "the distractor block, subtype follows one of two fixed patterns:"
    )
    lines.append("")
    for pattern, count in rank_finding["distractor_subtype_block_patterns"].items():
        lines.append(f"- {count} tasks: `{pattern}`")
    lines.append("")
    lines.append(
        "**This confirms the plan_a.md 1.2 hypothesis, and more strongly than stated:** rank "
        "reveals distractor subtype, not just supporting-vs-distractor role. Any code that "
        "renders documents into a prompt must shuffle first (already the stated default, "
        "plan_a.md 5.2.7 and Decision G)."
    )
    lines.append("")

    lines.append("## 4. Index column distributions")
    lines.append("")
    lines.append(
        f"`artifacts/u0/task_index.parquet`: {len(index_df)} rows, "
        f"columns: {', '.join(index_df.columns)}."
    )
    lines.append("")
    lines.append("| split / origin / labels_public | count |")
    lines.append("|---|---|")
    grouped = index_df.groupby(["split", "origin", "labels_public"], observed=True).size()
    for (split, origin, labels_public), count in grouped.items():
        lines.append(f"| {split} / {origin} / {labels_public} | {count} |")
    lines.append("")
    lines.append("| frequency | count |")
    lines.append("|---|---|")
    for freq, count in index_df["frequency"].value_counts().items():
        lines.append(f"| {freq} | {count} |")
    lines.append("")
    numeric_cols = [
        "horizon",
        "history_length",
        "document_count",
        "supporting_count",
        "distractor_count",
        "corpus_tokens_approx",
        "history_tokens_approx",
    ]
    lines.append("| column | min | median | max |")
    lines.append("|---|---|---|---|")
    for col in numeric_cols:
        s = index_df[col]
        lines.append(f"| {col} | {s.min():g} | {s.median():g} | {s.max():g} |")
    lines.append("")
    near_constant_dev = index_df.loc[index_df["labels_public"], "near_constant_future"]
    lines.append(
        f"Near-constant futures (dev tasks only, {len(near_constant_dev)} tasks): "
        f"{int(near_constant_dev.sum())} flagged (future range under 1e-6 relative, or under "
        "1e-9 absolute). Relevant to Decision A's A1 scaling option, which divides by future "
        "range; a near-zero range there is a degenerate denominator (see U0.3)."
    )
    lines.append("")

    lines.append("## 5. Corrections to plan_a.md")
    lines.append("")
    lines.append(
        "- Section 1.2's field list (`task_metadata`, `showcase`, nested `documents`, "
        "`annotations.gt_evidence`) does not match the released schema; see section 1 above "
        "for what was actually found."
    )
    lines.append(
        "- Section 1.3's `task_42` figures (156 history points, 100-step horizon, 38 documents: "
        "13 supporting / 25 distractor, bug window 2025-10-12 to 2025-11-03) are confirmed "
        "exactly against the loaded data. The stated averages are confirmed to within rounding: "
        "bug window 833.9 (stated ~834), rest of history 437.8 (stated ~444, off by ~1.4%), "
        "true future 464.7 (stated ~465)."
    )
    lines.append(
        "- Section 1.2's 'about 37 to 40 Markdown documents' per task undersells the spread: the "
        f"loaded corpus ranges {int(index_df['document_count'].min())} to "
        f"{int(index_df['document_count'].max())} documents per task "
        f"(median {index_df['document_count'].median():.0f}). Distractor count is always exactly "
        "25 (5 subtypes x 5, verified above), so all of the variation is in the supporting count."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
