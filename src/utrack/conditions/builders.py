"""Condition builders C0 to C4 (plan_a.md 3.3).

This module must never touch a label: it reads a task's evidence through `Dataset.evidence`
and its documents through `Dataset.documents_by_task`, and nothing else from the label side.
`tests/test_conditions.py` enforces that by inspecting this file's source.
"""

from __future__ import annotations

from utrack.conditions.base import CONDITION_NAMES, Condition
from utrack.conditions.placebo import assign_placebo, placebo_seed
from utrack.conditions.render import render_documents, render_evidence
from utrack.data.audit import approx_token_count
from utrack.data.loader import Dataset
from utrack.seeds import derive_seed


def _condition(
    condition_id: str,
    benchmark_id: str,
    context: str | None,
    *,
    document_ids: tuple[str, ...] = (),
    evidence_span_ids: tuple[str, ...] = (),
    source_benchmark_id: str | None = None,
    seed: int | None = None,
) -> Condition:
    return Condition(
        condition_id=condition_id,
        name=CONDITION_NAMES[condition_id],
        benchmark_id=benchmark_id,
        context=context,
        document_ids=document_ids,
        evidence_span_ids=evidence_span_ids,
        source_benchmark_id=source_benchmark_id,
        approx_tokens=approx_token_count(context) if context else 0,
        seed=seed,
    )


def build_condition(condition_id: str, dataset: Dataset, benchmark_id: str, base_seed: int) -> Condition:
    if condition_id not in CONDITION_NAMES:
        raise ValueError(f"unknown condition {condition_id!r}; expected one of {sorted(CONDITION_NAMES)}")
    documents = dataset.documents_by_task.get(benchmark_id, [])

    if condition_id == "C0":
        return _condition("C0", benchmark_id, None)

    if condition_id == "C1":
        spans = dataset.evidence(benchmark_id)
        if not spans:
            raise ValueError(f"{benchmark_id} has no gt_evidence (hidden task), so C1 cannot be built")
        text, span_ids = render_evidence(spans)
        return _condition("C1", benchmark_id, text, evidence_span_ids=span_ids)

    if condition_id == "C3":
        source_id = assign_placebo(dataset, benchmark_id, base_seed)
        text, span_ids = render_evidence(dataset.evidence(source_id))
        return _condition(
            "C3",
            benchmark_id,
            text,
            evidence_span_ids=span_ids,
            source_benchmark_id=source_id,
            seed=placebo_seed(base_seed, benchmark_id),
        )

    # C2 and C4 concatenate documents
    chosen = [d for d in documents if d.role == "supporting"] if condition_id == "C2" else list(documents)
    if not chosen:
        raise ValueError(f"{benchmark_id} has no documents for {condition_id}")
    seed = derive_seed(base_seed, benchmark_id, condition_id, "document-order")
    text, document_ids = render_documents(chosen, seed)
    return _condition(condition_id, benchmark_id, text, document_ids=document_ids, seed=seed)
