"""The `Condition` record (plan_a.md 3.1, 3.3): what context a forecaster is given, and how it was made.

A condition is produced without any model call, deterministically from (task, dataset, seed).
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump when the rendering rules (Decision G) change, so a stored cell can be traced to the
# rendering that produced its context. Previews record it in their header.
RENDER_VERSION = "g-default-v1"

CONDITION_NAMES: dict[str, str] = {
    "C0": "no-context",
    "C1": "gt-evidence",
    "C2": "supporting-concat",
    "C3": "placebo",
    "C4": "all-docs-concat",
}


@dataclass(frozen=True)
class Condition:
    condition_id: str
    name: str
    benchmark_id: str
    context: str | None  # None for C0
    document_ids: tuple[str, ...]  # documents used, in the order they were rendered
    evidence_span_ids: tuple[str, ...]  # evidence spans used, in the order they were rendered
    source_benchmark_id: str | None  # C3 only: the task the placebo evidence was taken from
    approx_tokens: int  # characters / 4, see data.audit.approx_token_count
    seed: int | None  # None when nothing random was involved (C0, C1)
    render_version: str = RENDER_VERSION
