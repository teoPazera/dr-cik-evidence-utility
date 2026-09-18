"""Placebo assignment (Decision H1, default): evidence from a different task.

The source is drawn from the dev tasks (only they have evidence) whose entity and variable both
differ from the target's, so no task is its own placebo and the placebo is not about the same
entity or series. Each target gets an independent seeded draw; the pool is always the full dev
set, so a task's placebo does not depend on which tasks a stage happens to run.
"""

from __future__ import annotations

import numpy as np

from utrack.data.loader import Dataset
from utrack.seeds import derive_seed


def placebo_seed(base_seed: int, benchmark_id: str) -> int:
    return derive_seed(base_seed, benchmark_id, "placebo-source")


def placebo_candidates(dataset: Dataset, benchmark_id: str) -> list[str]:
    target = dataset.tasks[benchmark_id]
    return [
        source_id
        for source_id in dataset.dev_task_ids()
        if dataset.tasks[source_id].gt_evidence
        and dataset.tasks[source_id].entity_name != target.entity_name
        and dataset.tasks[source_id].time_series_variable != target.time_series_variable
    ]


def assign_placebo(dataset: Dataset, benchmark_id: str, base_seed: int) -> str:
    candidates = placebo_candidates(dataset, benchmark_id)
    if not candidates:
        raise ValueError(f"no placebo source with a different entity and variable for {benchmark_id}")
    rng = np.random.default_rng(placebo_seed(base_seed, benchmark_id))
    return candidates[int(rng.integers(len(candidates)))]
