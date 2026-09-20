"""Placebo assignment (Decision H1 + H2): evidence from a different task.

The source is drawn from the dev tasks (only they have evidence) whose entity and variable both
differ from the target's, so no task is its own placebo and the placebo is not about the same
entity or series. Each target gets an independent seeded draw; the pool is always the full dev
set, so a task's placebo does not depend on which tasks a stage happens to run.
"""

from __future__ import annotations

import numpy as np

from utrack.data.loader import Dataset
from utrack.seeds import derive_seed


def length_match_text(text: str, target_characters: int) -> str:
    """Deterministically truncate or repeat whole evidence lines to match target length.

    The target is the rendered C1 character count. Truncation happens at a line boundary
    when possible; a source shorter than target is repeated line-by-line so H2 controls
    prompt length without inventing semantic filler.
    """
    if target_characters <= 0:
        return ""
    lines = [line for line in text.splitlines() if line]
    if not lines:
        raise ValueError("placebo source has empty rendered evidence")
    chunks: list[str] = []
    index = 0
    while len("\n".join(chunks)) < target_characters:
        chunks.append(lines[index % len(lines)])
        index += 1
    rendered = "\n".join(chunks)
    if len(rendered) <= target_characters:
        return rendered
    prefix = rendered[:target_characters]
    boundary = prefix.rfind("\n")
    return prefix if boundary <= 0 else prefix[:boundary]


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
