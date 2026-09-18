"""U1 task selection (Decision F, default rule; the decision itself is still Teo's).

The three repository sample tasks, plus `n_extra` dev tasks picked one at a time by a seeded
draw among dev tasks whose frequency AND horizon are both not yet covered by the tasks chosen so
far. Candidates are sorted by natural id before the draw, so the result never depends on the
order the dataset happens to be loaded in.
"""

from __future__ import annotations

import numpy as np

from utrack.conditions.placebo import assign_placebo
from utrack.data.audit import approx_token_count
from utrack.data.loader import Dataset
from utrack.seeds import derive_seed

RULE_TEXT = (
    "Fixed: the repository sample tasks. Then, n_extra times: among dev tasks not yet selected whose "
    "frequency and prediction_length are both absent from the tasks selected so far, draw one with "
    "numpy default_rng(sha256-derived seed of (seed, 'u1-task-selection', step)) over the natural-id-"
    "sorted candidates."
)


def selection_record(
    dataset: Dataset,
    selected: list[dict],
    *,
    fixed: list[str],
    n_extra: int,
    seed: int,
    base_seed: int,
    condition_ids: list[str],
    dataset_revision: str,
) -> dict:
    """The content of artifacts/u0/u1_tasks.json: the rule, the seed and one row per chosen task."""
    tasks = []
    for row in selected:
        task = dataset.tasks[row["benchmark_id"]]
        documents = dataset.documents_by_task[row["benchmark_id"]]
        tasks.append(
            {
                **row,
                "origin": task.origin,
                "frequency": task.frequency,
                "prediction_length": task.prediction_length,
                "history_length": len(task.history_values),
                "evidence_spans": len(task.gt_evidence),
                "supporting_documents": sum(d.role == "supporting" for d in documents),
                "approx_tokens_all_documents": sum(approx_token_count(d.text) for d in documents),
                "placebo_source": assign_placebo(dataset, row["benchmark_id"], base_seed),
            }
        )
    return {
        "decision": "F",
        "status": "default rule applied; Decision F is not yet resolved by Teo (configs/decisions.md)",
        "rule": RULE_TEXT,
        "dataset_revision": dataset_revision,
        "fixed_tasks": fixed,
        "n_extra": n_extra,
        "selection_seed": seed,
        "conditions_seed": base_seed,
        "condition_ids": condition_ids,
        "tasks": tasks,
    }


def select_u1_tasks(dataset: Dataset, fixed: list[str], n_extra: int, seed: int) -> list[dict]:
    dev = set(dataset.dev_task_ids())
    for benchmark_id in fixed:
        if benchmark_id not in dev:
            raise ValueError(f"fixed U1 task {benchmark_id} is not a dev task with public labels")

    selected = [{"benchmark_id": b, "source": "repository_sample", "step": None, "n_eligible": None} for b in fixed]
    for step in range(n_extra):
        chosen_ids = {row["benchmark_id"] for row in selected}
        covered_frequencies = {dataset.tasks[b].frequency for b in chosen_ids}
        covered_horizons = {dataset.tasks[b].prediction_length for b in chosen_ids}
        eligible = [
            b
            for b in dataset.dev_task_ids()
            if b not in chosen_ids
            and dataset.tasks[b].frequency not in covered_frequencies
            and dataset.tasks[b].prediction_length not in covered_horizons
        ]
        if not eligible:
            raise ValueError(
                f"step {step}: no dev task adds both a new frequency and a new horizon "
                f"(covered frequencies {sorted(covered_frequencies)}, horizons {sorted(covered_horizons)})"
            )
        rng = np.random.default_rng(derive_seed(seed, "u1-task-selection", str(step)))
        pick = eligible[int(rng.integers(len(eligible)))]
        selected.append({"benchmark_id": pick, "source": "seeded_pick", "step": step, "n_eligible": len(eligible)})
    return selected
