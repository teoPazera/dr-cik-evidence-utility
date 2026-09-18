"""Append-only forecast store (plan_a.md 3.1, 3.2 rules 2 and 4).

Records raw sample trajectories, never only scores, so a change to scaling,
winsorisation or the CRPS estimator never requires new (paid) forecaster
calls. One JSONL file per store; each line is one immutable cell record,
keyed by (benchmark_id, condition_id, forecaster_name, repeat).

Not committed for the U0 baseline store (plan_a.md 5.2.9): it is free and
deterministic to regenerate, so only its derived scores and report are.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utrack.forecasters.base import ForecasterOutput


def build_cell_record(
    *,
    benchmark_id: str,
    condition_id: str,
    repeat: int,
    condition_seed: int,
    output: ForecasterOutput,
    dataset_revision: str,
    code_commit: str,
    config_hash: str,
) -> dict:
    """plan_a.md 3.2 rule 4: every field a cell must carry to be traceable."""
    return {
        "benchmark_id": benchmark_id,
        "condition_id": condition_id,
        "repeat": repeat,
        "condition_seed": condition_seed,
        "forecaster_name": output.forecaster_name,
        "forecaster_version": output.forecaster_version,
        "model_identifier": output.model_identifier,
        "seed": output.seed,
        "sampling_params": output.sampling_params,
        "token_counts": output.token_counts,
        "cost_usd": output.cost_usd,
        "n_requested": output.n_requested,
        "n_valid": output.n_valid,
        "generated_at": output.generated_at,
        "dataset_revision": dataset_revision,
        "code_commit": code_commit,
        "config_hash": config_hash,
        "notes": output.notes,
        "samples": output.samples.tolist(),
    }


class ForecastStore:
    """Append-only JSONL store at `path`."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def samples_as_array(record: dict) -> np.ndarray:
        return np.asarray(record["samples"], dtype=float)
