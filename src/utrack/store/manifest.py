"""Run manifest: provenance for one batch of forecast-store writes (plan_a.md 3.2 rule 4)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def config_hash(config: dict) -> str:
    """Stable hash of a config dict (plan_a.md 5.2.7: dictionary keys sorted before hashing)."""
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


@dataclass
class RunManifest:
    run_id: str
    machine: str
    dataset_revision: str
    code_commit: str
    config_hash: str
    started_at: str
    finished_at: str | None = None
    n_cells: int = 0
    forecaster_names: list[str] = field(default_factory=list)
    condition_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "machine": self.machine,
            "dataset_revision": self.dataset_revision,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_cells": self.n_cells,
            "forecaster_names": self.forecaster_names,
            "condition_ids": self.condition_ids,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
