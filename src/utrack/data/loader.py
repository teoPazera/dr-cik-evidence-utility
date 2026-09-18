"""Load the raw Dr-CiK snapshot into typed `Task`/`Document` objects (plan_a.md U0.2).

Documents are kept in stored `rank` order here; U0.2 found that order to fully
reveal role and distractor subtype (see artifacts/u0/data_audit.md), so any
code that renders documents to a prompt must shuffle first (plan_a.md 1.2,
5.2.7) rather than relying on the loader to do it.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utrack.data.schema import Document, ForecastInput, Task, TaskLabels

logger = logging.getLogger(__name__)


def fill_history_forward(history_values: list[float | None], benchmark_id: str = "") -> np.ndarray:
    """Forward/back-fill NaN or None history values (U0.2's audit found 13 tasks, none in
    the dev set, with NaN history; reported there, not fixed - handled here instead since
    neither a forecaster nor a scaling denominator can compute on NaN). Logs when it
    actually does something."""
    arr = pd.Series(history_values, dtype="float64")
    n_missing = int(arr.isna().sum())
    if n_missing:
        logger.warning(
            "forward/back-filling %d/%d missing history values for %s",
            n_missing,
            len(arr),
            benchmark_id or "<unknown task>",
        )
        arr = arr.ffill().bfill()
    return arr.to_numpy()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class Dataset:
    tasks: dict[str, Task]  # keyed by benchmark_id
    documents_by_task: dict[str, list[Document]]  # sorted by stored rank

    def forecast_input(self, benchmark_id: str) -> ForecastInput:
        return ForecastInput.from_task(self.tasks[benchmark_id])

    def labels(self, benchmark_id: str) -> TaskLabels:
        return TaskLabels.from_task(self.tasks[benchmark_id])


def load_dataset(repo_root: Path, dataset_cfg: dict) -> Dataset:
    configs = dataset_cfg["configs"]
    raw_tasks = load_jsonl(repo_root / configs["tasks"])
    raw_documents = load_jsonl(repo_root / configs["documents"])
    raw_task_documents = load_jsonl(repo_root / configs["task_documents"])

    text_by_document_id = {d["document_id"]: d["text"] for d in raw_documents}

    documents_by_task: dict[str, list[Document]] = defaultdict(list)
    for row in raw_task_documents:
        doc = Document(
            document_id=row["document_id"],
            benchmark_id=row["benchmark_id"],
            rank=row["rank"],
            role=row["role"],
            subtype=row["subtype"],
            text=text_by_document_id[row["document_id"]],
            raw_document_path=row["raw_document_path"],
        )
        documents_by_task[doc.benchmark_id].append(doc)
    for docs in documents_by_task.values():
        docs.sort(key=lambda d: d.rank)

    tasks = {r["benchmark_id"]: Task.from_raw(r) for r in raw_tasks}

    return Dataset(tasks=tasks, documents_by_task=dict(documents_by_task))
