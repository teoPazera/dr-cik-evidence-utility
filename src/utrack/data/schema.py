"""Typed records for the Dr-CiK snapshot, as the schema was actually found in U0.2.

plan_a.md section 1.2 describes a prior schema (nested `task_metadata`, a
`showcase` object, documents embedded in the task record, `annotations.gt_evidence`).
The released Hugging Face configs do not match that prior: fields are flat on
the task record, and document content/role/subtype live in two separate
configs joined by `document_id`. This module reflects what U0.2 found, not
what section 1.2 guessed; see artifacts/u0/data_audit.md for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSpan:
    id: str
    evidence: str


@dataclass(frozen=True)
class Document:
    """One document as attached to one task (`task_documents` joined with `documents`).

    Every document in the release belongs to exactly one task (verified in
    U0.2: `documents.task_ids` always has length 1), so `benchmark_id` is a
    plain field, not a list.
    """

    document_id: str
    benchmark_id: str
    rank: int
    role: str  # "supporting" | "distractor"
    subtype: str | None  # None for supporting; one of 5 subtypes for distractor
    text: str
    raw_document_path: str


@dataclass(frozen=True)
class Task:
    """One row of `data/tasks/train.jsonl`, field-for-field."""

    benchmark_id: str
    split: str
    origin: str  # "human" | "synthetic"
    labels_public: bool
    reasoning_hops: int
    entity_name: str
    entity_type: str
    profile_id: str
    profile_name: str
    profile_details: dict
    time_series_variable: str
    frequency: str
    prediction_length: int
    seasonal_period: str | int | None  # inconsistently typed in the release; see data_audit.md
    target_description: str
    history_timestamps: list[str]
    history_values: list[float | None]
    future_timestamps: list[str]
    future_values: list[float]  # empty when labels_public is False
    document_ids: list[str]
    gt_evidence: list[EvidenceSpan]  # empty when labels_public is False
    raw_task_path: str

    @classmethod
    def from_raw(cls, record: dict) -> "Task":
        return cls(
            benchmark_id=record["benchmark_id"],
            split=record["split"],
            origin=record["origin"],
            labels_public=record["labels_public"],
            reasoning_hops=record["reasoning_hops"],
            entity_name=record["entity_name"],
            entity_type=record["entity_type"],
            profile_id=record["profile_id"],
            profile_name=record["profile_name"],
            profile_details=record["profile_details"],
            time_series_variable=record["time_series_variable"],
            frequency=record["frequency"],
            prediction_length=record["prediction_length"],
            seasonal_period=record.get("seasonal_period"),
            target_description=record["target_description"],
            history_timestamps=record["history_timestamps"],
            history_values=record["history_values"],
            future_timestamps=record["future_timestamps"],
            future_values=record["future_values"],
            document_ids=record["document_ids"],
            gt_evidence=[EvidenceSpan(**e) for e in record["gt_evidence"]],
            raw_task_path=record["raw_task_path"],
        )


@dataclass(frozen=True)
class ForecastInput:
    """What a forecaster (or condition builder) is allowed to see. No labels field exists
    on this type by construction (plan_a.md 3.2 rule 3, label isolation)."""

    benchmark_id: str
    origin: str
    frequency: str
    prediction_length: int
    seasonal_period: str | None
    history_timestamps: list[str]
    history_values: list[float | None]
    future_timestamps: list[str]
    entity_name: str
    entity_type: str
    profile_id: str
    profile_name: str
    profile_details: dict
    time_series_variable: str
    target_description: str
    reasoning_hops: int

    # No `document_ids` (removed in U0.5): the ids are sequential in stored rank order, which
    # U0.2 showed reveals role, and documents reach a forecaster only through a Condition.

    @classmethod
    def from_task(cls, task: Task) -> "ForecastInput":
        return cls(
            benchmark_id=task.benchmark_id,
            origin=task.origin,
            frequency=task.frequency,
            prediction_length=task.prediction_length,
            seasonal_period=task.seasonal_period,
            history_timestamps=task.history_timestamps,
            history_values=task.history_values,
            future_timestamps=task.future_timestamps,
            entity_name=task.entity_name,
            entity_type=task.entity_type,
            profile_id=task.profile_id,
            profile_name=task.profile_name,
            profile_details=task.profile_details,
            time_series_variable=task.time_series_variable,
            target_description=task.target_description,
            reasoning_hops=task.reasoning_hops,
        )


@dataclass(frozen=True)
class TaskLabels:
    """Withheld from the code path that builds prompts (plan_a.md 3.2 rule 3)."""

    benchmark_id: str
    future_values: list[float]
    gt_evidence: list[EvidenceSpan]

    @classmethod
    def from_task(cls, task: Task) -> "TaskLabels":
        return cls(
            benchmark_id=task.benchmark_id,
            future_values=task.future_values,
            gt_evidence=task.gt_evidence,
        )
