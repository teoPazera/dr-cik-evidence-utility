import json
from pathlib import Path

from utrack.data.loader import fill_history_forward, load_dataset
from utrack.data.schema import ForecastInput, TaskLabels


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_fixture(repo_root: Path) -> dict:
    dataset_cfg = {
        "configs": {
            "tasks": "data/tasks/train.jsonl",
            "documents": "data/documents/train.jsonl",
            "task_documents": "data/task_documents/train.jsonl",
        }
    }

    tasks = [
        {
            "benchmark_id": "task_dev",
            "split": "open",
            "origin": "synthetic",
            "labels_public": True,
            "reasoning_hops": 2,
            "entity_name": "Widget Co",
            "entity_type": "store",
            "profile_id": "1",
            "profile_name": "Widget Co",
            "profile_details": {"a": 1},
            "time_series_variable": "sales",
            "frequency": "1 day",
            "prediction_length": 2,
            "seasonal_period": "1D",
            "target_description": "daily sales",
            "history_timestamps": ["2026-01-01 00:00:00", "2026-01-02 00:00:00"],
            "history_values": [1.0, 2.0],
            "future_timestamps": ["2026-01-03 00:00:00", "2026-01-04 00:00:00"],
            "future_values": [3.0, 4.0],
            "document_ids": ["doc_1", "doc_2"],
            "gt_evidence": [{"id": "E1", "evidence": "a note"}],
            "raw_task_path": "tasks/task_dev.json",
        },
        {
            "benchmark_id": "task_hidden",
            "split": "open",
            "origin": "human",
            "labels_public": False,
            "reasoning_hops": 2,
            "entity_name": "Gadget Co",
            "entity_type": "store",
            "profile_id": "2",
            "profile_name": "Gadget Co",
            "profile_details": {},
            "time_series_variable": "sales",
            "frequency": "1 day",
            "prediction_length": 1,
            "seasonal_period": None,
            "target_description": "daily sales",
            "history_timestamps": ["2026-01-01 00:00:00"],
            "history_values": [5.0],
            "future_timestamps": ["2026-01-02 00:00:00"],
            "future_values": [],
            "document_ids": ["doc_3"],
            "gt_evidence": [],
            "raw_task_path": "tasks/task_hidden.json",
        },
    ]

    documents = [
        {"document_id": "doc_1", "raw_document_path": "docs/doc_1.md", "task_ids": ["task_dev"], "roles": ["supporting"], "subtypes": [None], "text": "supporting text"},
        {"document_id": "doc_2", "raw_document_path": "docs/doc_2.md", "task_ids": ["task_dev"], "roles": ["distractor"], "subtypes": ["confounder"], "text": "distractor text"},
        {"document_id": "doc_3", "raw_document_path": "docs/doc_3.md", "task_ids": ["task_hidden"], "roles": ["supporting"], "subtypes": [None], "text": "hidden supporting text"},
    ]

    task_documents = [
        {"benchmark_id": "task_dev", "document_id": "doc_1", "rank": 0, "role": "supporting", "subtype": None, "raw_document_path": "docs/doc_1.md"},
        {"benchmark_id": "task_dev", "document_id": "doc_2", "rank": 1, "role": "distractor", "subtype": "confounder", "raw_document_path": "docs/doc_2.md"},
        {"benchmark_id": "task_hidden", "document_id": "doc_3", "rank": 0, "role": "supporting", "subtype": None, "raw_document_path": "docs/doc_3.md"},
    ]

    _write_jsonl(repo_root / dataset_cfg["configs"]["tasks"], tasks)
    _write_jsonl(repo_root / dataset_cfg["configs"]["documents"], documents)
    _write_jsonl(repo_root / dataset_cfg["configs"]["task_documents"], task_documents)
    return dataset_cfg


def test_load_dataset_builds_tasks_and_documents(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    dataset = load_dataset(tmp_path, cfg)

    assert set(dataset.tasks.keys()) == {"task_dev", "task_hidden"}
    assert len(dataset.documents_by_task["task_dev"]) == 2
    assert [d.document_id for d in dataset.documents_by_task["task_dev"]] == ["doc_1", "doc_2"]
    assert dataset.documents_by_task["task_dev"][0].role == "supporting"
    assert dataset.documents_by_task["task_dev"][1].role == "distractor"
    assert dataset.documents_by_task["task_dev"][1].subtype == "confounder"
    assert dataset.documents_by_task["task_dev"][0].text == "supporting text"


def test_forecast_input_has_no_label_fields(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    dataset = load_dataset(tmp_path, cfg)

    fi = dataset.forecast_input("task_dev")
    assert isinstance(fi, ForecastInput)
    field_names = {f for f in vars(fi)}
    assert "future_values" not in field_names
    assert "gt_evidence" not in field_names
    assert fi.future_timestamps == ["2026-01-03 00:00:00", "2026-01-04 00:00:00"]


def test_task_labels_carries_labels(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    dataset = load_dataset(tmp_path, cfg)

    labels = dataset.labels("task_dev")
    assert isinstance(labels, TaskLabels)
    assert labels.future_values == [3.0, 4.0]
    assert labels.gt_evidence[0].id == "E1"

    hidden_labels = dataset.labels("task_hidden")
    assert hidden_labels.future_values == []
    assert hidden_labels.gt_evidence == []


def test_fill_history_forward_fills_nan_and_leading_nan() -> None:
    filled = fill_history_forward([None, 1.0, float("nan"), 3.0])
    assert list(filled) == [1.0, 1.0, 1.0, 3.0]


def test_fill_history_forward_no_missing_is_unchanged() -> None:
    filled = fill_history_forward([1.0, 2.0, 3.0])
    assert list(filled) == [1.0, 2.0, 3.0]
