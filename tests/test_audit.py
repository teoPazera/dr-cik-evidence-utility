from utrack.data.audit import (
    DISTRACTOR_SUBTYPES,
    approx_token_count,
    compute_index,
    rank_reveals_role,
    validate_invariants,
)
from utrack.data.loader import Dataset
from utrack.data.schema import Document, EvidenceSpan, Task


def _make_task(**overrides) -> Task:
    defaults = dict(
        benchmark_id="task_x",
        split="open",
        origin="synthetic",
        labels_public=True,
        reasoning_hops=1,
        entity_name="e",
        entity_type="t",
        profile_id="1",
        profile_name="p",
        profile_details={},
        time_series_variable="v",
        frequency="1 day",
        prediction_length=2,
        seasonal_period="1D",
        target_description="d",
        history_timestamps=["2026-01-01 00:00:00", "2026-01-02 00:00:00"],
        history_values=[1.0, 2.0],
        future_timestamps=["2026-01-03 00:00:00", "2026-01-04 00:00:00"],
        future_values=[3.0, 4.0],
        document_ids=["doc_1"],
        gt_evidence=[EvidenceSpan(id="E1", evidence="note")],
        raw_task_path="tasks/task_x.json",
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_full_distractor_set(benchmark_id: str, start_rank: int) -> list[Document]:
    docs = []
    rank = start_rank
    for subtype in DISTRACTOR_SUBTYPES:
        for i in range(5):
            docs.append(
                Document(
                    document_id=f"doc_{subtype}_{i}_{benchmark_id}",
                    benchmark_id=benchmark_id,
                    rank=rank,
                    role="distractor",
                    subtype=subtype,
                    text="distractor text " * 5,
                    raw_document_path="p",
                )
            )
            rank += 1
    return docs


def _valid_dataset() -> Dataset:
    task = _make_task()
    supporting = Document(
        document_id="doc_1",
        benchmark_id="task_x",
        rank=0,
        role="supporting",
        subtype=None,
        text="supporting text " * 5,
        raw_document_path="p",
    )
    distractors = _make_full_distractor_set("task_x", start_rank=1)
    task = _make_task(document_ids=["doc_1"] + [d.document_id for d in distractors])
    return Dataset(tasks={"task_x": task}, documents_by_task={"task_x": [supporting] + distractors})


def test_validate_invariants_clean_dataset_has_no_violations() -> None:
    dataset = _valid_dataset()
    violations = validate_invariants(dataset)
    assert all(len(v) == 0 for v in violations.values()), violations


def test_validate_invariants_catches_length_mismatch() -> None:
    dataset = _valid_dataset()
    bad_task = _make_task(history_values=[1.0])  # history_timestamps has 2 entries
    dataset.tasks["task_x"] = bad_task
    violations = validate_invariants(dataset)
    assert "task_x" in violations["history_values_length_mismatch"]


def test_validate_invariants_catches_non_strictly_increasing_history() -> None:
    dataset = _valid_dataset()
    bad_task = _make_task(
        history_timestamps=["2026-01-02 00:00:00", "2026-01-01 00:00:00"],
        history_values=[1.0, 2.0],
    )
    dataset.tasks["task_x"] = bad_task
    violations = validate_invariants(dataset)
    assert "task_x" in violations["history_timestamps_not_strictly_increasing"]


def test_validate_invariants_catches_history_future_overlap() -> None:
    dataset = _valid_dataset()
    bad_task = _make_task(
        history_timestamps=["2026-01-01 00:00:00", "2026-01-03 00:00:00"],
        future_timestamps=["2026-01-03 00:00:00", "2026-01-04 00:00:00"],
    )
    dataset.tasks["task_x"] = bad_task
    violations = validate_invariants(dataset)
    assert len(violations["history_future_not_strictly_before"]) == 1


def test_validate_invariants_catches_non_finite_history() -> None:
    dataset = _valid_dataset()
    bad_task = _make_task(history_values=[1.0, float("nan")])
    dataset.tasks["task_x"] = bad_task
    violations = validate_invariants(dataset)
    assert len(violations["history_values_non_finite"]) == 1


def test_validate_invariants_catches_dev_task_missing_labels() -> None:
    dataset = _valid_dataset()
    bad_task = _make_task(future_values=[], gt_evidence=[])
    dataset.tasks["task_x"] = bad_task
    violations = validate_invariants(dataset)
    assert "task_x" in violations["dev_task_missing_future_values"]
    assert "task_x" in violations["dev_task_missing_gt_evidence"]


def test_validate_invariants_catches_hidden_task_with_leaked_labels() -> None:
    dataset = _valid_dataset()
    bad_task = _make_task(labels_public=False, origin="human")
    dataset.tasks["task_x"] = bad_task  # still has future_values/gt_evidence from defaults
    violations = validate_invariants(dataset)
    assert "task_x" in violations["hidden_task_has_future_values"]
    assert "task_x" in violations["hidden_task_has_gt_evidence"]


def test_validate_invariants_catches_missing_document_reference() -> None:
    dataset = _valid_dataset()
    task = dataset.tasks["task_x"]
    dataset.tasks["task_x"] = _make_task(
        document_ids=task.document_ids + ["doc_does_not_exist"],
    )
    violations = validate_invariants(dataset)
    assert len(violations["task_references_missing_document"]) == 1
    assert len(violations["document_ids_mismatch_vs_task_documents"]) == 1


def test_validate_invariants_catches_wrong_distractor_subtype_count() -> None:
    dataset = _valid_dataset()
    docs = dataset.documents_by_task["task_x"]
    dataset.documents_by_task["task_x"] = docs[:-1]  # drop one 'temporal' distractor
    violations = validate_invariants(dataset)
    assert len(violations["not_exactly_five_per_distractor_subtype"]) == 1


def test_rank_reveals_role_detects_clean_ordering() -> None:
    dataset = _valid_dataset()
    finding = rank_reveals_role(dataset)
    assert finding["role_fully_revealed_by_rank"] is True
    assert finding["tasks_with_supporting_before_distractor"] == 1


def test_rank_reveals_role_detects_shuffled_ordering() -> None:
    dataset = _valid_dataset()
    docs = dataset.documents_by_task["task_x"]
    shuffled = [docs[1], docs[0]] + docs[2:]  # swap first supporting/distractor
    dataset.documents_by_task["task_x"] = shuffled
    finding = rank_reveals_role(dataset)
    assert finding["role_fully_revealed_by_rank"] is False


def test_compute_index_row_values() -> None:
    dataset = _valid_dataset()
    df = compute_index(dataset)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["benchmark_id"] == "task_x"
    assert row["document_count"] == 26
    assert row["supporting_count"] == 1
    assert row["distractor_count"] == 25
    assert row["distractor_confounder_count"] == 5
    assert row["evidence_span_count"] == 1
    assert row["future_min"] == 3.0
    assert row["future_max"] == 4.0
    assert row["near_constant_future"] == False  # noqa: E712 (pandas bool)


def test_compute_index_flags_near_constant_future() -> None:
    dataset = _valid_dataset()
    dataset.tasks["task_x"] = _make_task(
        document_ids=dataset.tasks["task_x"].document_ids,
        future_values=[100.0, 100.0000001],
    )
    df = compute_index(dataset)
    assert bool(df.iloc[0]["near_constant_future"]) is True


def test_compute_index_hidden_task_has_null_future_stats() -> None:
    task = _make_task(labels_public=False, origin="human", future_values=[], gt_evidence=[])
    supporting = Document(
        document_id="doc_1", benchmark_id="task_x", rank=0, role="supporting", subtype=None,
        text="s", raw_document_path="p",
    )
    dataset = Dataset(tasks={"task_x": task}, documents_by_task={"task_x": [supporting]})
    df = compute_index(dataset)
    assert df.iloc[0]["future_min"] is None
    assert df.iloc[0]["near_constant_future"] is None


def test_approx_token_count_scales_with_length() -> None:
    assert approx_token_count("") == 0
    assert approx_token_count("abcd") == 1
    assert approx_token_count("a" * 400) == 100
