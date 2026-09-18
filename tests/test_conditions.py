import dataclasses
import hashlib
import importlib.util
import json

import numpy as np
import pytest

from utrack.conditions import build_condition
from utrack.conditions import builders as builders_mod
from utrack.conditions import leakage as leakage_mod
from utrack.conditions import placebo as placebo_mod
from utrack.conditions import preview as preview_mod
from utrack.conditions import render as render_mod
from utrack.conditions import selection as selection_mod
from utrack.data.loader import Dataset, natural_sort_key
from utrack.data.schema import Document, EvidenceSpan, Task
from utrack.reports import leakage as leakage_report_mod

FREQUENCIES = ["1 hour", "1 day", "5 minutes", "1 minute", "1 second"]


def _task(
    n: int,
    *,
    labels_public: bool = True,
    frequency: str = "1 hour",
    horizon: int = 8,
    entity: str | None = None,
    variable: str | None = None,
    evidence: list[str] | None = None,
) -> Task:
    rng = np.random.default_rng(n)
    history = [round(float(v), 3) for v in 50 + 10 * rng.normal(size=30)]
    future = [round(float(v), 3) for v in 50 + 10 * rng.normal(size=horizon)]
    if evidence is None:
        evidence = [f"Evidence {n} first.", f"Evidence {n} second."]
    spans = [EvidenceSpan(id=f"E{i + 1}", evidence=t) for i, t in enumerate(evidence)]
    return Task(
        benchmark_id=f"task_{n}",
        split="open",
        origin="synthetic" if labels_public else "human",
        labels_public=labels_public,
        reasoning_hops=1,
        entity_name=entity or f"entity {n}",
        entity_type="t",
        profile_id="1",
        profile_name="p",
        profile_details={},
        time_series_variable=variable or f"variable {n}",
        frequency=frequency,
        prediction_length=horizon,
        seasonal_period=None,
        target_description="d",
        history_timestamps=[f"2026-01-01 {h:02d}:00:00" for h in range(24)] + [f"2026-01-02 {h:02d}:00:00" for h in range(6)],
        history_values=history,
        future_timestamps=[f"2026-01-03 {h:02d}:00:00" for h in range(horizon)],
        future_values=future if labels_public else [],
        document_ids=[],
        gt_evidence=spans if labels_public else [],
        raw_task_path="p",
    )


def _documents(n: int, n_supporting: int = 3, n_distractor: int = 4) -> list[Document]:
    docs = []
    for rank in range(n_supporting + n_distractor):
        supporting = rank < n_supporting
        docs.append(
            Document(
                document_id=f"doc_{n * 100 + rank}",
                benchmark_id=f"task_{n}",
                rank=rank,
                role="supporting" if supporting else "distractor",
                subtype=None if supporting else "noisy",
                text=f"Body of document {n * 100 + rank}.  \n",
                raw_document_path="p",
            )
        )
    return docs


def _dataset(n_tasks: int = 6) -> Dataset:
    tasks = {f"task_{n}": _task(n) for n in range(1, n_tasks + 1)}
    tasks[f"task_{n_tasks + 1}"] = _task(n_tasks + 1, labels_public=False)
    documents = {b: _documents(int(b.split("_")[1])) for b in tasks}
    return Dataset(tasks=tasks, documents_by_task=documents)


# --- loader helpers ---------------------------------------------------------


def test_natural_sort_key_orders_by_trailing_number() -> None:
    assert sorted(["E10", "E2", "E1"], key=natural_sort_key) == ["E1", "E2", "E10"]
    assert sorted(["task_10", "task_9"], key=natural_sort_key) == ["task_9", "task_10"]
    assert sorted(["b", "a"], key=natural_sort_key) == ["a", "b"]


def test_dataset_evidence_and_dev_ids() -> None:
    ds = _dataset()
    assert [s.id for s in ds.evidence("task_1")] == ["E1", "E2"]
    assert ds.evidence("task_7") == ()  # the hidden task has no evidence
    assert ds.dev_task_ids() == [f"task_{n}" for n in range(1, 7)]


# --- rendering (Decision G) -------------------------------------------------


def test_render_evidence_natural_order_one_span_per_line() -> None:
    spans = [
        EvidenceSpan("E10", "tenth"),
        EvidenceSpan("E2", "second\nwith a line break"),
        EvidenceSpan("E1", "first"),
    ]
    text, ids = render_mod.render_evidence(spans)
    assert ids == ("E1", "E2", "E10")
    assert text.split("\n") == ["first", "second with a line break", "tenth"]


def test_render_documents_ignores_input_order_and_is_seeded() -> None:
    docs = _documents(1)
    a, ids_a = render_mod.render_documents(docs, seed=5)
    b, ids_b = render_mod.render_documents(list(reversed(docs)), seed=5)
    assert (a, ids_a) == (b, ids_b)  # stored (rank) order cannot influence the output
    c, _ = render_mod.render_documents(docs, seed=6)
    assert c != a
    assert sorted(ids_a) == sorted(d.document_id for d in docs)


def test_render_documents_delimiters_carry_only_a_neutral_index() -> None:
    docs = _documents(1)
    text, _ = render_mod.render_documents(docs, seed=1)
    delimiters = [line for line in text.split("\n") if line.startswith("[Document ")]
    assert delimiters == [f"[Document {i}]" for i in range(1, len(docs) + 1)]
    assert "supporting" not in text and "distractor" not in text and "noisy" not in text
    assert "  \n" not in text  # trailing whitespace of each document is stripped


def test_shuffled_order_carries_no_role_signal() -> None:
    supporting_positions = []
    for seed in range(200):
        docs = _documents(1, n_supporting=3, n_distractor=7)
        _, ids = render_mod.render_documents(docs, seed=seed)
        role = {d.document_id: d.role for d in docs}
        supporting_positions.extend(i / 9 for i, d in enumerate(ids) if role[d] == "supporting")
    assert abs(np.mean(supporting_positions) - 0.5) < 0.03  # stored order would give 0.15


# --- builders ---------------------------------------------------------------


def test_c0_has_no_context() -> None:
    c = build_condition("C0", _dataset(), "task_1", 1)
    assert c.context is None and c.approx_tokens == 0 and c.seed is None


def test_c1_is_the_evidence() -> None:
    c = build_condition("C1", _dataset(), "task_1", 1)
    assert c.context == "Evidence 1 first.\nEvidence 1 second."
    assert c.evidence_span_ids == ("E1", "E2") and c.approx_tokens > 0


def test_c1_refuses_a_hidden_task() -> None:
    with pytest.raises(ValueError, match="no gt_evidence"):
        build_condition("C1", _dataset(), "task_7", 1)


def test_c2_uses_only_supporting_documents_and_c4_uses_all() -> None:
    ds = _dataset()
    c2 = build_condition("C2", ds, "task_1", 1)
    c4 = build_condition("C4", ds, "task_1", 1)
    supporting = {d.document_id for d in ds.documents_by_task["task_1"] if d.role == "supporting"}
    everything = {d.document_id for d in ds.documents_by_task["task_1"]}
    assert set(c2.document_ids) == supporting
    assert set(c4.document_ids) == everything
    assert c2.seed is not None and c2.seed != c4.seed


def test_builders_are_deterministic_and_seed_sensitive() -> None:
    ds = _dataset()
    for cid in ("C0", "C1", "C2", "C3", "C4"):
        assert build_condition(cid, ds, "task_1", 1) == build_condition(cid, ds, "task_1", 1)
    assert build_condition("C2", ds, "task_1", 1).context != build_condition("C2", ds, "task_1", 2).context


def test_unknown_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        build_condition("C9", _dataset(), "task_1", 1)


# --- placebo (Decision H1) --------------------------------------------------


def test_placebo_is_another_tasks_evidence_with_different_entity_and_variable() -> None:
    ds = _dataset()
    ds.tasks["task_2"] = _task(2, entity=ds.tasks["task_1"].entity_name)  # same entity as task_1
    ds.tasks["task_3"] = _task(3, variable=ds.tasks["task_1"].time_series_variable)  # same variable
    for base_seed in range(20):
        c = build_condition("C3", ds, "task_1", base_seed)
        assert c.source_benchmark_id not in ("task_1", "task_2", "task_3", "task_7")
        source = ds.tasks[c.source_benchmark_id]
        assert c.context == "\n".join(e.evidence for e in source.gt_evidence)


def test_placebo_assignment_is_deterministic_and_varies_with_seed() -> None:
    ds = _dataset(30)
    picks = {placebo_mod.assign_placebo(ds, "task_1", s) for s in range(30)}
    assert len(picks) > 1
    assert placebo_mod.assign_placebo(ds, "task_1", 3) == placebo_mod.assign_placebo(ds, "task_1", 3)


def test_placebo_fails_loudly_without_candidates() -> None:
    ds = Dataset(
        tasks={"task_1": _task(1, entity="same", variable="v1"), "task_2": _task(2, entity="same", variable="v2")},
        documents_by_task={},
    )
    with pytest.raises(ValueError, match="no placebo source"):
        placebo_mod.assign_placebo(ds, "task_1", 1)


# --- U1 task selection (Decision F) -----------------------------------------


def _selection_dataset() -> Dataset:
    tasks = {}
    spec = [  # (frequency, horizon)
        ("1 day", 100), ("1 hour", 24), ("5 minutes", 90),  # the fixed three
        ("1 minute", 120), ("1 minute", 24), ("1 second", 50), ("1 second", 100), ("1 hour", 50),
    ]
    for n, (frequency, horizon) in enumerate(spec, start=1):
        tasks[f"task_{n}"] = _task(n, frequency=frequency, horizon=horizon)
    return Dataset(tasks=tasks, documents_by_task={b: _documents(int(b.split("_")[1])) for b in tasks})


def test_selection_adds_new_frequency_and_new_horizon() -> None:
    ds = _selection_dataset()
    fixed = ["task_1", "task_2", "task_3"]
    chosen = selection_mod.select_u1_tasks(ds, fixed, n_extra=2, seed=7)
    assert [c["benchmark_id"] for c in chosen[:3]] == fixed
    extra = [c["benchmark_id"] for c in chosen[3:]]
    assert len(extra) == 2
    frequencies = [ds.tasks[b].frequency for b in fixed + extra]
    horizons = [ds.tasks[b].prediction_length for b in fixed + extra]
    assert len(set(frequencies)) == 5 and len(set(horizons)) == 5
    assert all(c["source"] == "seeded_pick" for c in chosen[3:])


def test_selection_is_deterministic() -> None:
    ds = _selection_dataset()
    fixed = ["task_1", "task_2", "task_3"]
    assert selection_mod.select_u1_tasks(ds, fixed, 2, 7) == selection_mod.select_u1_tasks(ds, fixed, 2, 7)


def test_selection_rejects_a_hidden_fixed_task_and_an_exhausted_pool() -> None:
    ds = _dataset()
    with pytest.raises(ValueError, match="not a dev task"):
        selection_mod.select_u1_tasks(ds, ["task_7"], 0, 1)
    with pytest.raises(ValueError, match="no dev task adds both"):
        selection_mod.select_u1_tasks(_selection_dataset(), ["task_1", "task_2", "task_3"], 5, 1)


def test_selection_record_lists_rule_seed_and_placebo_source() -> None:
    ds = _selection_dataset()
    fixed = ["task_1", "task_2", "task_3"]
    selected = selection_mod.select_u1_tasks(ds, fixed, 2, 7)
    record = selection_mod.selection_record(
        ds, selected, fixed=fixed, n_extra=2, seed=7, base_seed=11, condition_ids=["C0", "C1"], dataset_revision="r"
    )
    assert record["selection_seed"] == 7 and record["conditions_seed"] == 11
    assert record["rule"] and "not yet resolved" in record["status"]
    assert all(row["placebo_source"] != row["benchmark_id"] for row in record["tasks"])
    json.dumps(record)  # serialisable


# --- previews and their SHA-256 list ----------------------------------------


def _preview_files(ds: Dataset) -> dict[str, bytes]:
    return preview_mod.render_preview_files(ds, ["task_1", "task_2"], ["C0", "C1", "C2", "C3"], 5, "rev")


def test_preview_files_are_deterministic_lf_only_and_hash_listed() -> None:
    ds = _dataset()
    files = _preview_files(ds)
    assert files == _preview_files(ds)
    assert set(files) == {f"task_{t}/C{c}.md" for t in (1, 2) for c in range(4)} | {"sha256.json"}
    assert all(b"\r" not in data for data in files.values())
    listing = json.loads(files["sha256.json"])["files"]
    assert listing == {p: hashlib.sha256(d).hexdigest() for p, d in files.items() if p != "sha256.json"}


def test_preview_header_has_ids_tokens_and_seed() -> None:
    text = _preview_files(_dataset())["task_1/C2.md"].decode("utf-8")
    assert "approximate tokens" in text and "seed:" in text and "documents used, in rendered order: doc_" in text
    assert "dataset revision: `rev`" in text
    placebo = _preview_files(_dataset())["task_1/C3.md"].decode("utf-8")
    assert "placebo evidence taken from: task_" in placebo


def test_verify_previews_detects_missing_changed_and_extra_files(tmp_path) -> None:
    files = _preview_files(_dataset())
    preview_mod.write_previews(files, tmp_path)
    assert preview_mod.verify_previews(files, tmp_path) == []

    (tmp_path / "task_1" / "C1.md").write_bytes(b"tampered")
    (tmp_path / "task_2" / "C0.md").unlink()
    (tmp_path / "task_2" / "extra.md").write_bytes(b"x")
    problems = preview_mod.verify_previews(files, tmp_path)
    assert "differs: task_1/C1.md" in problems
    assert "missing: task_2/C0.md" in problems
    assert "unexpected extra file: task_2/extra.md" in problems


# --- leakage detector -------------------------------------------------------

FUTURE = [312.5, 298.1, 305.7, 301.2, 299.9, 310.4]


def _in_cik_format(values: list[float]) -> str:
    return "\n".join(f"(2026-03-{i + 1:02d} 00:00:00, {v:.6g})" for i, v in enumerate(values))


def test_detector_finds_a_planted_run_in_cik_format() -> None:
    hit = leakage_mod.find_future_run(_in_cik_format(FUTURE), FUTURE)
    assert hit is not None and hit.run_length == len(FUTURE) and hit.future_start == 0


def test_detector_finds_a_run_in_prose_and_reports_where_it_starts() -> None:
    text = "Analysts expect 301.2, then 299.9 and 310.4, before a drop."
    assert leakage_mod.find_future_run(text, FUTURE) is None  # 3 values are below the default run of 4
    hit = leakage_mod.find_future_run(text, FUTURE, min_run=3)
    assert (hit.future_start, hit.run_length) == (3, 3)
    longer = "Analysts expect 305.7, then 301.2, 299.9 and 310.4."
    hit = leakage_mod.find_future_run(longer, FUTURE)
    assert (hit.future_start, hit.run_length) == (2, 4)


def test_detector_handles_thousands_separators_and_six_digit_rounding() -> None:
    future = [1234567.0, 1234891.0, 1235100.0, 1236000.0]
    text = "values: 1,234,567 then 1,234,891 then 1,235,100 then 1,236,000"
    assert leakage_mod.find_future_run(text, future) is not None
    rounded = [123456.789, 223456.789, 323456.789, 423456.789]
    assert leakage_mod.find_future_run(", ".join(f"{v:.6g}" for v in rounded), rounded) is not None


def test_detector_ignores_unrelated_numbers_wrong_order_and_timestamps() -> None:
    assert leakage_mod.find_future_run("sales were 12, 45, 78, 99 units", FUTURE) is None
    assert leakage_mod.find_future_run(_in_cik_format(list(reversed(FUTURE))), FUTURE) is None
    future = [2026.0, 1.0, 3.0, 5.0]  # numbers that also occur inside timestamps
    assert leakage_mod.find_future_run("recorded on 2026-01-03 05:00:00", future) is None


def test_detector_does_not_count_flat_or_two_valued_runs() -> None:
    # the two coincidences the first version of the scan flagged in real distractor documents
    assert leakage_mod.find_future_run("342, 342, 342, 342, 6", [1.0, 342.0, 342.0, 342.0, 342.0, 6.0, 9.0]) is None
    assert leakage_mod.find_future_run("5, 0, 0, 0", [9.0, 5.0, 0.0, 0.0, 0.0, 7.0]) is None
    assert leakage_mod.find_future_run("6 6 6 6 6 6", [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]) is None


def test_untestable_futures_are_reported_as_such() -> None:
    step_future = [217.0] * 10 + [218.0] * 10 + [219.0] * 10
    assert leakage_mod.is_untestable(step_future)
    assert not leakage_mod.is_untestable(FUTURE)


def test_forecast_input_holds_no_label_field() -> None:
    assert leakage_mod.label_fields_on_forecast_input() == set()


def test_forecast_input_has_no_document_ids() -> None:
    from utrack.data.schema import ForecastInput

    assert "document_ids" not in {f.name for f in dataclasses.fields(ForecastInput)}


def test_builder_modules_never_reference_labels() -> None:
    for module in (builders_mod, placebo_mod, render_mod, preview_mod):
        assert leakage_mod.label_references_in_source(module) == [], module.__name__


def test_source_check_detects_a_reference(tmp_path) -> None:
    path = tmp_path / "bad_module.py"
    path.write_text(
        '"""future_values in a docstring is fine."""\n'
        "from utrack.data.schema import TaskLabels\n"
        "def f(task):\n"
        "    return task.future_values\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("bad_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = leakage_mod.label_references_in_source(module)
    assert any("imports TaskLabels" in f for f in found)
    assert any("attribute .future_values" in f for f in found)


def test_scan_flags_a_planted_leak_in_evidence_and_keeps_metadata_clean() -> None:
    ds = _dataset()
    clean = leakage_report_mod.scan_dev_tasks(ds, 1)
    assert (clean["run_length"] > 0).sum() == 0

    task = ds.tasks["task_1"]
    leaked_text = "Prices will be " + ", ".join(str(v) for v in task.future_values) + "."
    ds.tasks["task_1"] = dataclasses.replace(task, gt_evidence=[EvidenceSpan("E1", leaked_text)])
    scan = leakage_report_mod.scan_dev_tasks(ds, 1)
    hits = scan[scan["run_length"] > 0]
    assert set(hits["surface"]) == {"C1"} and set(hits["benchmark_id"]) == {"task_1"}
    assert scan[(scan["surface"] == "metadata")]["run_length"].sum() == 0


def test_scan_flags_a_planted_leak_in_metadata() -> None:
    ds = _dataset()
    task = ds.tasks["task_2"]
    ds.tasks["task_2"] = dataclasses.replace(task, target_description="Actual: " + _in_cik_format(task.future_values))
    scan = leakage_report_mod.scan_dev_tasks(ds, 1)
    flagged = scan[(scan["surface"] == "metadata") & (scan["run_length"] > 0)]
    assert set(flagged["benchmark_id"]) == {"task_2"}


def test_leakage_report_lists_the_checks(tmp_path) -> None:
    ds = _dataset()
    scan = leakage_report_mod.scan_dev_tasks(ds, 1)
    order = leakage_report_mod.order_position_summary(ds, 1)
    out = tmp_path / "leakage.md"
    leakage_report_mod.write_leakage_report(scan, order, 4, out)
    text = out.read_text(encoding="utf-8")
    assert "Label fields present on `ForecastInput`: none" in text
    assert "Asserted clean" in text and "Does document order still reveal role" in text
    assert "stored rank order" in text and "C4 rendered order" in text
    assert order["stored rank order"][0] < 0.5  # supporting documents are stored first
