import json

from click.testing import CliRunner

from utrack.cli import main


def test_doctor_runs_without_crashing() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert "machine profile" in result.output
    assert "OS" in result.output
    assert "Python" in result.output


def test_data_verify_reports_missing_fingerprint_cleanly(tmp_path, monkeypatch) -> None:
    import utrack.cli as cli_mod

    monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["data", "verify"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_external_group_has_sync_command() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["external", "--help"])
    assert result.exit_code == 0
    assert "sync" in result.output


def test_data_audit_writes_artifacts(tmp_path, monkeypatch) -> None:
    import utrack.cli as cli_mod

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "u0.yaml").write_text(
        "dataset:\n"
        "  repo_id: x/y\n"
        "  revision: abc\n"
        "  configs:\n"
        "    tasks: data/tasks/train.jsonl\n"
        "    documents: data/documents/train.jsonl\n"
        "    task_documents: data/task_documents/train.jsonl\n",
        encoding="utf-8",
    )

    task = {
        "benchmark_id": "task_x", "split": "open", "origin": "synthetic", "labels_public": True,
        "reasoning_hops": 1, "entity_name": "e", "entity_type": "t", "profile_id": "1", "profile_name": "p",
        "profile_details": {}, "time_series_variable": "v", "frequency": "1 day", "prediction_length": 1,
        "seasonal_period": "1D", "target_description": "d",
        "history_timestamps": ["2026-01-01 00:00:00"], "history_values": [1.0],
        "future_timestamps": ["2026-01-02 00:00:00"], "future_values": [2.0],
        "document_ids": ["doc_1"], "gt_evidence": [{"id": "E1", "evidence": "n"}],
        "raw_task_path": "tasks/task_x.json",
    }
    doc = {
        "document_id": "doc_1", "raw_document_path": "p", "task_ids": ["task_x"],
        "roles": ["supporting"], "subtypes": [None], "text": "hello",
    }
    tdoc = {
        "benchmark_id": "task_x", "document_id": "doc_1", "rank": 0,
        "role": "supporting", "subtype": None, "raw_document_path": "p",
    }

    for rel, record in [
        ("data/tasks/train.jsonl", task),
        ("data/documents/train.jsonl", doc),
        ("data/task_documents/train.jsonl", tdoc),
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["data", "audit"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "artifacts" / "u0" / "task_index.parquet").exists()
    assert (tmp_path / "artifacts" / "u0" / "data_audit.md").exists()
