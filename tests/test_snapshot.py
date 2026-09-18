from pathlib import Path

from utrack.data import snapshot


def test_sha256_and_verify_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "data" / "tasks" / "train.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text('{"a": 1}\n', encoding="utf-8")

    fingerprint = {
        "repo_id": "x/y",
        "revision": "deadbeef",
        "downloaded_at": "now",
        "machine": "windows-pc",
        "files": {
            "data/tasks/train.jsonl": {
                "config": "tasks",
                "sha256": snapshot.sha256_of(f),
                "bytes": f.stat().st_size,
            }
        },
    }
    snapshot.write_fingerprint(tmp_path, fingerprint)
    assert snapshot.verify_snapshot(tmp_path) == []


def test_verify_snapshot_detects_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "data" / "tasks" / "train.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text('{"a": 1}\n', encoding="utf-8")
    fingerprint = {
        "repo_id": "x/y",
        "revision": "deadbeef",
        "downloaded_at": "now",
        "machine": "windows-pc",
        "files": {
            "data/tasks/train.jsonl": {
                "config": "tasks",
                "sha256": snapshot.sha256_of(f),
                "bytes": f.stat().st_size,
            }
        },
    }
    snapshot.write_fingerprint(tmp_path, fingerprint)

    f.write_text('{"a": 2}\n', encoding="utf-8")
    problems = snapshot.verify_snapshot(tmp_path)
    assert len(problems) == 1
    assert "checksum mismatch" in problems[0]


def test_verify_snapshot_detects_missing_file(tmp_path: Path) -> None:
    fingerprint = {
        "repo_id": "x/y",
        "revision": "deadbeef",
        "downloaded_at": "now",
        "machine": "windows-pc",
        "files": {"data/tasks/train.jsonl": {"config": "tasks", "sha256": "0" * 64, "bytes": 0}},
    }
    snapshot.write_fingerprint(tmp_path, fingerprint)
    problems = snapshot.verify_snapshot(tmp_path)
    assert len(problems) == 1
    assert "missing local file" in problems[0]


def test_verify_snapshot_missing_fingerprint(tmp_path: Path) -> None:
    problems = snapshot.verify_snapshot(tmp_path)
    assert problems == ["data/fingerprint.json does not exist"]


def test_count_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n\n', encoding="utf-8")
    assert snapshot.count_jsonl(p) == 2


def test_check_counts_reports_mismatches() -> None:
    counts = {
        "tasks": 279,
        "labels_public_true": 199,
        "labels_public_false": 80,
        "documents": 10341,
        "documents_by_role": {"supporting": 3367, "distractor": 6974},
    }
    expected = {
        "tasks": 279,
        "labels_public_true": 199,
        "labels_public_false": 80,
        "documents": 10342,
        "documents_by_role": {"supporting": 3367, "distractor": 6975},
    }
    mismatches = snapshot.check_counts(counts, expected)
    assert any("documents:" in m for m in mismatches)
    assert any("documents_by_role[distractor]" in m for m in mismatches)
    assert not any("tasks:" in m for m in mismatches)


def test_check_counts_empty_when_matching() -> None:
    counts = {
        "tasks": 279,
        "labels_public_true": 199,
        "labels_public_false": 80,
        "documents": 10342,
        "documents_by_role": {"supporting": 3367, "distractor": 6975},
    }
    assert snapshot.check_counts(counts, counts) == []
