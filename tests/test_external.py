from pathlib import Path

from utrack.data import external


def _write_revisions(repo_root: Path, dr_cik_commit: str, cik_commit: str) -> None:
    configs_dir = repo_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "# External repository revisions\n\n"
        "## Dr-CiK\n"
        "- URL: https://github.com/ServiceNow/Dr-CiK\n"
        f"- Commit: `{dr_cik_commit}`\n"
        "- Synced: 2026-09-18T00:00:00+00:00 on windows-pc\n\n"
        "## context-is-key-forecasting\n"
        "- URL: https://github.com/ServiceNow/context-is-key-forecasting\n"
        f"- Commit: `{cik_commit}`\n"
        "- Synced: 2026-09-18T00:00:00+00:00 on windows-pc\n"
        "\n"
        "Verified paths (Decision D, U0.3):\n"
        "- `cik_benchmark/baselines/direct_prompt.py` (def make_prompt): OK\n"
    )
    (configs_dir / "external_revisions.md").write_text(content, encoding="utf-8")


def test_read_pinned_commits_roundtrip(tmp_path: Path) -> None:
    dr_cik = "abc123def4567890abc123def4567890abc123d"
    cik = "1234567890abcdef1234567890abcdef12345678"
    _write_revisions(tmp_path, dr_cik, cik)

    pinned = external.read_pinned_commits(tmp_path)
    assert pinned["Dr-CiK"] == dr_cik
    assert pinned["context-is-key-forecasting"] == cik


def test_read_pinned_commits_missing_file(tmp_path: Path) -> None:
    assert external.read_pinned_commits(tmp_path) == {}


def test_doctor_status_not_cloned(tmp_path: Path) -> None:
    status = external.doctor_status(tmp_path)
    assert status["Dr-CiK"] == "not cloned"
    assert status["context-is-key-forecasting"] == "not cloned"


def test_verify_targets_reports_missing_file(tmp_path: Path) -> None:
    results = external.verify_targets(tmp_path, "context-is-key-forecasting")
    assert all(r["exists"] is False and r["symbol_found"] is False for r in results)


def test_verify_targets_finds_symbol(tmp_path: Path) -> None:
    p = tmp_path / "external" / "context-is-key-forecasting" / "cik_benchmark" / "baselines"
    p.mkdir(parents=True)
    (p / "direct_prompt.py").write_text("def make_prompt(x):\n    return x\n", encoding="utf-8")
    results = external.verify_targets(tmp_path, "context-is-key-forecasting")
    match = next(r for r in results if r["path"].endswith("direct_prompt.py") and r["symbol"] == "def make_prompt")
    assert match["exists"] is True
    assert match["symbol_found"] is True
