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
