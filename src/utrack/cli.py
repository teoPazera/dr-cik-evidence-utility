"""utrack command-line interface (plan_a.md 5.2.2: Python entry points, not shell scripts)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

from utrack.data import external as external_mod
from utrack.data import snapshot as snapshot_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _machine_name() -> str:
    return os.environ.get("UTRACK_MACHINE", "windows-pc")


def _machine_profile_path() -> Path:
    return REPO_ROOT / "configs" / "machines" / f"{_machine_name()}.yaml"


def _u0_config() -> dict:
    return _load_yaml(REPO_ROOT / "configs" / "u0.yaml")


@click.group()
def main() -> None:
    """Evidence-utility track CLI (plan_a.md)."""


@main.command()
def doctor() -> None:
    """Report toolchain, machine profile, external clones and data snapshot state."""
    load_dotenv(REPO_ROOT / ".env")

    lines: list[str] = []
    lines.append(f"machine profile     : {_machine_name()}")
    profile_path = _machine_profile_path()
    lines.append(f"  profile file      : {profile_path.relative_to(REPO_ROOT)} ({'found' if profile_path.exists() else 'MISSING'})")
    lines.append(f"OS                  : {platform.platform()}")
    lines.append(f"Python              : {platform.python_version()} ({sys.executable})")

    if shutil.which("uv"):
        try:
            uv_out = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True).stdout.strip()
        except subprocess.CalledProcessError as exc:
            uv_out = f"error: {exc}"
    else:
        uv_out = "not found on PATH"
    lines.append(f"uv                  : {uv_out}")

    env_path = REPO_ROOT / ".env"
    lines.append(f".env present        : {env_path.exists()}")

    lines.append("external/ clones vs configs/external_revisions.md:")
    for name, status in external_mod.doctor_status(REPO_ROOT).items():
        lines.append(f"  {name:<32}: {status}")

    problems = snapshot_mod.verify_snapshot(REPO_ROOT)
    if problems:
        lines.append("data/fingerprint.json vs local snapshot: MISMATCH")
        for p in problems:
            lines.append(f"  - {p}")
    else:
        lines.append("data/fingerprint.json vs local snapshot: OK")

    click.echo("\n".join(lines))
    if problems:
        raise SystemExit(1)


@main.group()
def data() -> None:
    """Manage the Dr-CiK dataset snapshot (plan_a.md 3.1, U0.1)."""


@data.command("snapshot")
def data_snapshot() -> None:
    """Download the pinned Dr-CiK configs into data/ and write the fingerprint."""
    cfg = _u0_config()
    fingerprint = snapshot_mod.take_snapshot(REPO_ROOT, cfg["dataset"], _machine_name())
    path = snapshot_mod.write_fingerprint(REPO_ROOT, fingerprint)
    click.echo(f"wrote {path.relative_to(REPO_ROOT)}")

    counts = snapshot_mod.summarize_counts(REPO_ROOT, cfg["dataset"])
    click.echo(json.dumps(counts, indent=2))

    mismatches = snapshot_mod.check_counts(counts, cfg["expected_counts"])
    if mismatches:
        click.echo("count mismatches against configs/u0.yaml expected_counts:")
        for m in mismatches:
            click.echo(f"  - {m}")
        raise SystemExit(1)
    click.echo("counts match configs/u0.yaml expected_counts.")


@data.command("verify")
def data_verify() -> None:
    """Check the local snapshot's checksums against the committed fingerprint."""
    problems = snapshot_mod.verify_snapshot(REPO_ROOT)
    if problems:
        for p in problems:
            click.echo(f"MISMATCH: {p}")
        raise SystemExit(1)
    click.echo("data/fingerprint.json matches the local snapshot.")


@main.group()
def external() -> None:
    """Manage pinned clones of the external reference repositories (plan_a.md 5.3)."""


@external.command("sync")
def external_sync() -> None:
    """Clone/checkout Dr-CiK and context-is-key-forecasting at their pinned commits."""
    summary = external_mod.sync_all(REPO_ROOT, _machine_name())
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
