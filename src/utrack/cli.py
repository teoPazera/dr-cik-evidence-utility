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

from utrack.data import audit as audit_mod
from utrack.data import external as external_mod
from utrack.data import loader as loader_mod
from utrack.data import snapshot as snapshot_mod
from utrack.reports import baseline as baseline_mod
from utrack.store.forecast_store import ForecastStore

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where U0.4's outputs live. The forecast store is git-ignored (plan_a.md 5.2.9: free and
# deterministic, regenerated where needed); the scores parquet and report are committed.
BASELINE_STORE = Path("artifacts") / "u0" / "forecast_store" / "baseline_cells.jsonl"
BASELINE_MANIFEST = Path("artifacts") / "u0" / "forecast_store" / "baseline_manifest.json"
BASELINE_SCORES = Path("artifacts") / "u0" / "baseline_scores.parquet"
BASELINE_REPORT = Path("artifacts") / "u0" / "baseline_report.md"


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


@data.command("audit")
def data_audit_cmd() -> None:
    """Load the snapshot, validate invariants, and write task_index.parquet + data_audit.md."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])

    raw_field_names = {
        name: sorted({k for r in loader_mod.load_jsonl(REPO_ROOT / path) for k in r.keys()})
        for name, path in cfg["dataset"]["configs"].items()
    }

    violations = audit_mod.validate_invariants(dataset)
    rank_finding = audit_mod.rank_reveals_role(dataset)
    index_df = audit_mod.compute_index(dataset)

    index_path = REPO_ROOT / "artifacts" / "u0" / "task_index.parquet"
    audit_mod.write_task_index(index_df, index_path)

    audit_path = REPO_ROOT / "artifacts" / "u0" / "data_audit.md"
    audit_mod.write_data_audit(
        dataset=dataset,
        raw_field_names=raw_field_names,
        violations=violations,
        rank_finding=rank_finding,
        index_df=index_df,
        out_path=audit_path,
    )

    total_violations = sum(len(v) for v in violations.values())
    click.echo(f"wrote {index_path.relative_to(REPO_ROOT)} ({len(index_df)} rows)")
    click.echo(
        f"wrote {audit_path.relative_to(REPO_ROOT)} "
        f"({total_violations} invariant violations across {len(violations)} checks)"
    )


@main.group()
def external() -> None:
    """Manage pinned clones of the external reference repositories (plan_a.md 5.3)."""


@external.command("sync")
def external_sync() -> None:
    """Clone/checkout Dr-CiK and context-is-key-forecasting at their pinned commits."""
    summary = external_mod.sync_all(REPO_ROOT, _machine_name())
    click.echo(json.dumps(summary, indent=2))


@main.group()
def run() -> None:
    """Run forecasters and write raw trajectories to the forecast store."""


@run.command("baseline")
def run_baseline_cmd() -> None:
    """U0.4: zero-cost statistical forecasters, condition C0, all dev tasks."""
    store_path = REPO_ROOT / BASELINE_STORE
    if store_path.exists():
        raise click.ClickException(
            f"{BASELINE_STORE} already exists; the store is append-only, so a rerun would duplicate "
            "every cell. Delete it (it is git-ignored and deterministic) to regenerate."
        )
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    manifest = baseline_mod.run_baseline(
        REPO_ROOT, dataset, cfg, _machine_name(), store_path, REPO_ROOT / BASELINE_MANIFEST
    )
    click.echo(f"wrote {manifest.n_cells} cells to {BASELINE_STORE}")
    click.echo(f"wrote {BASELINE_MANIFEST}")


@main.group()
def score() -> None:
    """Score stored forecasts (pure function of the store; free to rerun)."""


@score.command("baseline")
def score_baseline_cmd() -> None:
    """Score the U0.4 baseline store under all three scalings."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    store = ForecastStore(REPO_ROOT / BASELINE_STORE)
    df = baseline_mod.score_baseline(dataset, store, cfg)
    if df.empty:
        raise click.ClickException(f"no cells found in {BASELINE_STORE}; run `utrack run baseline` first")
    baseline_mod.write_baseline_scores(df, REPO_ROOT / BASELINE_SCORES)
    click.echo(f"wrote {BASELINE_SCORES} ({len(df)} rows)")


@main.group()
def report() -> None:
    """Write human-readable reports from scored results."""


@report.command("baseline")
def report_baseline_cmd() -> None:
    """Write the U0.4 baseline report from baseline_scores.parquet."""
    import pandas as pd

    scores_path = REPO_ROOT / BASELINE_SCORES
    if not scores_path.exists():
        raise click.ClickException(f"{BASELINE_SCORES} not found; run `utrack score baseline` first")
    df = pd.read_parquet(scores_path)
    baseline_mod.write_baseline_report(
        df, REPO_ROOT / BASELINE_REPORT, paper_reference=baseline_mod.PAPER_REFERENCE
    )
    click.echo(f"wrote {BASELINE_REPORT}")


if __name__ == "__main__":
    main()
