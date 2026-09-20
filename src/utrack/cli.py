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

from utrack.conditions import preview as preview_mod
from utrack.conditions import selection as selection_mod
from utrack.data import audit as audit_mod
from utrack.data import external as external_mod
from utrack.data import loader as loader_mod
from utrack.data import snapshot as snapshot_mod
from utrack.reports import baseline as baseline_mod
from utrack.reports import cost_estimate as cost_mod
from utrack.reports import leakage as leakage_mod
from utrack.reports import u1_smoke as u1_smoke_mod
from utrack.reports import u1 as u1_mod
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


U1_TASKS = Path("artifacts") / "u0" / "u1_tasks.json"
CONDITIONS_PREVIEW = Path("artifacts") / "u0" / "conditions_preview"
LEAKAGE_REPORT = Path("artifacts") / "u0" / "leakage_report.md"


def _active_condition_ids(cfg: dict) -> list[str]:
    if not cfg["conditions"].get("placebo_length_matched", False):
        raise ValueError("Decision H requires the length-matched C3 placebo; set conditions.placebo_length_matched: true")
    ids = ["C0", "C1", "C2", "C3"]
    if cfg["conditions"]["enable_c4"]:
        ids.append("C4")
    return ids


@main.group()
def conditions() -> None:
    """Build and inspect the experimental conditions C0 to C4 (plan_a.md 3.3, U0.5)."""


@conditions.command("select")
def conditions_select() -> None:
    """Select the configured U1 task set and write artifacts/u0/u1_tasks.json."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    fixed, n_extra, seed = cfg["u1_tasks"]["fixed"], cfg["u1_tasks"]["n_extra"], cfg["u1_tasks"]["seed"]
    selected = selection_mod.select_u1_tasks(dataset, fixed, n_extra, seed)
    record = selection_mod.selection_record(
        dataset,
        selected,
        fixed=fixed,
        n_extra=n_extra,
        seed=seed,
        base_seed=cfg["conditions"]["seed"],
        condition_ids=_active_condition_ids(cfg),
        dataset_revision=cfg["dataset"]["revision"],
        selection_mode=cfg["u1_tasks"].get("selection_mode", "seeded_coverage"),
        rationale=cfg["u1_tasks"].get("rationale"),
    )
    path = REPO_ROOT / U1_TASKS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for row in record["tasks"]:
        click.echo(
            f"{row['benchmark_id']:<10} {row['source']:<18} {row['frequency']:<10} "
            f"horizon {row['prediction_length']:<4} placebo from {row['placebo_source']}"
        )
    click.echo(f"wrote {U1_TASKS}")


def _preview_files(cfg: dict, dataset: loader_mod.Dataset) -> dict[str, bytes]:
    selection = json.loads((REPO_ROOT / U1_TASKS).read_text(encoding="utf-8"))
    return preview_mod.render_preview_files(
        dataset,
        [row["benchmark_id"] for row in selection["tasks"]],
        _active_condition_ids(cfg),
        cfg["conditions"]["seed"],
        cfg["dataset"]["revision"],
    )


@conditions.command("preview")
def conditions_preview() -> None:
    """Write the rendered context of every U1 cell plus a SHA-256 list."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    files = _preview_files(cfg, dataset)
    preview_mod.write_previews(files, REPO_ROOT / CONDITIONS_PREVIEW)
    click.echo(f"wrote {len(files) - 1} previews and {preview_mod.SHA256_FILE} under {CONDITIONS_PREVIEW}")


@conditions.command("verify")
def conditions_verify() -> None:
    """Regenerate the previews in memory and compare with the committed files, byte for byte."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    problems = preview_mod.verify_previews(_preview_files(cfg, dataset), REPO_ROOT / CONDITIONS_PREVIEW)
    if problems:
        for p in problems:
            click.echo(f"MISMATCH: {p}")
        raise SystemExit(1)
    click.echo("conditions previews reproduce byte for byte.")


@conditions.command("leakage")
def conditions_leakage() -> None:
    """Scan all dev tasks and conditions for runs of future values; write the leakage report."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    min_run = cfg["conditions"]["leakage_min_run"]
    scan = leakage_mod.scan_dev_tasks(dataset, cfg["conditions"]["seed"], min_run)
    order = leakage_mod.order_position_summary(dataset, cfg["conditions"]["seed"])
    leakage_mod.write_leakage_report(scan, order, min_run, REPO_ROOT / LEAKAGE_REPORT)
    click.echo(f"wrote {LEAKAGE_REPORT}")

    input_hits = scan[(scan["surface"] == "metadata") & (scan["run_length"] > 0)]
    if input_hits.empty and not leakage_mod.label_fields_on_forecast_input():
        click.echo("leakage test passed: the forecaster input's metadata holds no run of future values and no label field.")
    else:
        click.echo(f"LEAK: {sorted(input_hits['benchmark_id'])}")
        raise SystemExit(1)


U1_PROMPTS = Path("artifacts") / "u1" / "prompt_preview"


@main.group()
def prompts() -> None:
    """Render exact U1 LiteLLM request payloads for review. Nothing is sent."""


@prompts.command("u1")
@click.option("--task", "task_ids", multiple=True, help="Benchmark id to render; defaults to all selected U1 tasks.")
@click.option("--condition", "condition_ids", multiple=True, help="Condition id to render; defaults to enabled U1 conditions.")
def prompts_u1(task_ids: tuple[str, ...], condition_ids: tuple[str, ...]) -> None:
    """Write exact Gemini/LiteLLM JSON payloads for Teo to inspect before paid calls."""
    from utrack.conditions.builders import build_condition
    from utrack.forecasters.llm_direct import LiteLLMDirectForecaster
    from utrack.forecasters.llm_prompt import TEMPLATE_VERSION, render_user_prompt

    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    selection = json.loads((REPO_ROOT / U1_TASKS).read_text(encoding="utf-8"))
    selected_task_ids = list(task_ids) or [row["benchmark_id"] for row in selection["tasks"]]
    selected_condition_ids = list(condition_ids) or _active_condition_ids(cfg)
    for task_id in selected_task_ids:
        if task_id not in dataset.tasks:
            raise click.ClickException(f"unknown task: {task_id}")
        forecast_input = dataset.forecast_input(task_id)
        for condition_id in selected_condition_ids:
            condition = build_condition(condition_id, dataset, task_id, cfg["conditions"]["seed"])
            prompt = render_user_prompt(forecast_input, condition.context)
            payload = {
                "model": "gemini-3.1-flash-lite",
                "messages": LiteLLMDirectForecaster._messages(prompt),
                "temperature": 1.0,
                "max_tokens": int(_load_yaml(REPO_ROOT / "configs" / "u1.yaml")["max_output_tokens"]),
            }
            body = {
                "purpose": "review-only exact LiteLLM /chat/completions payload; no request was sent",
                "task": task_id,
                "condition": condition_id,
                "template_version": TEMPLATE_VERSION,
                "condition_metadata": {
                    "document_ids": list(condition.document_ids),
                    "evidence_span_ids": list(condition.evidence_span_ids),
                    "source_benchmark_id": condition.source_benchmark_id,
                    "seed": condition.seed,
                    "render_version": condition.render_version,
                },
                "payload": payload,
            }
            path = REPO_ROOT / U1_PROMPTS / task_id / f"{condition_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            click.echo(f"wrote {path.relative_to(REPO_ROOT)}")


COST_ESTIMATE = Path("artifacts") / "u0" / "u1_cost_estimate.md"


@main.group()
def estimate() -> None:
    """Dry-run estimates. Nothing is sent to any provider."""


@estimate.command("u1")
def estimate_u1() -> None:
    """U0.6: build every U1 request with the draft prompt, count tokens, write u1_cost_estimate.md."""
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    selection = json.loads((REPO_ROOT / U1_TASKS).read_text(encoding="utf-8"))
    cost_cfg = cfg["cost_estimate"]
    requests = cost_mod.build_u1_requests(
        dataset,
        [row["benchmark_id"] for row in selection["tasks"]],
        _active_condition_ids(cfg),
        cost_cfg["repeats"],
        cost_cfg["n_samples"],
        cfg["conditions"]["seed"],
    )
    df = cost_mod.estimate_rows(dataset, requests, cost_cfg["token_models"])
    leaks = cost_mod.request_leaks(dataset, requests, cfg["conditions"]["leakage_min_run"])
    pair_matches = cost_mod.request_pair_matches(dataset, requests)
    example = next(
        (r for r in requests if (r.benchmark_id, r.condition_id) == cost_mod.EXAMPLE_REQUEST and r.repeat == 0), None
    )
    cost_mod.write_cost_estimate(df, cfg, leaks, pair_matches, example, REPO_ROOT / COST_ESTIMATE)
    click.echo(f"built {len(requests)} requests (nothing sent); wrote {COST_ESTIMATE}")
    if leaks:
        click.echo(f"LEAK in request text: {leaks}")
        raise SystemExit(1)


@main.group()
def run() -> None:
    """Run forecasters and write raw trajectories to the forecast store."""


U1_SMOKE_STORE = Path("artifacts") / "u1" / "smoke" / "cells.jsonl"
U1_SMOKE_MANIFEST = Path("artifacts") / "u1" / "smoke" / "manifest.json"
U1_SMOKE_SCORES = Path("artifacts") / "u1" / "smoke" / "scores.parquet"
U1_SMOKE_REPORT = Path("artifacts") / "u1" / "smoke" / "report.md"
U1_SMOKE_PLOT = Path("artifacts") / "u1" / "smoke" / "task42_c0_c1_trajectories.png"
U1_SMOKE_CHECK = Path("artifacts") / "u1" / "smoke" / "u1_2_check.md"


def _format_u1_progress(progress: dict) -> str:
    """One flushed, human-readable progress line for long-running paid U1 calls."""
    cell = f"cell {progress.get('cell_index', '?')}/{progress.get('total_cells', '?')}"
    task = progress.get("benchmark_id", "?")
    condition = progress.get("condition_id", "?")
    repeat = progress.get("repeat")
    repeat_text = f", repeat {repeat + 1}" if isinstance(repeat, int) else ""
    samples = (
        f"samples {progress.get('valid_samples', 0)}/{progress.get('requested_samples', progress.get('n_samples', '?'))}"
    )
    attempts = progress.get("attempts")
    max_attempts = progress.get("max_attempts")
    attempt_text = f", attempt {attempts}/{max_attempts}" if attempts is not None else ""
    spent = progress.get("spent_usd")
    cap = progress.get("cost_cap_usd")
    cost_text = f", spend ${spent:.4f}/${cap:.2f}" if isinstance(spent, (int, float)) and isinstance(cap, (int, float)) else ""
    detail = progress.get("message")
    suffix = f" — {detail}" if detail else ""
    return f"[U1 {progress.get('event', 'progress')}] {cell}: {task}/{condition}{repeat_text}, {samples}{attempt_text}{cost_text}{suffix}"



U1_STORE = Path("artifacts") / "u1" / "forecast_store" / "u1_cells.jsonl"
U1_MANIFEST = Path("artifacts") / "u1" / "forecast_store" / "u1_manifest.json"
U1_LEDGER = Path("artifacts") / "u1" / "cost_ledger.md"
U1_SCORES = Path("artifacts") / "u1" / "u1_scores.parquet"
U1_REVIEW_DIR = Path("artifacts") / "u1" / "review"
U1_REPORT = Path("artifacts") / "u1" / "u1_report.md"


@run.command("u1")
@click.option("--task", "task_ids", multiple=True, help="Task id; repeat for a subset. Defaults to configured U1 tasks.")
@click.option("--condition", "condition_ids", multiple=True, help="Condition id; repeat for a subset. Defaults to enabled C0-C3.")
@click.option("--repeat", "repeats", multiple=True, type=int, help="Zero-based repeat; defaults to 0 only for staged execution.")
@click.option("--no-resume", is_flag=True, help="Fail-safe option: do not skip cells already in the append-only store.")
def run_u1_cmd(task_ids: tuple[str, ...], condition_ids: tuple[str, ...], repeats: tuple[int, ...], no_resume: bool) -> None:
    """Paid, resumable U1 cells; writes a live cost ledger after every stored cell."""
    cfg = _u0_config(); u1_cfg = _load_yaml(REPO_ROOT / "configs" / "u1.yaml")
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    selected = list(task_ids) or [r["benchmark_id"] for r in json.loads((REPO_ROOT / U1_TASKS).read_text())["tasks"]]
    conditions = list(condition_ids) or _active_condition_ids(cfg)
    selected_repeats = list(repeats) or [0]
    unknown = set(selected) - set(dataset.tasks)
    if unknown: raise click.ClickException(f"unknown task ids: {sorted(unknown)}")
    unknown_conditions = set(conditions) - {"C0", "C1", "C2", "C3", "C4"}
    if unknown_conditions: raise click.ClickException(f"unknown conditions: {sorted(unknown_conditions)}")
    if any(r < 0 for r in selected_repeats): raise click.ClickException("repeats must be non-negative")
    click.echo(f"[U1] planned {len(selected)*len(conditions)*len(selected_repeats)} cells; emergency cap ${float(u1_cfg['cost_cap_usd']):.2f}.")
    manifest = u1_mod.run_u1(REPO_ROOT, dataset, cfg, u1_cfg, _machine_name(), REPO_ROOT / U1_STORE, REPO_ROOT / U1_MANIFEST, REPO_ROOT / U1_LEDGER, selected, conditions, selected_repeats, resume=not no_resume, progress_callback=lambda p: click.echo(_format_u1_progress(p), err=True))
    click.echo(f"stored {manifest.n_cells} new cells; manifest {U1_MANIFEST}; live ledger {U1_LEDGER}")


@run.command("u1-smoke")
def run_u1_smoke_cmd() -> None:
    """Paid U1.2-style smoke: task_42, C0/C1, one repeat, 25 samples each; prints live progress."""
    if U1_SMOKE_STORE.exists():
        raise click.ClickException(f"{U1_SMOKE_STORE} already exists; smoke stores are append-only.")
    cfg = _u0_config()
    u1_cfg = _load_yaml(REPO_ROOT / "configs" / "u1.yaml")
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    click.echo("[U1] starting smoke run: 2 cells × 25 requested samples; progress is printed after every provider attempt.")
    manifest = u1_smoke_mod.run_smoke(
        REPO_ROOT,
        dataset,
        cfg,
        u1_cfg,
        _machine_name(),
        REPO_ROOT / U1_SMOKE_STORE,
        REPO_ROOT / U1_SMOKE_MANIFEST,
        progress_callback=lambda progress: click.echo(_format_u1_progress(progress), err=True),
    )
    click.echo(f"wrote {manifest.n_cells} smoke cells to {U1_SMOKE_STORE}")
    click.echo(f"wrote {U1_SMOKE_MANIFEST}")


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



@score.command("u1")
def score_u1_cmd() -> None:
    """Score the staged/full U1 forecast store under A1-A3; A3/B1 is headline."""
    if not (REPO_ROOT / U1_STORE).exists(): raise click.ClickException(f"{U1_STORE} not found; run `utrack run u1` first")
    scores = u1_mod.score_u1(loader_mod.load_dataset(REPO_ROOT, _u0_config()["dataset"]), _u0_config(), REPO_ROOT / U1_STORE)
    if scores.empty: raise click.ClickException("no scoreable U1 cells")
    (REPO_ROOT / U1_SCORES).parent.mkdir(parents=True, exist_ok=True); scores.to_parquet(REPO_ROOT / U1_SCORES, index=False)
    click.echo(f"wrote {U1_SCORES} ({len(scores)} rows)")



@score.command("u1-smoke")
def score_u1_smoke_cmd() -> None:
    """Score the task_42 U1 smoke store. No provider requests are made."""
    if not U1_SMOKE_STORE.exists():
        raise click.ClickException(f"{U1_SMOKE_STORE} not found; run `utrack run u1-smoke` first")
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    df = u1_smoke_mod.score_smoke(REPO_ROOT, dataset, cfg, REPO_ROOT / U1_SMOKE_STORE)
    if df.empty:
        raise click.ClickException("no scoreable smoke cells (need at least two valid trajectories)")
    U1_SMOKE_SCORES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REPO_ROOT / U1_SMOKE_SCORES, index=False)
    click.echo(f"wrote {U1_SMOKE_SCORES} ({len(df)} rows)")


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



@report.command("u1")
def report_u1_cmd() -> None:
    """Generate staged U1 score table and per-task median/quantile review plots."""
    import pandas as pd
    if not (REPO_ROOT / U1_SCORES).exists(): raise click.ClickException(f"{U1_SCORES} not found; run `utrack score u1` first")
    u1_mod.write_u1_review(loader_mod.load_dataset(REPO_ROOT, _u0_config()["dataset"]), pd.read_parquet(REPO_ROOT / U1_SCORES), REPO_ROOT / U1_STORE, REPO_ROOT / U1_REVIEW_DIR, REPO_ROOT / U1_REPORT)
    click.echo(f"wrote {U1_REPORT} and plots under {U1_REVIEW_DIR}")


@report.command("u1-smoke")
def report_u1_smoke_cmd() -> None:
    """Write the U1 smoke CRPS, baseline comparison, token and cache-cost report."""
    import pandas as pd
    if not U1_SMOKE_SCORES.exists():
        raise click.ClickException(f"{U1_SMOKE_SCORES} not found; run `utrack score u1-smoke` first")
    df = pd.read_parquet(REPO_ROOT / U1_SMOKE_SCORES)
    u1_smoke_mod.write_smoke_report(df, REPO_ROOT / BASELINE_SCORES, REPO_ROOT / U1_SMOKE_REPORT)
    click.echo(f"wrote {U1_SMOKE_REPORT}")


@report.command("u1-smoke-check")
def report_u1_smoke_check_cmd() -> None:
    """Write U1.2 plots plus kill-condition and dry-run comparison check."""
    if not U1_SMOKE_SCORES.exists():
        raise click.ClickException(f"{U1_SMOKE_SCORES} not found; score the retained smoke store first")
    store_path = REPO_ROOT / "artifacts" / "u1" / "smoke" / "cells_25samples.jsonl"
    if not store_path.exists():
        raise click.ClickException(f"{store_path.relative_to(REPO_ROOT)} not found")
    import pandas as pd
    cfg = _u0_config()
    dataset = loader_mod.load_dataset(REPO_ROOT, cfg["dataset"])
    scores = pd.read_parquet(REPO_ROOT / U1_SMOKE_SCORES)
    u1_smoke_mod.write_smoke_plots(dataset, store_path, REPO_ROOT / U1_SMOKE_PLOT)
    u1_smoke_mod.write_smoke_check(
        dataset, scores, store_path, REPO_ROOT / COST_ESTIMATE, REPO_ROOT / U1_SMOKE_CHECK
    )
    click.echo(f"wrote {U1_SMOKE_PLOT}")
    click.echo(f"wrote {U1_SMOKE_CHECK}")


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
