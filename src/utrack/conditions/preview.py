"""Condition previews for Teo to read before any paid call, and their SHA-256 list (plan_a.md U0.5, 5.2.7).

Files are produced as bytes (UTF-8, LF only), never through text-mode writes, so the same
inputs give the same bytes on Windows and macOS. Nothing time-dependent goes into a file.
`verify_previews` regenerates every file in memory and compares it with what is on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utrack.conditions.base import Condition
from utrack.conditions.builders import build_condition
from utrack.data.loader import Dataset

SHA256_FILE = "sha256.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_preview(condition: Condition, dataset: Dataset, dataset_revision: str) -> str:
    task = dataset.tasks[condition.benchmark_id]
    lines = [
        f"# {condition.benchmark_id} / {condition.condition_id} ({condition.name})",
        "",
        f"- dataset revision: `{dataset_revision}`",
        f"- render version: `{condition.render_version}`",
        f"- task: {task.entity_name} / {task.time_series_variable}, frequency {task.frequency}, "
        f"horizon {task.prediction_length}, origin {task.origin}",
        f"- seed: {condition.seed if condition.seed is not None else 'none (nothing random)'}",
        f"- approximate tokens (characters / 4): {condition.approx_tokens}",
    ]
    if condition.context is not None:
        lines.append(f"- context sha256: `{_sha256(condition.context.encode('utf-8'))}`")
    if condition.document_ids:
        lines.append(f"- documents used, in rendered order: {', '.join(condition.document_ids)}")
    if condition.evidence_span_ids:
        lines.append(f"- evidence spans used, in rendered order: {', '.join(condition.evidence_span_ids)}")
    if condition.target_approx_tokens is not None:
        lines.append(f"- length-match target C1 approximate tokens (characters / 4): {condition.target_approx_tokens}")
    if condition.source_benchmark_id:
        source = dataset.tasks[condition.source_benchmark_id]
        lines.append(
            f"- placebo evidence taken from: {condition.source_benchmark_id} "
            f"({source.entity_name} / {source.time_series_variable})"
        )
    lines += ["", "## Rendered context", ""]
    lines.append(condition.context if condition.context is not None else "(none: C0 passes no context)")
    return "\n".join(lines) + "\n"


def render_preview_files(
    dataset: Dataset, task_ids: list[str], condition_ids: list[str], base_seed: int, dataset_revision: str
) -> dict[str, bytes]:
    """Relative path -> file bytes, including the SHA-256 list."""
    files: dict[str, bytes] = {}
    for task_id in task_ids:
        for condition_id in condition_ids:
            condition = build_condition(condition_id, dataset, task_id, base_seed)
            text = render_preview(condition, dataset, dataset_revision)
            files[f"{task_id}/{condition_id}.md"] = text.encode("utf-8")
    listing = {
        "algorithm": "sha256 of the file bytes (UTF-8, LF line endings)",
        "files": {path: _sha256(data) for path, data in sorted(files.items())},
    }
    files[SHA256_FILE] = (json.dumps(listing, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return files


def write_previews(files: dict[str, bytes], out_dir: Path) -> None:
    for relative, data in files.items():
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def verify_previews(files: dict[str, bytes], out_dir: Path) -> list[str]:
    """Problems found comparing freshly rendered `files` with what is on disk; empty = identical."""
    problems = []
    for relative, expected in sorted(files.items()):
        path = out_dir / relative
        if not path.exists():
            problems.append(f"missing: {relative}")
        elif path.read_bytes() != expected:
            problems.append(f"differs: {relative}")
    on_disk = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()} if out_dir.exists() else set()
    for relative in sorted(on_disk - set(files)):
        problems.append(f"unexpected extra file: {relative}")
    return problems
