"""Download the pinned Dr-CiK dataset configs and fingerprint the local copy.

Downloads go directly to the Hugging Face `resolve/<revision>/<path>` URLs over
plain HTTP rather than through `huggingface_hub`, so no cache directory outside
the repository tree is ever created (plan_a.md 5.2.8). The dataset is public
and non-gated as of U0.1; if it becomes gated, `HF_TOKEN` (see .env.example)
would need to be sent as a bearer token here.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"
FINGERPRINT_RELATIVE_PATH = "data/fingerprint.json"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(repo_id: str, revision: str, remote_path: str, dest: Path, chunk_size: int = 1 << 20) -> str:
    """Stream `remote_path` at `revision` to `dest`, return its SHA-256."""
    url = HF_RESOLVE_URL.format(repo_id=repo_id, revision=revision, path=remote_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        while chunk := response.read(chunk_size):
            fh.write(chunk)
            digest.update(chunk)
    tmp.replace(dest)
    return digest.hexdigest()


def take_snapshot(repo_root: Path, dataset_cfg: dict, machine_name: str) -> dict:
    """Download every configured file and build the fingerprint record (not yet written)."""
    repo_id = dataset_cfg["repo_id"]
    revision = dataset_cfg["revision"]
    files: dict[str, dict] = {}
    for name, remote_path in dataset_cfg["configs"].items():
        dest = repo_root / remote_path
        digest = download_file(repo_id, revision, remote_path, dest)
        files[remote_path] = {"config": name, "sha256": digest, "bytes": dest.stat().st_size}
    return {
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine_name,
        "files": files,
    }


def write_fingerprint(repo_root: Path, fingerprint: dict) -> Path:
    path = repo_root / FINGERPRINT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_snapshot(repo_root: Path, fingerprint_path: Path | None = None) -> list[str]:
    """Return problems comparing the local snapshot to the committed fingerprint (empty = match)."""
    path = fingerprint_path or (repo_root / FINGERPRINT_RELATIVE_PATH)
    if not path.exists():
        return [f"{FINGERPRINT_RELATIVE_PATH} does not exist"]
    committed = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for remote_path, meta in committed["files"].items():
        local = repo_root / remote_path
        if not local.exists():
            problems.append(f"missing local file: {remote_path}")
            continue
        digest = sha256_of(local)
        if digest != meta["sha256"]:
            problems.append(f"checksum mismatch: {remote_path}")
    return problems


def count_jsonl(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def summarize_counts(repo_root: Path, dataset_cfg: dict) -> dict:
    """Minimal record counts for the U0.1 CHECK. Full schema audit is U0.2's job."""
    tasks_path = repo_root / dataset_cfg["configs"]["tasks"]
    documents_path = repo_root / dataset_cfg["configs"]["documents"]

    task_count = 0
    labels_public_true = 0
    labels_public_false = 0
    with open(tasks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_count += 1
            if record.get("labels_public"):
                labels_public_true += 1
            else:
                labels_public_false += 1

    doc_count = 0
    documents_by_role: dict[str, int] = {}
    with open(documents_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_count += 1
            roles = record.get("roles") or [record.get("role")]
            role = roles[0] if roles else None
            documents_by_role[role] = documents_by_role.get(role, 0) + 1

    return {
        "tasks": task_count,
        "labels_public_true": labels_public_true,
        "labels_public_false": labels_public_false,
        "documents": doc_count,
        "documents_by_role": documents_by_role,
    }


def check_counts(counts: dict, expected: dict) -> list[str]:
    """Compare `summarize_counts` output to `configs/u0.yaml`'s `expected_counts`. Empty = match."""
    mismatches: list[str] = []
    for key in ("tasks", "labels_public_true", "labels_public_false", "documents"):
        if counts.get(key) != expected.get(key):
            mismatches.append(f"{key}: got {counts.get(key)}, expected {expected.get(key)}")
    for role, expected_count in expected.get("documents_by_role", {}).items():
        got = counts.get("documents_by_role", {}).get(role)
        if got != expected_count:
            mismatches.append(f"documents_by_role[{role}]: got {got}, expected {expected_count}")
    return mismatches
