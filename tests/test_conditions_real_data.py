"""Checks against the real snapshot and the committed previews. Skipped where `data/` is absent
(a fresh clone before `utrack data snapshot`); on both machines they must run before any paid call."""

import json
from pathlib import Path

import pytest
import yaml

from utrack.cli import CONDITIONS_PREVIEW, REPO_ROOT, U1_TASKS, _active_condition_ids
from utrack.conditions import preview as preview_mod
from utrack.data.loader import load_dataset
from utrack.reports import leakage as leakage_report_mod

CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "u0.yaml").read_text(encoding="utf-8"))
HAVE_DATA = all((REPO_ROOT / p).exists() for p in CONFIG["dataset"]["configs"].values())
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="data/ snapshot not present; run `utrack data snapshot`")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(REPO_ROOT, CONFIG["dataset"])


def test_committed_previews_reproduce_byte_for_byte(dataset) -> None:
    selection = json.loads((REPO_ROOT / U1_TASKS).read_text(encoding="utf-8"))
    files = preview_mod.render_preview_files(
        dataset,
        [row["benchmark_id"] for row in selection["tasks"]],
        _active_condition_ids(CONFIG),
        CONFIG["conditions"]["seed"],
        CONFIG["dataset"]["revision"],
    )
    assert preview_mod.verify_previews(files, REPO_ROOT / CONDITIONS_PREVIEW) == []


def test_u1_selection_matches_the_config_rule(dataset) -> None:
    from utrack.conditions import selection as selection_mod

    cfg = CONFIG["u1_tasks"]
    expected = selection_mod.select_u1_tasks(dataset, cfg["fixed"], cfg["n_extra"], cfg["seed"])
    recorded = json.loads((REPO_ROOT / U1_TASKS).read_text(encoding="utf-8"))["tasks"]
    assert [r["benchmark_id"] for r in recorded] == [e["benchmark_id"] for e in expected]


def test_no_future_run_reaches_the_forecaster_input_on_any_dev_task(dataset) -> None:
    scan = leakage_report_mod.scan_dev_tasks(dataset, CONFIG["conditions"]["seed"], CONFIG["conditions"]["leakage_min_run"])
    metadata_hits = scan[(scan["surface"] == "metadata") & (scan["run_length"] > 0)]
    assert metadata_hits.empty, sorted(metadata_hits["benchmark_id"])
    assert len(scan["benchmark_id"].unique()) == 199
