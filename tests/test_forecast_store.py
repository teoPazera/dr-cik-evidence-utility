import numpy as np

from utrack.forecasters.base import ForecasterOutput
from utrack.store.forecast_store import ForecastStore, build_cell_record
from utrack.store.manifest import RunManifest, config_hash, new_run_id


def _make_output(seed=1, n=3, horizon=4) -> ForecasterOutput:
    return ForecasterOutput(
        samples=np.arange(n * horizon, dtype=float).reshape(n, horizon),
        forecaster_name="last_value_naive",
        forecaster_version="1.0.0",
        model_identifier=None,
        n_requested=n,
        n_valid=n,
        seed=seed,
    )


def test_build_cell_record_has_required_fields() -> None:
    record = build_cell_record(
        benchmark_id="task_1",
        condition_id="C0",
        repeat=0,
        condition_seed=99,
        output=_make_output(),
        dataset_revision="rev123",
        code_commit="abc123",
        config_hash="hash456",
    )
    for key in (
        "benchmark_id", "condition_id", "repeat", "condition_seed",
        "forecaster_name", "forecaster_version", "model_identifier", "seed",
        "sampling_params", "token_counts", "cost_usd", "n_requested", "n_valid",
        "generated_at", "dataset_revision", "code_commit", "config_hash", "samples",
    ):
        assert key in record
    assert record["samples"] == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]


def test_forecast_store_append_and_read_roundtrip(tmp_path) -> None:
    store = ForecastStore(tmp_path / "store" / "cells.jsonl")
    r1 = build_cell_record(
        benchmark_id="task_1", condition_id="C0", repeat=0, condition_seed=0,
        output=_make_output(seed=1), dataset_revision="rev", code_commit="c1", config_hash="h1",
    )
    r2 = build_cell_record(
        benchmark_id="task_2", condition_id="C0", repeat=0, condition_seed=0,
        output=_make_output(seed=2), dataset_revision="rev", code_commit="c1", config_hash="h1",
    )
    store.append(r1)
    store.append(r2)

    records = store.read_all()
    assert len(records) == 2
    assert {r["benchmark_id"] for r in records} == {"task_1", "task_2"}

    arr = ForecastStore.samples_as_array(records[0])
    assert arr.shape == (3, 4)


def test_forecast_store_read_all_missing_file_returns_empty(tmp_path) -> None:
    store = ForecastStore(tmp_path / "does_not_exist.jsonl")
    assert store.read_all() == []


def test_config_hash_is_stable_and_order_independent() -> None:
    a = config_hash({"x": 1, "y": 2})
    b = config_hash({"y": 2, "x": 1})
    assert a == b
    assert config_hash({"x": 1, "y": 3}) != a


def test_new_run_id_is_a_nonempty_string() -> None:
    assert isinstance(new_run_id(), str)
    assert len(new_run_id()) > 0


def test_run_manifest_write_and_reload(tmp_path) -> None:
    manifest = RunManifest(
        run_id="r1", machine="windows-pc", dataset_revision="rev", code_commit="c1",
        config_hash="h1", started_at="t0", n_cells=2,
        forecaster_names=["last_value_naive"], condition_ids=["C0"],
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)

    import json
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "r1"
    assert loaded["n_cells"] == 2
