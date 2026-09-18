import numpy as np
import pandas as pd

from utrack.data.loader import Dataset
from utrack.data.schema import EvidenceSpan, Task
from utrack.reports import baseline as baseline_mod
from utrack.store.forecast_store import ForecastStore


def _task(benchmark_id: str, *, labels_public: bool = True, seasonal_period=None, future=None) -> Task:
    rng = np.random.default_rng(int(benchmark_id.split("_")[1]))
    history = list(50 + 5 * np.sin(np.arange(40) / 2.0) + rng.normal(0, 1, 40))
    horizon = 5
    if future is None:
        future = list(50 + 5 * np.sin(np.arange(40, 40 + horizon) / 2.0))
    return Task(
        benchmark_id=benchmark_id,
        split="open",
        origin="synthetic" if labels_public else "human",
        labels_public=labels_public,
        reasoning_hops=1,
        entity_name="e",
        entity_type="t",
        profile_id="1",
        profile_name="p",
        profile_details={},
        time_series_variable="v",
        frequency="1 hour",
        prediction_length=horizon,
        seasonal_period=seasonal_period,
        target_description="d",
        history_timestamps=[f"2026-01-01 {h:02d}:00:00" for h in range(24)] + [f"2026-01-02 {h:02d}:00:00" for h in range(16)],
        history_values=history,
        future_timestamps=[f"2026-01-03 {h:02d}:00:00" for h in range(horizon)],
        future_values=future if labels_public else [],
        document_ids=[],
        gt_evidence=[EvidenceSpan(id="E1", evidence="x")] if labels_public else [],
        raw_task_path="p",
    )


def _dataset() -> Dataset:
    tasks = {
        "task_1": _task("task_1", seasonal_period=4),
        "task_2": _task("task_2", seasonal_period=None),
        "task_3": _task("task_3", labels_public=False),
        "task_4": _task("task_4", future=[7.0] * 5),  # constant future -> A1 denominator is 0
    }
    return Dataset(tasks=tasks, documents_by_task={})


def _config() -> dict:
    return {
        "dataset": {"revision": "rev123"},
        "baseline": {
            "condition_id": "C0",
            "n_samples": 20,
            "seed": 7,
            "forecasters": ["last_value_naive", "seasonal_naive"],
            "scaling_a2_fallback_lag_steps": 1,
        },
    }


def test_cell_seed_is_deterministic_and_input_sensitive() -> None:
    a = baseline_mod._cell_seed(1, "task_1", "C0", "last_value_naive")
    assert a == baseline_mod._cell_seed(1, "task_1", "C0", "last_value_naive")
    assert a != baseline_mod._cell_seed(1, "task_2", "C0", "last_value_naive")
    assert a != baseline_mod._cell_seed(2, "task_1", "C0", "last_value_naive")


def test_run_baseline_covers_dev_tasks_only(tmp_path) -> None:
    store_path = tmp_path / "store" / "cells.jsonl"
    manifest = baseline_mod.run_baseline(
        tmp_path, _dataset(), _config(), "windows-pc", store_path, tmp_path / "store" / "manifest.json"
    )
    records = ForecastStore(store_path).read_all()

    assert manifest.n_cells == 6  # 3 dev tasks x 2 forecasters
    assert len(records) == 6
    assert {r["benchmark_id"] for r in records} == {"task_1", "task_2", "task_4"}
    assert all(r["condition_id"] == "C0" for r in records)
    assert all(r["dataset_revision"] == "rev123" for r in records)
    assert all(np.asarray(r["samples"]).shape == (20, 5) for r in records)
    assert (tmp_path / "store" / "manifest.json").exists()


def test_run_baseline_is_deterministic(tmp_path) -> None:
    for name in ("a", "b"):
        baseline_mod.run_baseline(
            tmp_path, _dataset(), _config(), "windows-pc",
            tmp_path / name / "cells.jsonl", tmp_path / name / "manifest.json",
        )
    a = ForecastStore(tmp_path / "a" / "cells.jsonl").read_all()
    b = ForecastStore(tmp_path / "b" / "cells.jsonl").read_all()
    for ra, rb in zip(a, b):
        assert ra["samples"] == rb["samples"]


def test_score_baseline_columns_and_degenerate_flag(tmp_path) -> None:
    store_path = tmp_path / "cells.jsonl"
    baseline_mod.run_baseline(
        tmp_path, _dataset(), _config(), "windows-pc", store_path, tmp_path / "manifest.json"
    )
    df = baseline_mod.score_baseline(_dataset(), ForecastStore(store_path), _config())

    assert len(df) == 6
    for col in (
        "raw_crps", "raw_mae", "raw_rmse",
        "denom_a1", "denom_a2", "denom_a3",
        "scaled_crps_a1", "scaled_crps_a2", "scaled_crps_a3",
        "scaled_crps_a1_capped", "scaled_mae_a1", "scaled_rmse_a1",
    ):
        assert col in df.columns

    degenerate = df[df["denom_a1_degenerate"]]
    assert set(degenerate["benchmark_id"]) == {"task_4"}
    assert degenerate["scaled_crps_a1"].isna().all()

    healthy = df[df["benchmark_id"] == "task_1"]
    assert not healthy["denom_a1_degenerate"].any()
    assert (healthy["raw_crps"] >= 0).all()


def test_score_baseline_scaled_score_is_raw_over_denominator(tmp_path) -> None:
    store_path = tmp_path / "cells.jsonl"
    baseline_mod.run_baseline(
        tmp_path, _dataset(), _config(), "windows-pc", store_path, tmp_path / "manifest.json"
    )
    df = baseline_mod.score_baseline(_dataset(), ForecastStore(store_path), _config())
    row = df[(df["benchmark_id"] == "task_1") & (df["forecaster_name"] == "last_value_naive")].iloc[0]
    expected = min(row["raw_crps"] / row["denom_a3"], 5.0)
    assert row["scaled_crps_a3"] == expected


def test_rank_agreement_matches_known_spearman() -> None:
    df = pd.DataFrame(
        {
            "scaled_crps_a1": [1.0, 2.0, 3.0, 4.0],
            "scaled_crps_a2": [10.0, 20.0, 30.0, 40.0],  # same order -> +1
            "scaled_crps_a3": [4.0, 3.0, 2.0, 1.0],  # reversed -> -1
        }
    )
    agreement = baseline_mod._rank_agreement(df)
    assert agreement["a1_vs_a2"] == 1.0
    assert agreement["a1_vs_a3"] == -1.0


def test_write_scores_and_report_roundtrip(tmp_path) -> None:
    store_path = tmp_path / "cells.jsonl"
    baseline_mod.run_baseline(
        tmp_path, _dataset(), _config(), "windows-pc", store_path, tmp_path / "manifest.json"
    )
    df = baseline_mod.score_baseline(_dataset(), ForecastStore(store_path), _config())

    scores_path = tmp_path / "scores.parquet"
    baseline_mod.write_baseline_scores(df, scores_path)
    reloaded = pd.read_parquet(scores_path)
    assert len(reloaded) == len(df)

    report_path = tmp_path / "report.md"
    baseline_mod.write_baseline_report(reloaded, report_path)
    text = report_path.read_text(encoding="utf-8")
    assert "Score distributions per scaling" in text
    assert "winsorisation cap" in text
    assert "degenerate denominators" in text
    assert "Rank agreement" in text
    assert "Horizon dependence" in text
    assert "Seasonal-naive" in text
    assert "task_4" in text  # the constant-future task is listed as degenerate for a1


def test_paper_comparison_section_is_computed_from_scores(tmp_path) -> None:
    store_path = tmp_path / "cells.jsonl"
    baseline_mod.run_baseline(
        tmp_path, _dataset(), _config(), "windows-pc", store_path, tmp_path / "manifest.json"
    )
    df = baseline_mod.score_baseline(_dataset(), ForecastStore(store_path), _config())

    ref = {
        "source": "test source",
        "task_set": "test set",
        "scaled_mae": (0.8, 1.0),
        "scaled_rmse": (0.9, 1.0),
        "scaled_crps": (0.5, 0.7),
    }
    lines = baseline_mod._paper_comparison_lines(df, ref)
    text = "\n".join(lines)
    assert "paper, Naive (no context)" in text
    for key in ("a1", "a2", "a3"):
        assert f"ours, last_value_naive, {key}" in text

    # the CRPS-vs-paper ratio in the a3 row must equal our computed mean / the paper value
    naive = df[df["forecaster_name"] == "last_value_naive"]
    expected_ratio = naive["scaled_crps_a3"].mean() / 0.5
    a3_row = next(line for line in lines if "ours, last_value_naive, a3" in line)
    assert f"{expected_ratio:.2f}x" in a3_row

    # no reference -> no section
    out = tmp_path / "r.md"
    baseline_mod.write_baseline_report(df, out)
    assert "paper comparison" not in out.read_text(encoding="utf-8").lower()


def test_score_baseline_records_season_length_and_fallback(tmp_path) -> None:
    store_path = tmp_path / "cells.jsonl"
    baseline_mod.run_baseline(
        tmp_path, _dataset(), _config(), "windows-pc", store_path, tmp_path / "manifest.json"
    )
    df = baseline_mod.score_baseline(_dataset(), ForecastStore(store_path), _config())

    seasonal = df[df["forecaster_name"] == "seasonal_naive"].set_index("benchmark_id")
    assert seasonal.loc["task_1", "season_length_steps"] == 4  # int seasonal_period=4 is usable
    assert not seasonal.loc["task_1", "fell_back_to_naive"]
    assert seasonal.loc["task_2", "season_length_steps"] == 1  # seasonal_period=None
    assert seasonal.loc["task_2", "fell_back_to_naive"]

    naive = df[df["forecaster_name"] == "last_value_naive"]
    assert not naive["fell_back_to_naive"].any()
