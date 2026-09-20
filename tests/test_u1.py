from __future__ import annotations

import numpy as np

from utrack.data.loader import Dataset
from utrack.data.schema import Task
from utrack.forecasters.base import ForecasterOutput
from utrack.reports import u1 as u1_mod
from utrack.store.forecast_store import ForecastStore


def _task(n: int) -> Task:
    raw = {
        "benchmark_id": f"task_{n}", "split": "dev", "origin": "synthetic", "labels_public": True, "reasoning_hops": 1,
        "entity_name": f"entity {n}", "entity_type": "t", "profile_id": "1", "profile_name": "p", "profile_details": {}, "time_series_variable": f"v{n}",
        "frequency": "1 hour", "prediction_length": 2, "seasonal_period": None, "target_description": "d",
        "history_timestamps": ["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00"], "history_values": [1., 2., 3.],
        "future_timestamps": ["2026-01-01 03:00:00", "2026-01-01 04:00:00"], "future_values": [4., 5.], "document_ids": [],
        "gt_evidence": [{"id": "E1", "evidence": f"evidence {n}"}], "raw_task_path": "x",
    }
    return Task.from_raw(raw)


def _dataset() -> Dataset:
    tasks = {f"task_{n}": _task(n) for n in (1, 2, 3)}
    return Dataset(tasks=tasks, documents_by_task={k: [] for k in tasks})


def _configs() -> tuple[dict, dict]:
    u0 = {"dataset": {"revision": "r"}, "conditions": {"seed": 1}, "baseline": {"scaling_a2_fallback_lag_steps": 1}}
    u1 = {"model": "m", "sampling": {"temperature": 1.0, "samples_per_cell": 2}, "n_retries": 1, "cost_cap_usd": 15.0,
          "pricing_usd_per_million_tokens": {"input": .25, "cached_input": .025, "output": 1.5}, "max_output_tokens": 8}
    return u0, u1


def test_run_u1_resumes_and_writes_live_cost_ledger(tmp_path, monkeypatch) -> None:
    class FakeForecaster:
        name = "fake"
        def __init__(self, *, ledger, **kwargs): self.ledger = ledger
        def forecast(self, forecast_input, context, n_samples, seed, progress_callback=None):
            self.ledger.record(.25)
            if progress_callback: progress_callback({"event": "attempt_accepted", "benchmark_id": forecast_input.benchmark_id, "requested_samples": n_samples, "valid_samples": n_samples, "attempts": 1, "max_attempts": 3, "spent_usd": self.ledger.spent_usd, "cost_cap_usd": self.ledger.cap_usd})
            return ForecasterOutput(samples=np.tile([[4., 5.]], (n_samples, 1)), forecaster_name="fake", forecaster_version="v", model_identifier="m", n_requested=n_samples, n_valid=n_samples, seed=seed, token_counts={"input": 10, "output": 2, "total": 12, "cached_input": 8}, cost_usd=.25, sampling_params={"attempts": 1, "request_costs": [{"cost_usd": .25}]})
    monkeypatch.setattr(u1_mod, "LiteLLMDirectForecaster", FakeForecaster)
    u0, u1 = _configs(); store = tmp_path / "cells.jsonl"; ledger = tmp_path / "ledger.md"; events = []
    first = u1_mod.run_u1(tmp_path, _dataset(), u0, u1, "test", store, tmp_path / "manifest.json", ledger, ["task_1", "task_2"], ["C0"], [0], progress_callback=events.append)
    assert first.n_cells == 2
    assert "Total recorded spend: **$0.500000**" in ledger.read_text()
    second = u1_mod.run_u1(tmp_path, _dataset(), u0, u1, "test", store, tmp_path / "manifest2.json", ledger, ["task_1", "task_2"], ["C0"], [0], progress_callback=events.append)
    assert second.n_cells == 0
    assert len(ForecastStore(store).read_all()) == 2
    assert any(e["event"] == "cell_skipped" for e in events)


def test_score_u1_uses_a3_b1_and_review_writes_plot(tmp_path) -> None:
    ds = _dataset(); u0, _ = _configs(); store = ForecastStore(tmp_path / "cells.jsonl")
    for condition, samples in [("C0", [[8., 8.], [8., 8.]]), ("C1", [[4., 5.], [4., 5.]])]:
        store.append({"benchmark_id": "task_1", "condition_id": condition, "repeat": 0, "forecaster_name": "fake", "n_requested": 2, "n_valid": 2, "cost_usd": .1, "token_counts": {"input": 10, "output": 2, "cached_input": 0}, "samples": samples})
    scores = u1_mod.score_u1(ds, u0, store.path)
    c1 = scores[scores.condition_id == "C1"].iloc[0]
    assert c1.utility_b1_a3 > 0
    report = tmp_path / "report.md"; figures = tmp_path / "figures"
    u1_mod.write_u1_review(ds, scores, store.path, figures, report)
    assert report.exists() and (figures / "task_1_repeat0.png").exists()
