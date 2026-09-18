import numpy as np
import pandas as pd

from utrack.data.schema import ForecastInput, Task
from utrack.data.loader import Dataset
from utrack.reports.u1_smoke import score_smoke, write_smoke_report
from utrack.store.forecast_store import ForecastStore


def _dataset() -> Dataset:
    raw = {
        "benchmark_id": "task_42", "split": "dev", "origin": "synthetic", "labels_public": True, "reasoning_hops": 1,
        "entity_name": "e", "entity_type": "t", "profile_id": "1", "profile_name": "p", "profile_details": {}, "time_series_variable": "v",
        "frequency": "1 hour", "prediction_length": 2, "seasonal_period": None, "target_description": "d",
        "history_timestamps": ["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00"], "history_values": [1.,2.,3.],
        "future_timestamps": ["2026-01-01 03:00:00", "2026-01-01 04:00:00"], "future_values": [4.,5.],
        "document_ids": [], "gt_evidence": [{"id":"e1","evidence":"x"}], "raw_task_path":"x"
    }
    return Dataset(tasks={"task_42": Task.from_raw(raw)}, documents_by_task={})


def test_score_smoke_includes_cache_cost_and_crps(tmp_path) -> None:
    store = ForecastStore(tmp_path / "cells.jsonl")
    store.append({
        "benchmark_id":"task_42", "condition_id":"C0", "forecaster_name":"llm_direct", "n_requested":2, "n_valid":2,
        "cost_usd":0.01, "token_counts":{"input":100,"output":20,"total":120,"cached_input":80},
        "sampling_params":{"request_costs":[{"cache_read_cost_usd":0.003,"input_cost_usd":0.002,"output_cost_usd":0.005}]},
        "samples":[[4.,5.],[3.,6.]],
    })
    cfg={"baseline":{"scaling_a2_fallback_lag_steps":1}}
    df=score_smoke(tmp_path,_dataset(),cfg,store.path)
    assert len(df)==1
    assert df.loc[0,"cached_input_fraction"]==0.8
    assert df.loc[0,"cached_input_tokens"]==80
    assert np.isfinite(df.loc[0,"raw_crps"])


def test_write_smoke_report_mentions_proxy_cost_and_cache(tmp_path) -> None:
    df=pd.DataFrame([{"condition_id":"C0","n_valid":2,"valid_rate":1.0,"raw_crps":1.0,"scaled_crps_a1":.5,"scaled_crps_a2":.5,"scaled_crps_a3":.5,"cost_usd":.01,"input_tokens":100,"cached_input_tokens":80,"cached_input_fraction":.8,"output_tokens":20,"uncached_input_tokens":20,"cache_read_cost_usd":.003}])
    path=tmp_path/"report.md"
    write_smoke_report(df,tmp_path/"missing.parquet",path)
    text=path.read_text()
    assert "proxy-reported total cost" in text
    assert "80/100" in text
