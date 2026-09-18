import numpy as np

from utrack.data.schema import ForecastInput
from utrack.forecasters.naive import LastValueNaiveForecaster
from utrack.forecasters.seasonal_naive import SeasonalNaiveForecaster


def _make_forecast_input(**overrides) -> ForecastInput:
    defaults = dict(
        benchmark_id="task_x",
        origin="synthetic",
        frequency="1 hour",
        prediction_length=10,
        seasonal_period=None,
        history_timestamps=[f"2026-01-01 {h:02d}:00:00" for h in range(24)],
        history_values=list(np.sin(np.arange(24) / 3.0) * 10 + 50),
        future_timestamps=[f"2026-01-02 {h:02d}:00:00" for h in range(10)],
        document_ids=[],
        entity_name="e",
        entity_type="t",
        profile_id="1",
        profile_name="p",
        profile_details={},
        time_series_variable="v",
        target_description="d",
        reasoning_hops=1,
    )
    defaults.update(overrides)
    return ForecastInput(**defaults)


def test_naive_forecaster_output_shape_and_metadata() -> None:
    fi = _make_forecast_input()
    forecaster = LastValueNaiveForecaster()
    out = forecaster.forecast(fi, context=None, n_samples=100, seed=42)
    assert out.samples.shape == (100, 10)
    assert out.n_requested == 100
    assert out.n_valid == 100
    assert out.forecaster_name == "last_value_naive"
    assert out.seed == 42
    assert out.cost_usd == 0.0
    assert out.token_counts == {"input": 0, "output": 0}


def test_naive_forecaster_deterministic_given_seed() -> None:
    fi = _make_forecast_input()
    forecaster = LastValueNaiveForecaster()
    out1 = forecaster.forecast(fi, context=None, n_samples=20, seed=7)
    out2 = forecaster.forecast(fi, context=None, n_samples=20, seed=7)
    np.testing.assert_array_equal(out1.samples, out2.samples)


def test_seasonal_naive_falls_back_to_m1_when_no_season(caplog) -> None:
    fi = _make_forecast_input(seasonal_period=None)
    forecaster = SeasonalNaiveForecaster()
    out = forecaster.forecast(fi, context=None, n_samples=5, seed=1)
    assert out.sampling_params["resolved_season_length_steps"] == 1
    assert any("fell back to m=1" in n for n in out.notes)


def test_seasonal_naive_uses_resolved_season_length() -> None:
    fi = _make_forecast_input(seasonal_period=6, frequency="1 hour")  # season = 6 steps
    forecaster = SeasonalNaiveForecaster()
    out = forecaster.forecast(fi, context=None, n_samples=5, seed=1)
    assert out.sampling_params["resolved_season_length_steps"] == 6
    assert out.notes == []
    assert out.samples.shape == (5, 10)


def test_seasonal_naive_falls_back_when_history_too_short_for_season() -> None:
    fi = _make_forecast_input(
        seasonal_period=20,
        history_timestamps=["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00"],
        history_values=[1.0, 2.0, 3.0],
    )
    forecaster = SeasonalNaiveForecaster()
    out = forecaster.forecast(fi, context=None, n_samples=5, seed=1)
    assert out.sampling_params["resolved_season_length_steps"] == 1
    assert any("too short" in n for n in out.notes)
