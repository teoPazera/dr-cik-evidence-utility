"""Seasonal-naive forecaster (plan_a.md U0.4): X_{T+h} = X_{T+h-m} + bootstrapped
seasonal residual, m resolved from `seasonal_period`/`frequency` (base.py). Falls back
to m=1 (identical to LastValueNaiveForecaster) when no usable season is found, logging
why: this is common, not an error - see base.py:resolve_seasonal_period_steps."""

from __future__ import annotations

import logging

import numpy as np

from utrack.data.schema import ForecastInput
from utrack.forecasters.base import (
    ForecasterOutput,
    fill_history_forward,
    resolve_seasonal_period_steps,
    simulate_lag_m_bootstrap_paths,
)

logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster:
    name = "seasonal_naive"
    version = "1.0.0"

    def forecast(
        self,
        forecast_input: ForecastInput,
        context: str | None,
        n_samples: int,
        seed: int,
    ) -> ForecasterOutput:
        history = fill_history_forward(forecast_input.history_values, forecast_input.benchmark_id)
        notes: list[str] = []

        m = resolve_seasonal_period_steps(
            forecast_input.seasonal_period, forecast_input.frequency, forecast_input.benchmark_id
        )
        if m is None:
            notes.append("no usable seasonal_period; fell back to m=1 (same as last_value_naive)")
            m = 1
        elif len(history) <= m:
            logger.warning(
                "seasonal_naive: history (%d points) too short for season length %d on %s; falling back to m=1",
                len(history),
                m,
                forecast_input.benchmark_id,
            )
            notes.append(f"history too short for resolved season length {m}; fell back to m=1")
            m = 1

        rng = np.random.default_rng(seed)
        samples = simulate_lag_m_bootstrap_paths(
            history, m=m, horizon=forecast_input.prediction_length, n_samples=n_samples, rng=rng
        )
        return ForecasterOutput(
            samples=samples,
            forecaster_name=self.name,
            forecaster_version=self.version,
            model_identifier=None,
            n_requested=n_samples,
            n_valid=n_samples,
            seed=seed,
            sampling_params={"method": "lag_m_residual_bootstrap", "resolved_season_length_steps": m},
            notes=notes,
        )
