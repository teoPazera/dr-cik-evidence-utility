"""Last-value naive forecaster (plan_a.md U0.4): X_{T+h} = X_T + sum of h bootstrapped
one-step residuals. The m=1 case of the shared lag-m bootstrap (base.py)."""

from __future__ import annotations

import numpy as np

from utrack.data.schema import ForecastInput
from utrack.forecasters.base import ForecasterOutput, fill_history_forward, simulate_lag_m_bootstrap_paths


class LastValueNaiveForecaster:
    name = "last_value_naive"
    version = "1.0.0"

    def forecast(
        self,
        forecast_input: ForecastInput,
        context: str | None,
        n_samples: int,
        seed: int,
    ) -> ForecasterOutput:
        history = fill_history_forward(forecast_input.history_values, forecast_input.benchmark_id)
        rng = np.random.default_rng(seed)
        samples = simulate_lag_m_bootstrap_paths(
            history, m=1, horizon=forecast_input.prediction_length, n_samples=n_samples, rng=rng
        )
        return ForecasterOutput(
            samples=samples,
            forecaster_name=self.name,
            forecaster_version=self.version,
            model_identifier=None,
            n_requested=n_samples,
            n_valid=n_samples,
            seed=seed,
            sampling_params={"method": "lag_1_residual_bootstrap"},
        )
