"""Unit test 5 from plan_a.md U0.3: multiplying series and samples by a constant
leaves each scaled score unchanged."""

import numpy as np
import pytest

from utrack.scoring.crps import mean_crps
from utrack.scoring.scaling import (
    scale_a1_future_range,
    scale_a2_seasonal_naive_mae,
    scale_a3_mean_abs_history,
)


def _scenario():
    rng = np.random.default_rng(5)
    future = rng.normal(50, 5, size=12)
    samples = future[None, :] + rng.normal(0, 3, size=(30, 12))
    history = rng.normal(50, 5, size=40)
    return future, samples, history


@pytest.mark.parametrize("k", [3.7, 0.25, 1000.0])
def test_scaled_score_invariant_to_positive_constant_a1(k: float) -> None:
    future, samples, _ = _scenario()

    raw = mean_crps(future, samples)
    scaled = raw / scale_a1_future_range(future)

    raw_k = mean_crps(future * k, samples * k)
    scaled_k = raw_k / scale_a1_future_range(future * k)

    assert scaled_k == pytest.approx(scaled, rel=1e-9)


@pytest.mark.parametrize("k", [3.7, 0.25, 1000.0])
def test_scaled_score_invariant_to_positive_constant_a3(k: float) -> None:
    future, samples, history = _scenario()

    raw = mean_crps(future, samples)
    scaled = raw / scale_a3_mean_abs_history(history)

    raw_k = mean_crps(future * k, samples * k)
    scaled_k = raw_k / scale_a3_mean_abs_history(history * k)

    assert scaled_k == pytest.approx(scaled, rel=1e-9)


@pytest.mark.parametrize("k", [3.7, 0.25, 1000.0])
def test_scaled_score_invariant_to_positive_constant_a2(k: float) -> None:
    future, samples, history = _scenario()

    raw = mean_crps(future, samples)
    scaled = raw / scale_a2_seasonal_naive_mae(history, seasonal_period_steps=4)

    raw_k = mean_crps(future * k, samples * k)
    scaled_k = raw_k / scale_a2_seasonal_naive_mae(history * k, seasonal_period_steps=4)

    assert scaled_k == pytest.approx(scaled, rel=1e-9)
