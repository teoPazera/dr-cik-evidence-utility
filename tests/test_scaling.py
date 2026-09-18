import logging

import numpy as np
import pytest

from utrack.scoring.scaling import (
    scale_a1_future_range,
    scale_a2_seasonal_naive_mae,
    scale_a3_mean_abs_history,
)


def test_scale_a1_future_range() -> None:
    future = np.array([10.0, 30.0, 20.0])
    assert scale_a1_future_range(future) == pytest.approx(20.0)


def test_scale_a1_warns_on_near_zero_range(caplog) -> None:
    future = np.array([5.0, 5.0, 5.0])
    with caplog.at_level(logging.WARNING):
        denom = scale_a1_future_range(future, benchmark_id="task_x")
    assert denom == pytest.approx(0.0)
    assert any("near-zero" in r.message for r in caplog.records)


def test_scale_a2_seasonal_naive_mae() -> None:
    # period 2: errors = |v[2]-v[0]|, |v[3]-v[1]|, |v[4]-v[2]| = |5-1|,|4-2|,|3-5| = 4,2,2
    history = np.array([1.0, 2.0, 5.0, 4.0, 3.0])
    denom = scale_a2_seasonal_naive_mae(history, seasonal_period_steps=2)
    assert denom == pytest.approx((4 + 2 + 2) / 3)


def test_scale_a2_falls_back_to_lag_1_when_history_too_short(caplog) -> None:
    history = np.array([1.0, 3.0, 2.0])
    with caplog.at_level(logging.WARNING):
        denom = scale_a2_seasonal_naive_mae(history, seasonal_period_steps=10, benchmark_id="task_y")
    # lag-1 errors: |3-1|, |2-3| = 2, 1
    assert denom == pytest.approx((2 + 1) / 2)
    assert any("falling back to lag-1" in r.message for r in caplog.records)


def test_scale_a3_mean_abs_history() -> None:
    history = np.array([-2.0, 4.0, -6.0])
    assert scale_a3_mean_abs_history(history) == pytest.approx((2 + 4 + 6) / 3)


def test_scale_a3_warns_on_near_zero(caplog) -> None:
    history = np.array([0.0, 0.0])
    with caplog.at_level(logging.WARNING):
        scale_a3_mean_abs_history(history, benchmark_id="task_z")
    assert any("near-zero" in r.message for r in caplog.records)
