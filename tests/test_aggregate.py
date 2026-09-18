"""Unit test 6 from plan_a.md U0.3, plus aggregate_over_tasks coverage."""

import math

import pytest

from utrack.scoring.aggregate import WINSOR_CAP, aggregate_over_tasks, winsorise


def test_winsorise_below_cap_is_unchanged() -> None:
    result = winsorise(2.0)
    assert result.raw == 2.0
    assert result.winsorised == 2.0
    assert result.capped is False


def test_winsorise_at_cap_boundary_is_not_flagged() -> None:
    result = winsorise(WINSOR_CAP)
    assert result.winsorised == WINSOR_CAP
    assert result.capped is False


def test_winsorise_above_cap_is_capped_and_flagged() -> None:
    result = winsorise(7.3)
    assert result.raw == 7.3
    assert result.winsorised == WINSOR_CAP
    assert result.capped is True


def test_aggregate_over_tasks_mean_and_stderr() -> None:
    agg = aggregate_over_tasks([1.0, 2.0, 3.0])
    assert agg.n == 3
    assert agg.mean == pytest.approx(2.0)
    assert agg.stderr == pytest.approx((1.0) / math.sqrt(3))  # sample std (ddof=1) = 1.0


def test_aggregate_over_tasks_single_value_has_nan_stderr() -> None:
    agg = aggregate_over_tasks([5.0])
    assert agg.n == 1
    assert agg.mean == pytest.approx(5.0)
    assert math.isnan(agg.stderr)


def test_aggregate_over_tasks_empty_is_nan() -> None:
    agg = aggregate_over_tasks([])
    assert agg.n == 0
    assert math.isnan(agg.mean)
    assert math.isnan(agg.stderr)
