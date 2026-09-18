"""Unit test 8 from plan_a.md U0.3: utility sign convention (section 2)."""

import math

import pytest

from utrack.scoring.utility import utility_b1_absolute, utility_b2_relative


def test_b1_positive_when_context_forecast_closer_to_truth() -> None:
    # lower score is better; context brought the score down from 2.0 to 0.5
    assert utility_b1_absolute(score_without_context=2.0, score_with_context=0.5) > 0


def test_b1_negative_when_context_forecast_hurts() -> None:
    assert utility_b1_absolute(score_without_context=0.5, score_with_context=2.0) < 0


def test_b1_zero_when_no_effect() -> None:
    assert utility_b1_absolute(score_without_context=1.2, score_with_context=1.2) == pytest.approx(0.0)


def test_b2_positive_when_context_forecast_closer_to_truth() -> None:
    assert utility_b2_relative(score_without_context=2.0, score_with_context=0.5) > 0


def test_b2_negative_when_context_forecast_hurts() -> None:
    assert utility_b2_relative(score_without_context=0.5, score_with_context=2.0) < 0


def test_b2_nan_when_reference_score_near_zero() -> None:
    assert math.isnan(utility_b2_relative(score_without_context=0.0, score_with_context=0.3))
