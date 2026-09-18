"""Utility of context (plan_a.md section 2, Decision B).

    utility(context) = score(forecast without context) - score(forecast with context)

The score is a scaled CRPS where lower is better, so positive utility means
the context helped, negative means it hurt, zero means no effect. Both forms
in Decision B are computed; B1 is the headline.
"""

from __future__ import annotations

import math


def utility_b1_absolute(score_without_context: float, score_with_context: float) -> float:
    """B1 (headline): absolute difference of winsorised scaled CRPS."""
    return score_without_context - score_with_context


def utility_b2_relative(score_without_context: float, score_with_context: float) -> float:
    """B2: relative utility, 1 - score_with_context / score_without_context.

    nan when `score_without_context` is ~0: the ratio is degenerate there,
    and a near-zero reference score is already flagged separately (scaling
    near-zero-denominator warnings, plan_a.md U0.3).
    """
    if abs(score_without_context) < 1e-12:
        return math.nan
    return 1.0 - (score_with_context / score_without_context)
