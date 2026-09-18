"""Scaling denominators for the headline scaled CRPS (Decision A, plan_a.md U0.3).

All three options are computed and stored; Teo picks the headline after the
U0 report. Each is explicit degree-1-homogeneous in its input series, so a
scaled score is unchanged if history/future/samples are all multiplied by the
same positive constant (plan_a.md U0.3 unit test 5).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

ZERO_DENOMINATOR_EPS = 1e-9


def scale_a1_future_range(future_values: np.ndarray, benchmark_id: str = "") -> float:
    """A1: divide by (max - min) of the task's future values (the CiK convention,
    `inverse_mean_forecast_range`). Only computable for dev tasks (needs future_values);
    the hidden test set withholds them."""
    denom = float(np.max(future_values) - np.min(future_values))
    if denom < ZERO_DENOMINATOR_EPS:
        logger.warning(
            "A1 scaling: near-zero future range for %s (%.3g); scaled score will be unstable",
            benchmark_id or "<unknown task>",
            denom,
        )
    return denom


def scale_a2_seasonal_naive_mae(
    history_values: np.ndarray, seasonal_period_steps: int, benchmark_id: str = ""
) -> float:
    """A2: divide by the in-sample mean absolute seasonal-naive error (MASE-style, history only).

    `seasonal_period_steps` is the seasonal lag measured in time steps, already
    resolved from the task's `frequency`/`seasonal_period` fields by the caller
    (that resolution is shared with the seasonal-naive forecaster, U0.4, so it
    is not duplicated here to keep this function pure and independently testable).
    """
    m = max(1, int(seasonal_period_steps))
    history_values = np.asarray(history_values, dtype=float)
    if len(history_values) <= m:
        logger.warning(
            "A2 scaling: history too short (%d points) for season length %d on %s; falling back to lag-1",
            len(history_values),
            m,
            benchmark_id or "<unknown task>",
        )
        m = 1
    if len(history_values) <= m:
        logger.warning(
            "A2 scaling: history has only %d point(s), cannot compute even a lag-1 seasonal-naive "
            "error for %s; returning nan",
            len(history_values),
            benchmark_id or "<unknown task>",
        )
        return float("nan")
    errors = np.abs(history_values[m:] - history_values[:-m])
    denom = float(np.mean(errors))
    if denom < ZERO_DENOMINATOR_EPS:
        logger.warning(
            "A2 scaling: near-zero seasonal-naive MAE for %s (%.3g)", benchmark_id or "<unknown task>", denom
        )
    return denom


def scale_a3_mean_abs_history(history_values: np.ndarray, benchmark_id: str = "") -> float:
    """A3: divide by the mean absolute history value."""
    denom = float(np.mean(np.abs(history_values)))
    if denom < ZERO_DENOMINATOR_EPS:
        logger.warning(
            "A3 scaling: near-zero mean abs history for %s (%.3g)", benchmark_id or "<unknown task>", denom
        )
    return denom
