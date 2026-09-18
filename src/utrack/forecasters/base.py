"""Forecaster interface and shared building blocks (plan_a.md 3.1, 3.2, U0.4).

One interface for statistical and LLM forecasters (U1's `llm_direct` uses the
same `ForecasterOutput` shape). A forecaster never receives `TaskLabels`
(plan_a.md 3.2 rule 3, label isolation): `Forecaster.forecast` takes a
`ForecastInput` and a rendered context string, nothing else task-specific.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from utrack.data.loader import fill_history_forward  # re-exported for forecaster call sites
from utrack.data.schema import ForecastInput

__all__ = [
    "ForecasterOutput",
    "Forecaster",
    "fill_history_forward",
    "resolve_seasonal_period_steps",
    "simulate_lag_m_bootstrap_paths",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecasterOutput:
    """plan_a.md 3.2 rule 4: every cell stores forecaster name/version, model
    identifier, sampling parameters, timestamp, token counts, cost, and the
    number of requested vs. valid samples. `samples` holds only the valid ones."""

    samples: np.ndarray  # shape (n_valid, horizon)
    forecaster_name: str
    forecaster_version: str
    model_identifier: str | None
    n_requested: int
    n_valid: int
    seed: int
    sampling_params: dict = field(default_factory=dict)
    token_counts: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    cost_usd: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: list[str] = field(default_factory=list)


class Forecaster(Protocol):
    name: str
    version: str

    def forecast(
        self,
        forecast_input: ForecastInput,
        context: str | None,
        n_samples: int,
        seed: int,
    ) -> ForecasterOutput: ...


_DEPRECATED_ALIAS_RE = re.compile(r"^(\d*)([A-Za-z]+)$")
_DEPRECATED_ALIAS_MAP = {"H": "h", "T": "min", "S": "s"}


def _normalize_pandas_alias(alias: str) -> str:
    m = _DEPRECATED_ALIAS_RE.match(alias)
    if not m:
        return alias
    num, unit = m.group(1), m.group(2)
    return f"{num}{_DEPRECATED_ALIAS_MAP.get(unit, unit)}"


def resolve_seasonal_period_steps(
    seasonal_period: str | int | None, frequency: str, benchmark_id: str = ""
) -> int | None:
    """Resolve `seasonal_period` + `frequency` into a season length in time steps, or
    None if there is no usable multi-step season.

    U0.4 found (artifacts/u0/baseline_report.md) that `seasonal_period` is
    inconsistently typed (U0.2) in a way that matters here: when it is an int, it is
    already a step count (e.g. 24 for hourly data with daily seasonality). When it is a
    string, it is a pandas frequency alias - but in every task observed, that alias's
    duration equals `frequency`'s own duration (ratio ~1.0), i.e. it re-encodes the
    sampling frequency, not a distinct season. Resolving it honestly (rather than
    assuming one behaviour) makes both cases fall out of the same formula: a string
    `seasonal_period` will legitimately resolve to a usable season if the data ever
    contains one where the ratio isn't ~1.
    """
    if seasonal_period is None:
        return None
    if isinstance(seasonal_period, int):
        return seasonal_period if seasonal_period >= 2 else None
    if not isinstance(seasonal_period, str):
        return None

    freq_delta = pd.Timedelta(frequency)
    if freq_delta <= pd.Timedelta(0):
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(_normalize_pandas_alias(seasonal_period))
    except ValueError:
        logger.warning(
            "could not parse seasonal_period=%r for %s", seasonal_period, benchmark_id or "<unknown task>"
        )
        return None

    # Apply the (possibly calendar-anchored) offset twice and diff the results, rather
    # than diffing once against a fixed anchor: for offsets like "W" (next Sunday) or
    # "MS" (next month start), a single application rolls to the nearest anchor point
    # first, which is not the offset's true periodic length.
    anchor = pd.Timestamp("2020-01-01")
    t1 = anchor + offset
    t2 = t1 + offset
    season_delta = t2 - t1
    steps = round(season_delta / freq_delta)
    return steps if steps >= 2 else None


def simulate_lag_m_bootstrap_paths(
    history: np.ndarray, m: int, horizon: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate `n_samples` future paths of length `horizon` from the lag-m random-walk
    model X_t = X_{t-m} + e_t, where e_t is drawn with replacement from the in-sample
    lag-m residuals. This is the standard bootstrap simulation for naive (m=1) and
    seasonal-naive (m=season length) prediction intervals (Hyndman & Athanasopoulos,
    "Forecasting: Principles and Practice", chapter on prediction intervals; the same
    R `forecast` package convention). Once a horizon step is simulated it becomes
    available as X_{t-m} for later steps within the same path, so uncertainty grows in
    steps of size m rather than every single step - this is what makes m=1 (plain
    naive) and m>1 (seasonal naive) genuinely different processes, not just different
    point forecasts.

    Vectorised by "phase" (h mod m): the m interleaved subsequences that make up the
    recursion are each an ordinary cumulative-sum random walk, so no per-step Python
    loop is needed even though the recursion is inherently sequential.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    residuals = history[m:] - history[:-m]
    if len(residuals) == 0:
        raise ValueError(f"not enough history ({len(history)} points) for lag {m}")

    paths = np.empty((n_samples, horizon), dtype=float)
    for phase in range(m):
        h_values = np.arange(phase + 1, horizon + 1, m)  # 1-indexed horizon steps at this phase
        if len(h_values) == 0:
            continue
        seed_value = history[-(m - phase)]
        eps = rng.choice(residuals, size=(n_samples, len(h_values)))
        cum = np.cumsum(eps, axis=1)
        paths[:, h_values - 1] = seed_value + cum
    return paths
