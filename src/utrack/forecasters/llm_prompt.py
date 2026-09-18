"""Approved CiK Direct Prompt template and request builder for U1.

Decision D was resolved on 2026-09-18: this is D1, a port of CiK's Direct
Prompt with a slot for the rendered condition context. The final rendered
requests are written for Teo's review before every paid execution; any prompt
change must receive a new `TEMPLATE_VERSION`.

Ported from `cik_benchmark/baselines/direct_prompt.py` (`make_prompt` and the system message in
`__call__`), Apache-2.0, ServiceNow, commit 73f46016f8c5643bf6799ca31875eab3e8d0d075 of
ServiceNow/context-is-key-forecasting. Differences from the original:
  - the context slot receives our rendered condition context (plan_a.md 3.3) instead of CiK's
    Background/Constraints/Scenario fields; for C0 the slot is empty, exactly as in CiK's
    no-context runs, so the only thing that changes between conditions is the slot;
  - history values that are missing are forward/back-filled (as the statistical forecasters do).

Nothing here sends a request. `build_requests` only assembles them, so the cost estimate (U0.6) can
count tokens. This module must never touch a label: it takes a `ForecastInput` and a `Condition`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utrack.conditions.base import Condition
from utrack.data.loader import fill_history_forward
from utrack.data.schema import ForecastInput

TEMPLATE_VERSION = "d1-cik-approved-v1"
SYSTEM_MESSAGE = "You are a useful forecasting assistant."
MAX_DIGITS = 6

_TEMPLATE = """
I have a time series forecasting task for you.

Here is some context about the task. Make sure to factor in any background knowledge,
satisfy any constraints, and respect any scenarios.
<context>
{context}
</context>

Here is a historical time series in (timestamp, value) format:
<history>
{history}
</history>

Now please predict the value at the following timestamps: {pred_time}.

Return the forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments) in the forecast.

Example:
<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>

"""


def format_history(forecast_input: ForecastInput) -> str:
    """CiK's history format: `(timestamp, value)`, 6 significant digits, no scientific notation."""
    values = fill_history_forward(forecast_input.history_values, forecast_input.benchmark_id)
    return "\n".join(
        f"({ts}, {y:.{MAX_DIGITS}g})" if y < 10**MAX_DIGITS else f"({ts}, {y:.0f})"
        for ts, y in zip(forecast_input.history_timestamps, values)
    )


def render_user_prompt(forecast_input: ForecastInput, context: str | None) -> str:
    return _TEMPLATE.format(
        context=context or "",
        history=format_history(forecast_input),
        # CiK interpolates a numpy array of timestamp strings; kept so token counts match the original
        pred_time=str(np.array(forecast_input.future_timestamps)),
    )


@dataclass(frozen=True)
class LLMRequest:
    """One request as it would be sent: a prompt asking for `n_samples` forecasts (via `n > 1`)."""

    benchmark_id: str
    condition_id: str
    repeat: int
    system_message: str
    user_prompt: str
    n_samples: int
    template_version: str
    condition_seed: int | None


def build_request(
    forecast_input: ForecastInput, condition: Condition, repeat: int, n_samples: int
) -> LLMRequest:
    return LLMRequest(
        benchmark_id=forecast_input.benchmark_id,
        condition_id=condition.condition_id,
        repeat=repeat,
        system_message=SYSTEM_MESSAGE,
        user_prompt=render_user_prompt(forecast_input, condition.context),
        n_samples=n_samples,
        template_version=TEMPLATE_VERSION,
        condition_seed=condition.seed,
    )


def expected_output_text(forecast_input: ForecastInput) -> str:
    """What one returned sample looks like in the requested format, with a typical value, so its
    length can be counted. Uses only the future timestamps, which the prompt already contains."""
    values = fill_history_forward(forecast_input.history_values, forecast_input.benchmark_id)
    typical = format(float(np.median(values)), f".{MAX_DIGITS}g")
    lines = [f"({ts}, {typical})" for ts in forecast_input.future_timestamps]
    return "<forecast>\n" + "\n".join(lines) + "\n</forecast>"
