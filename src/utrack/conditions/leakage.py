"""Leakage checks for U0.5 (plan_a.md U0.5, 3.2 rule 3).

Two kinds of check:

1. Structural: `ForecastInput` has no label field, and the condition builders never reference
   `future_values` or `TaskLabels`.
2. Textual: does any text a forecaster would see contain a run of consecutive future values? A
   run is found as consecutive numbers in the text (words in between are ignored, timestamps are
   removed first), compared to the future values with a relative tolerance loose enough to catch
   the CiK prompt's 6-significant-digit formatting. A run only counts if it has at least `min_run`
   values and at least `MIN_DISTINCT` distinct values: flat stretches and runs of a few small
   integers match unrelated text by coincidence (a first version that asked for two distinct values
   flagged the runs 342, 342, 342, 342, 6 and 5, 0, 0, 0 in distractor documents).

Evidence and documents may legitimately contain numbers (plan_a.md U0.5), so for those surfaces the
scan reports. The history is reported too: a future that repeats a stretch of the history is not a
leak, because the forecaster is allowed to see the history. Only the rest of the forecaster input
(its metadata) is asserted clean.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
from dataclasses import dataclass

import numpy as np

from utrack.data.schema import ForecastInput, TaskLabels

DEFAULT_MIN_RUN = 4
MIN_DISTINCT = 3
REL_TOL = 1e-5  # loose enough for "{:.6g}" formatting (relative error up to 5e-6)

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# A sign is taken only when not glued to a preceding word character ("5-10" is 5 and 10, not 5 and -10).
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class LeakHit:
    run_length: int
    future_start: int  # index into the future values where the matched run begins


def extract_numbers(text: str) -> np.ndarray:
    """Absolute values of the numbers in `text`, in order, with timestamps removed."""
    stripped = _TIMESTAMP.sub(" | ", text)
    return np.abs(np.array([float(m) for m in _NUMBER.findall(stripped)], dtype=float))


def _longest_valid_run(numbers: np.ndarray, future: np.ndarray, min_run: int) -> LeakHit | None:
    if len(numbers) == 0 or len(future) == 0:
        return None
    match = np.isclose(numbers[:, None], future[None, :], rtol=REL_TOL, atol=1e-12)
    run = np.zeros(len(numbers), dtype=int)
    best: LeakHit | None = None
    for j in range(len(future)):
        new = np.zeros_like(run)
        new[0] = 1 if match[0, j] else 0
        new[1:] = np.where(match[1:, j], run[:-1] + 1, 0)
        run = new
        # run[i] = length of the run of consecutive future values ending at future[j] that matches
        # the numbers ending at numbers[i]; it qualifies if that window of the future is varied enough
        for length in np.unique(run[run >= min_run]):
            window = future[j - length + 1 : j + 1]
            if len(set(window.tolist())) >= MIN_DISTINCT and (best is None or length > best.run_length):
                best = LeakHit(run_length=int(length), future_start=j - int(length) + 1)
    return best


def find_future_run(text: str, future_values: list[float], min_run: int = DEFAULT_MIN_RUN) -> LeakHit | None:
    """Longest qualifying run of consecutive future values appearing in `text`, or None."""
    future = np.abs(np.asarray(future_values, dtype=float))
    hits = []
    for variant in (text, _THOUSANDS_COMMA.sub("", text)):
        hit = _longest_valid_run(extract_numbers(variant), future, min_run)
        if hit is not None:
            hits.append(hit)
    return max(hits, key=lambda h: h.run_length) if hits else None


def is_untestable(future_values: list[float], min_run: int = DEFAULT_MIN_RUN) -> bool:
    """True when no window of `min_run` future values holds `MIN_DISTINCT` distinct values, so the
    scan cannot tell a leak from coincidence for this task."""
    values = np.asarray(future_values, dtype=float)
    return not any(
        len(set(values[i : i + min_run].tolist())) >= MIN_DISTINCT for i in range(max(len(values) - min_run + 1, 0))
    )


def history_text(forecast_input: ForecastInput) -> str:
    """The history in the CiK prompt's `(timestamp, value)` format with `.6g` values."""
    return "\n".join(
        f"({ts}, {'nan' if v is None else format(v, '.6g')})"
        for ts, v in zip(forecast_input.history_timestamps, forecast_input.history_values)
    )


def metadata_text(forecast_input: ForecastInput) -> str:
    """Every `ForecastInput` field except the history, as JSON: everything else a template could show."""
    other = {
        k: v
        for k, v in dataclasses.asdict(forecast_input).items()
        if k not in ("history_timestamps", "history_values")
    }
    return json.dumps(other, sort_keys=True, default=str)


def label_fields_on_forecast_input() -> set[str]:
    """Fields of `ForecastInput` that also exist on `TaskLabels` (other than the id): must be empty."""
    label_fields = {f.name for f in dataclasses.fields(TaskLabels)} - {"benchmark_id"}
    return label_fields & {f.name for f in dataclasses.fields(ForecastInput)}


def label_references_in_source(module) -> list[str]:
    """References to `future_values`, `TaskLabels` or `.labels` in a module's source (docstrings excluded)."""
    tree = ast.parse(inspect.getsource(module))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("future_values", "labels"):
            found.append(f"line {node.lineno}: attribute .{node.attr}")
        if isinstance(node, ast.Name) and node.id in ("TaskLabels", "future_values"):
            found.append(f"line {node.lineno}: name {node.id}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if "TaskLabels" in names:
                found.append(f"line {node.lineno}: imports TaskLabels")
    return found
