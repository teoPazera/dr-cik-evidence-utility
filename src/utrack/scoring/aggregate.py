"""Winsorisation and cross-task aggregation of scaled scores (plan_a.md U0.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINSOR_CAP = 5.0


@dataclass(frozen=True)
class WinsorisedScore:
    raw: float
    winsorised: float
    capped: bool


def winsorise(score: float, cap: float = WINSOR_CAP) -> WinsorisedScore:
    """Cap a per-task scaled score at `cap` (5.0 by default, plan_a.md U0.3), flagging
    whether the cap was applied. Capped scores hide real differences (Decision B), so
    the flag must be carried through to any downstream comparison."""
    capped = score > cap
    return WinsorisedScore(raw=score, winsorised=min(score, cap) if capped else score, capped=capped)


@dataclass(frozen=True)
class Aggregate:
    mean: float
    stderr: float
    n: int


def aggregate_over_tasks(scores: list[float]) -> Aggregate:
    """Mean and standard error over tasks (plan_a.md U0.3). `stderr` is nan for n<2,
    since a standard error is not defined for a single observation."""
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return Aggregate(mean=float("nan"), stderr=float("nan"), n=0)
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return Aggregate(mean=mean, stderr=stderr, n=n)
