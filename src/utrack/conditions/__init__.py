"""Condition builder (plan_a.md 3.1, U0.5): turns a task into the context a forecaster is given. No model calls."""

from utrack.conditions.base import CONDITION_NAMES, RENDER_VERSION, Condition
from utrack.conditions.builders import build_condition

__all__ = ["CONDITION_NAMES", "RENDER_VERSION", "Condition", "build_condition"]
