"""Deterministic seed derivation (plan_a.md 5.2.7).

Every seeded operation takes one seed from the config and derives a per-operation seed
from it and identifying strings. sha256, not Python's per-process `hash()`, so the value is
the same on every machine and in every process.
"""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(base_seed), *parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
