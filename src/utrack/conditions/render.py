"""Rendering of context text (Decision G, default rules; the decision itself is still Teo's).

- Evidence spans: natural id order (E2 before E10), text only, one span per line.
- Documents: put in a canonical order (by document id), then shuffled with a seeded
  permutation, each preceded by a delimiter line carrying only a neutral index.

Stored rank order reveals role and distractor subtype in every task (U0.2), and document ids are
sequential in that same order, so neither the stored order nor the ids may reach the output.
Starting the shuffle from a canonical order makes the result independent of the order in which
the documents were passed in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from utrack.data.loader import natural_sort_key
from utrack.data.schema import Document, EvidenceSpan

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def render_evidence(spans: Sequence[EvidenceSpan]) -> tuple[str, tuple[str, ...]]:
    """One span per line. Whitespace inside a span (11 dev spans contain a newline) is
    collapsed to single spaces so that a span stays on its own line."""
    ordered = sorted(spans, key=lambda s: natural_sort_key(s.id))
    lines = [_WHITESPACE_RUN.sub(" ", _normalise_newlines(s.evidence)).strip() for s in ordered]
    return "\n".join(lines), tuple(s.id for s in ordered)


def render_documents(documents: Sequence[Document], seed: int) -> tuple[str, tuple[str, ...]]:
    """Documents in a seeded shuffled order, separated by `[Document i]` delimiter lines."""
    canonical = sorted(documents, key=lambda d: natural_sort_key(d.document_id))
    permutation = np.random.default_rng(seed).permutation(len(canonical))
    shuffled = [canonical[i] for i in permutation]
    parts = [
        f"[Document {index}]\n{_normalise_newlines(doc.text).rstrip()}"
        for index, doc in enumerate(shuffled, start=1)
    ]
    return "\n\n".join(parts), tuple(d.document_id for d in shuffled)
