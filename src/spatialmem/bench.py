"""Lightweight retrieval evaluation: recall@k over scripted queries.

Reusable for the M2 demo metric ("ask 5 questions, get 4 right"). No external
deps; operates on a live SpatialMemory via its public query API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import SpatialMemory

EvalCase = tuple[str, str]  # (query_text, expected_label)


@dataclass
class EvalReport:
    total: int
    hits: int
    k: int
    misses: list[str]  # query texts that missed

    @property
    def recall(self) -> float:
        return self.hits / self.total if self.total else 0.0


def recall_at_k(mem: SpatialMemory, cases: list[EvalCase], *, k: int = 5) -> EvalReport:
    """Fraction of cases where the expected label appears in the top-k results.

    Label match is case-insensitive substring (handles "coffee mug" vs "mug").
    """
    hits = 0
    misses: list[str] = []
    for query_text, expected in cases:
        res = mem.query(query_text, k=k)
        labels = [h.label.lower() for h in res.nodes[:k]]
        exp = expected.lower()
        if any(exp in lab or lab in exp for lab in labels):
            hits += 1
        else:
            misses.append(query_text)
    return EvalReport(total=len(cases), hits=hits, k=k, misses=misses)
