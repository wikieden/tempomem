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


@dataclass
class HygieneReport:
    """Counts from one decay (+ optional forget) lifecycle pass."""

    nodes_before: int
    decayed: int
    pruned: int
    nodes_after_decay: int
    forgotten: int


def persistence_after_reopen(
    path: str, *, embedding_dim: int, cases: list[EvalCase], k: int = 5
) -> EvalReport:
    """Reopen a persisted ``.smem`` and measure recall@k.

    Proves memory survives a restart; when ``cases`` cover objects ingested under
    different episodes it also measures cross-episode persistence. The store at
    ``path`` must already be committed and closed by the caller.
    """
    from . import SpatialMemory

    mem = SpatialMemory.open(path, embedding_dim=embedding_dim, create=False)
    try:
        return recall_at_k(mem, cases, k=k)
    finally:
        mem.close()


def decay_forget(
    mem: SpatialMemory,
    *,
    half_life_days: float = 30.0,
    min_conf: float = 0.1,
    forget_ids: list[int] | None = None,
    now: float | None = None,
) -> HygieneReport:
    """Run a decay (+ optional forget) pass and report the lifecycle counts — the
    decay/forget-correctness dimension of the eval suite. Deterministic when
    ``now`` is pinned (seconds; defaults to wall-clock).
    """
    before = mem.stats().n_nodes
    decayed, pruned = mem.decay(half_life_days=half_life_days, min_conf=min_conf, now=now)
    after = mem.stats().n_nodes
    forgotten = 0
    for nid in forget_ids or []:
        mem.forget(nid)
        forgotten += 1
    return HygieneReport(
        nodes_before=before,
        decayed=decayed,
        pruned=pruned,
        nodes_after_decay=after,
        forgotten=forgotten,
    )
