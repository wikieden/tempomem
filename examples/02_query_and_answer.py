"""Chronotope M2 tour — semantic query, LLM answer, decay, resplit.

No GPU, no network. Brings up a synthetic kitchen, then exercises the M2
surface: encoder-backed semantic query, a BYO verbalizer, confidence decay,
and split detection. Run: python examples/02_query_and_answer.py
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from tempomem import Detection, TempoMem

DIM = 32
DAY = 86400.0


class HashEncoder:
    """Deterministic stand-in for CLIP: maps text -> seeded unit vector.

    Same string -> same vector, so a detection labelled "mug" and the query
    "mug" land on the same embedding. Swap for tempomem.encoders.OpenClipEncoder
    (the [clip] extra) for real open-vocabulary semantics.
    """

    @property
    def dim(self) -> int:
        return DIM

    def encode_text(self, texts) -> np.ndarray:
        rows = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32))
            v = rng.standard_normal(DIM).astype("float32")
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype="float32")


class EchoVerbalizer:
    """Trivial BYO verbalizer. Real use: wrap OpenAI / Anthropic / Ollama."""

    def complete(self, prompt: str) -> str:
        # A real LLM reads the SCENE block; here we just echo the top match line.
        for line in prompt.splitlines():
            if line.strip().startswith("#"):
                return f"Most likely: {line.strip()}"
        return "I don't know from this scene."


def _det(enc: HashEncoder, label: str, center, ts: float) -> Detection:
    cx, cy, cz = center
    return Detection(
        label=label,
        feature=enc.encode_text([label])[0],
        center_xyz=center,
        bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
        bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
        confidence=0.9,
        ts=ts,
    )


def main() -> None:
    enc = HashEncoder()
    path = Path(tempfile.mkdtemp()) / "kitchen.smem"
    with TempoMem.open(
        path, embedding_dim=DIM, encoder=enc, verbalizer=EchoVerbalizer()
    ) as mem:
        mem.add_detections(
            [
                _det(enc, "mug", (1.20, 0.30, 0.94), ts=0.0),
                _det(enc, "kettle", (1.55, 0.40, 0.93), ts=1.0),
                _det(enc, "fridge", (2.40, 0.00, 1.20), ts=2.0),
                _det(enc, "sink", (0.80, 0.35, 0.90), ts=3.0),
            ]
        )
        stats = mem.commit()
        print(f"ingested -> {stats.nodes_after} nodes")

        hits = mem.semantic("mug")
        print(f"semantic 'mug' -> #{hits[0].id} {hits[0].label} score={hits[0].score:.3f}")

        # HashEncoder only aligns on exact words; a real CLIP encoder embeds
        # full natural-language questions. Query the keyword here so the stub
        # retrieves sensibly.
        ans = mem.answer("mug")
        print(f"answer -> {ans}")

        # Decay: pretend 60 days passed; low-confidence stuff would prune.
        decayed, pruned = mem.decay(half_life_days=30.0, min_conf=0.1, now=60 * DAY)
        print(f"decay(60d) -> decayed={decayed} pruned={pruned}")

        # Split detection: no-op here (each object is a tight single cluster).
        split, created = mem.resplit()
        print(f"resplit -> split={split} created={created}")


if __name__ == "__main__":
    main()
