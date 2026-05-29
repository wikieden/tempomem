"""LLM verbalizer: turn retrieved nodes into a natural-language answer.

BYO model — SpatialMem ships no LLM dependency and no API key. Supply any
object implementing the `Verbalizer` protocol (wrap OpenAI / Anthropic /
Ollama / a local model). See spec/QUERY-ROUTER.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .query import NodeHit

_SYSTEM = (
    "You answer questions about a 3D scene from a spatial memory graph. "
    "Use only the SCENE facts below. Coordinates are meters (x, y, z). "
    "If the scene does not contain the answer, say you don't know. Be concise."
)


@runtime_checkable
class Verbalizer(Protocol):
    """Completes a prompt into an answer string. Wrap any chat/LLM backend."""

    def complete(self, prompt: str) -> str: ...


def build_answer_prompt(query: str, scene_text: str, hits: list[NodeHit]) -> str:
    """Assemble the verbalizer prompt from the query, serialized scene, and top hits."""
    lines = [_SYSTEM, "", "SCENE:", scene_text, ""]
    if hits:
        lines.append("TOP MATCHES (id, label, xyz, score):")
        for h in hits:
            c = h.center_xyz
            lines.append(
                f"  #{h.id} {h.label} [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}] score={h.score:.3f}"
            )
        lines.append("")
    lines.append(f"QUESTION: {query}")
    lines.append("ANSWER:")
    return "\n".join(lines)
