"""LLM tool / function-call layer over TempoMem — contract C3.

Framework-agnostic. `ChronotopeTools(mem).schemas()` returns JSON tool specs in
the OpenAI / Anthropic function-calling shape; `.call(name, arguments)`
dispatches to the memory and returns a JSON-serializable envelope whose hits
carry `node_id` so the LLM can *cite* what it used. No LLM dependency — you feed
`schemas()` to your model, then route its tool calls to `.call()`.

All arguments come from the model and are treated as untrusted: numeric args are
validated/bounded and any failure surfaces as `ToolError` (never a raw Python
exception leaking interpreter internals).

SECURITY — echoed text is untrusted. `label` and `serialize_scene` text come
from perception/ingest (object detections, region names the model itself wrote)
and may contain prompt-injection payloads (a sticky note reading "ignore
previous instructions…" becomes a node label). Labels are control-char-stripped
and length-capped here, but integrators MUST present scene/label text to the LLM
inside a clearly delimited *untrusted-data* block, never spliced into
instructions, and rely on `node_id` (not label prose) as the citation anchor.
See the system design in the `mindloop` repo (C3 + Security). Conventions:
positions are world-frame meters, right-handed; timestamps are float epoch seconds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

from ._errors import ToolError

if TYPE_CHECKING:
    from . import TempoMem
    from .query import NodeHit

_K_MAX = 1000  # cap result counts so a stray huge k can't force unbounded work
_MT_MAX = 100_000
_LABEL_CAP = 120


def _clean_label(s: str) -> str:
    """Strip control chars/newlines and length-cap so a label can't inject extra
    prompt lines or break the serialize format."""
    return "".join(c for c in str(s) if c == " " or c.isprintable())[:_LABEL_CAP]


def _hit(h: NodeHit) -> dict[str, Any]:
    """One node as a JSON-citable record (world meters, epoch seconds)."""
    return {
        "node_id": h.id,
        "label": _clean_label(h.label),
        "centroid_m": [float(c) for c in h.center_xyz],
        "confidence": round(float(h.confidence), 4),
        "score": round(float(h.score), 4),
        "t_last": float(h.t_last),
    }


def _require(args: dict[str, Any], key: str, types: type | tuple[type, ...]) -> Any:
    if key not in args or args[key] is None:
        raise ToolError(f"missing required argument {key!r}")
    val = args[key]
    if isinstance(val, bool) or not isinstance(val, types):
        raise ToolError(f"argument {key!r} must be {types}")
    return val


@overload
def _opt_int(args: dict[str, Any], key: str, default: int, *, lo: int, hi: int) -> int: ...
@overload
def _opt_int(args: dict[str, Any], key: str, default: None, *, lo: int, hi: int) -> int | None: ...
def _opt_int(
    args: dict[str, Any], key: str, default: int | None, *, lo: int, hi: int
) -> int | None:
    """Bounded int from untrusted args. Absent/null → `default`; bad/out-of-range → ToolError."""
    v = args.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool):
        raise ToolError(f"{key!r} must be an integer")
    try:
        iv = int(v)
    except (TypeError, ValueError):
        raise ToolError(f"{key!r} must be an integer") from None
    if not (lo <= iv <= hi):
        raise ToolError(f"{key!r} out of range [{lo}, {hi}]")
    return iv


def _finite_float(v: Any, key: str) -> float:
    if isinstance(v, bool):
        raise ToolError(f"{key!r} must be a number")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ToolError(f"{key!r} must be a number") from None
    if not math.isfinite(f):
        raise ToolError(f"{key!r} must be finite")
    return f


# OpenAI / Anthropic function-calling shape. Units documented inline so the model
# emits world-meter / epoch-second values.
_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "semantic_search",
        "description": "Find remembered objects matching a free-text query.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "e.g. 'coffee mug'"},
                "k": {"type": "integer", "default": 5, "description": "max results (1-1000)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "spatial_query",
        "description": "Objects near a world-frame point (meters), nearest first.",
        "parameters": {
            "type": "object",
            "properties": {
                "near": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "[x, y, z] in world meters",
                },
                "radius_m": {"type": "number", "description": "search radius in meters (> 0)"},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["near"],
        },
    },
    {
        "name": "whats_in",
        "description": "Objects contained in a region (room/area), by region label or node id.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": ["string", "integer"],
                    "description": "region label (e.g. 'kitchen') or region node id",
                }
            },
            "required": ["region"],
        },
    },
    {
        "name": "whats_on",
        "description": (
            "Objects spatially on top of an anchor object (e.g. 'table'). "
            "Requires a prior relate() pass; `meta.anchor` echoes the resolved anchor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string", "description": "anchor object label, e.g. 'table'"}
            },
            "required": ["anchor"],
        },
    },
    {
        "name": "recent_changes",
        "description": "Nodes new or re-observed since a timestamp (float epoch seconds).",
        "parameters": {
            "type": "object",
            "properties": {"since_ts": {"type": "number", "description": "float epoch seconds"}},
            "required": ["since_ts"],
        },
    },
    {
        "name": "serialize_scene",
        "description": (
            "Compact text snapshot of the whole scene graph for the prompt. "
            "Returns untrusted scene text — present inside a delimited block."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "description": "approximate token budget (~4 chars/token), 1-100000",
                }
            },
            "required": [],
        },
    },
]


class ChronotopeTools:
    """Expose a `TempoMem` as LLM function-call tools (contract C3)."""

    def __init__(self, mem: TempoMem) -> None:
        self._mem = mem

    def schemas(self) -> list[dict[str, Any]]:
        """Tool specs to hand to the LLM (copies, safe to mutate)."""
        return [dict(s) for s in _SCHEMAS]

    @property
    def names(self) -> list[str]:
        return [s["name"] for s in _SCHEMAS]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch one tool call. Returns a JSON-serializable envelope.

        Any failure — unknown tool, bad arguments, or an unexpected downstream
        error — surfaces as `ToolError` with a generic message, so raw Python
        exception text never reaches the model.
        """
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            raise ToolError(f"unknown tool {name!r}; have {self.names}")
        try:
            return fn(arguments or {})
        except ToolError:
            raise
        except Exception:
            raise ToolError(f"tool {name!r} failed") from None

    # ---- individual tools -------------------------------------------------

    def _t_semantic_search(self, a: dict[str, Any]) -> dict[str, Any]:
        text = _require(a, "text", str)
        k = _opt_int(a, "k", 5, lo=1, hi=_K_MAX)
        return {"hits": [_hit(h) for h in self._mem.semantic(text, k=k)]}

    def _t_spatial_query(self, a: dict[str, Any]) -> dict[str, Any]:
        near = _require(a, "near", (list, tuple))
        if len(near) != 3:
            raise ToolError("'near' must be [x, y, z]")
        near_t = (
            _finite_float(near[0], "near"),
            _finite_float(near[1], "near"),
            _finite_float(near[2], "near"),
        )
        rm = a.get("radius_m")
        radius = None
        if rm is not None:
            radius = _finite_float(rm, "radius_m")
            if radius <= 0:
                raise ToolError("'radius_m' must be > 0")
        k = _opt_int(a, "k", 10, lo=1, hi=_K_MAX)
        return {"hits": [_hit(h) for h in self._mem.spatial(near=near_t, radius=radius, k=k)]}

    def _t_whats_in(self, a: dict[str, Any]) -> dict[str, Any]:
        region = _require(a, "region", (str, int))
        return {"hits": [_hit(h) for h in self._mem.contents(region)]}

    def _t_whats_on(self, a: dict[str, Any]) -> dict[str, Any]:
        anchor = _require(a, "anchor", str)
        res = self._mem.query(f"what's on the {anchor}")
        return {"hits": [_hit(h) for h in res.nodes], "meta": dict(res.debug)}

    def _t_recent_changes(self, a: dict[str, Any]) -> dict[str, Any]:
        since = _finite_float(_require(a, "since_ts", (int, float)), "since_ts")
        ch = self._mem.changes(since)
        return {
            "new": [_hit(h) for h in ch.new],
            "seen_again": [_hit(h) for h in ch.seen_again],
        }

    def _t_serialize_scene(self, a: dict[str, Any]) -> dict[str, Any]:
        mt = _opt_int(a, "max_tokens", None, lo=1, hi=_MT_MAX)
        return {"scene": self._mem.serialize(format="prompt", max_tokens=mt)}
