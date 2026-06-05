"""NVIDIA Cosmos integration — Cosmos Reason as a SpatialMem `Verbalizer`.

Cosmos Reason is NVIDIA's Physical-AI reasoning vision-language model family:
Cosmos-Reason1 (7B), Cosmos Reason 2 (2B/8B/32B), and the "Cosmos 3 Reasoner"
inside the unified Cosmos 3 world-foundation-model family. It is a chain-of-
thought reasoner, not a detector — it does **not** emit world-frame 3D
detections, so it is not a `PerceptionAdapter`. It *is* a strong `answer()`
backend: it reads the serialized scene graph and reasons about it with physical
common sense.

This wraps any OpenAI-compatible NVIDIA NIM endpoint (the hosted
build.nvidia.com gateway, or a self-hosted NIM microservice). No new core
dependency — it uses only the standard library (`urllib`). The model id is
configurable, so the same wrapper targets Cosmos Reason 1/2 or the Cosmos 3
Reasoner NIM.

    from spatialmem import SpatialMemory
    from spatialmem.cosmos import CosmosReasonVerbalizer

    vb = CosmosReasonVerbalizer(model="nvidia/cosmos-reason2-8b")  # NVIDIA_API_KEY env
    mem = SpatialMemory.open("room.smem", embedding_dim=512, verbalizer=vb)
    print(mem.answer("what is on the kitchen table?"))
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from ._errors import QueryError

# NVIDIA-hosted, OpenAI-compatible NIM gateway (build.nvidia.com).
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# A confirmed published NIM. For the unified Cosmos 3 family the reasoning piece
# ships as the "Cosmos 3 Reasoner" NIM — set ``model=`` to its id once GA.
DEFAULT_MODEL = "nvidia/cosmos-reason2-8b"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# An unterminated <think> (response truncated by max_tokens) — drop to the end.
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"</?(?:think|answer)>", re.IGNORECASE)

Transport = Callable[[dict[str, Any]], dict[str, Any]]


def strip_reasoning(text: str) -> str:
    """Extract the final answer from a Cosmos chain-of-thought response.

    Cosmos Reason emits ``<think>...</think>`` reasoning and (Reason1) wraps the
    final answer in ``<answer>...</answer>``. Strip the reasoning *first* (so an
    ``<answer>`` quoted inside a think block can't win), tolerate an unterminated
    ``<think>`` from a truncated reply, then return the ``<answer>`` body if
    present. Falls back to the de-tagged text so the result is never empty when
    the model did produce content.
    """
    cleaned = _OPEN_THINK_RE.sub("", _THINK_RE.sub("", text))
    m = _ANSWER_RE.search(cleaned)
    if m and m.group(1).strip():
        return m.group(1).strip()
    out = _TAG_RE.sub("", cleaned).strip()
    return out or _TAG_RE.sub("", text).strip()


class CosmosReasonVerbalizer:
    """`Verbalizer` backed by an NVIDIA Cosmos Reason NIM (OpenAI-compatible).

    Targets a hosted (build.nvidia.com) or self-hosted NIM. Set ``model=`` to the
    desired Cosmos Reason / Cosmos 3 Reasoner id. Reads the API key from
    ``api_key=`` or the ``NVIDIA_API_KEY`` environment variable.

    ``transport`` is an injection seam for testing: a callable taking the request
    payload dict and returning the parsed JSON response dict. It defaults to a
    stdlib-``urllib`` POST, so the core stays dependency-free.

    Security: ``base_url`` is caller-supplied trusted input (your own NIM
    endpoint). Do not point it at an untrusted/user-derived URL — the POST goes
    wherever it says (SSRF), and the ``NVIDIA_API_KEY`` bearer token rides along.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._transport = transport or self._http_post

    def complete(self, prompt: str) -> str:
        """Send `prompt` to the Cosmos NIM and return the cleaned answer text."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        data = self._transport(payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise QueryError(f"Cosmos NIM returned an unexpected response: {data!r}") from e
        return strip_reasoning(text)

    def _http_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise QueryError(
                "CosmosReasonVerbalizer needs an API key: pass api_key= or set NVIDIA_API_KEY"
            )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            # URLError: transport failure. JSONDecodeError: 200-OK non-JSON body
            # (e.g. an HTML error page from a reverse proxy) — both must surface
            # as QueryError, never a raw ValueError, per the answer() contract.
            raise QueryError(f"Cosmos NIM request failed: {e}") from e
