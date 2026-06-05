from __future__ import annotations

import pytest

from spatialmem import CosmosReasonVerbalizer, HashEncoder, SpatialMemory
from spatialmem._errors import QueryError
from spatialmem.cosmos import strip_reasoning
from tests.conftest import DIM, make_det

_COSMOS_REPLY = "<think>reasoning here</think><answer>on the table</answer>"


def _fake_transport(captured: dict):
    def transport(payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": _COSMOS_REPLY}}]}

    return transport


def test_strip_reasoning_prefers_answer_tag() -> None:
    assert strip_reasoning("<think>foo</think><answer>the mug</answer>") == "the mug"


def test_strip_reasoning_drops_think_block() -> None:
    assert strip_reasoning("<think>foo</think>   bar  ") == "bar"


def test_strip_reasoning_passthrough() -> None:
    assert strip_reasoning("just a plain answer") == "just a plain answer"


def test_strip_reasoning_answer_inside_think_ignored() -> None:
    # an <answer> quoted inside the reasoning must not win over the real one
    text = "<think>I might say <answer>wrong</answer></think><answer>right</answer>"
    assert strip_reasoning(text) == "right"


def test_strip_reasoning_unclosed_think_truncated() -> None:
    # max_tokens cut the reply mid-reasoning: no closing tag, no answer
    assert strip_reasoning("<think>still reasoning when cut") == "still reasoning when cut"


def test_strip_reasoning_only_think_never_empty() -> None:
    # only a reasoning block, no answer -> de-tag content, never return ""
    assert strip_reasoning("<think>just reasoning</think>") == "just reasoning"


def test_complete_builds_openai_payload_and_cleans_output() -> None:
    cap: dict = {}
    vb = CosmosReasonVerbalizer(model="nvidia/cosmos-reason2-8b", transport=_fake_transport(cap))
    out = vb.complete("QUESTION: where is the mug?")
    assert out == "on the table"  # <think>/<answer> stripped
    p = cap["payload"]
    assert p["model"] == "nvidia/cosmos-reason2-8b"
    assert p["messages"][0]["role"] == "user"
    assert p["messages"][0]["content"].startswith("QUESTION")


def test_unexpected_response_raises_query_error() -> None:
    vb = CosmosReasonVerbalizer(transport=lambda payload: {"unexpected": True})
    with pytest.raises(QueryError):
        vb.complete("hi")


def test_http_post_non_json_body_raises_query_error(monkeypatch) -> None:
    # a 200-OK non-JSON body (e.g. an HTML 502 from a reverse proxy) must
    # surface as QueryError, not a raw json.JSONDecodeError / ValueError
    import urllib.request as urlreq

    class _Resp:
        def read(self) -> bytes:
            return b"<html>502 Bad Gateway</html>"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

    monkeypatch.setattr(urlreq, "urlopen", lambda req, timeout=None: _Resp())
    vb = CosmosReasonVerbalizer(api_key="dummy-key")
    with pytest.raises(QueryError, match="request failed"):
        vb._http_post({"model": "x"})


def test_missing_api_key_raises_on_http(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    vb = CosmosReasonVerbalizer(api_key=None)
    with pytest.raises(QueryError, match="API key"):
        vb._http_post({"model": "x"})


def test_answer_through_facade(tmp_path) -> None:
    cap: dict = {}
    vb = CosmosReasonVerbalizer(transport=_fake_transport(cap))
    enc = HashEncoder(DIM)
    with SpatialMemory.open(
        tmp_path / "c.smem", embedding_dim=DIM, encoder=enc, verbalizer=vb
    ) as mem:
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        mem.commit()
        ans = mem.answer("where is the mug?")
    assert ans == "on the table"
    # the serialized scene graph reached the model
    assert "SCENE" in cap["payload"]["messages"][0]["content"]
