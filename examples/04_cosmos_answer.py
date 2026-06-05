"""SpatialMem + NVIDIA Cosmos Reason — answer over the scene graph.

Cosmos Reason is NVIDIA's Physical-AI reasoning VLM. SpatialMem keeps the
persistent 3D memory; Cosmos is the reasoning brain that reads the serialized
scene graph and answers. `CosmosReasonVerbalizer` wraps any OpenAI-compatible
Cosmos NIM (hosted build.nvidia.com gateway or a self-hosted NIM).

This runs OFFLINE with a fake transport that mimics a Cosmos <think>/<answer>
reply, so it is CI-runnable with no GPU and no API key. To hit a real NIM, see
the commented block in main().

Run: python examples/04_cosmos_answer.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from spatialmem import CosmosReasonVerbalizer, HashEncoder, SpatialMemory, SyntheticScene, stream

DIM = 64

OBJECTS = [
    ("mug", (1.20, 0.0, 0.94)),
    ("kettle", (1.55, 0.0, 0.93)),
    ("table", (1.40, 0.0, 0.75)),
]


def _fake_cosmos_transport(payload: dict) -> dict:
    """Stand in for a Cosmos NIM: echo a <think>/<answer> formatted reply.

    The real NIM returns the same OpenAI chat shape; `CosmosReasonVerbalizer`
    strips the <think> reasoning and keeps the <answer> body.
    """
    prompt = payload["messages"][0]["content"]
    label = "mug" if "mug" in prompt.lower() else "the scene"
    content = (
        f"<think>Scanning the SCENE facts for {label} and its neighbours.</think>"
        f"<answer>The {label} sits on the table at roughly (1.2, 0.0, 0.94) m.</answer>"
    )
    return {"choices": [{"message": {"content": content}}]}


def main() -> None:
    enc = HashEncoder(DIM)
    scene = SyntheticScene(objects=list(OBJECTS), encoder=enc, n_frames=12)

    # OFFLINE demo verbalizer (no key, no network):
    vb = CosmosReasonVerbalizer(model="nvidia/cosmos-reason2-8b", transport=_fake_cosmos_transport)

    # REAL NIM (uncomment; needs NVIDIA_API_KEY + network):
    # vb = CosmosReasonVerbalizer(model="nvidia/cosmos-reason2-8b")
    #   or self-hosted: CosmosReasonVerbalizer(base_url="http://localhost:8000/v1", model=...)

    path = Path(tempfile.mkdtemp()) / "kitchen.smem"
    with SpatialMemory.open(path, embedding_dim=DIM, encoder=enc, verbalizer=vb) as mem:
        n_frames, n_obs = stream(mem, scene, commit_every=4)
        st = mem.stats()
        print(f"streamed {n_frames} frames, {n_obs} obs -> {st.n_nodes} nodes after fusion")
        print("answer('where is the mug?') ->", mem.answer("where is the mug?"))


if __name__ == "__main__":
    main()
