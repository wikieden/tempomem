"""`spatialmem` CLI. M0: inspect a store."""

from __future__ import annotations

import argparse
import sys

from . import SpatialMemory, __version__


def _inspect(path: str, embedding_dim: int) -> int:
    with SpatialMemory.open(path, embedding_dim=embedding_dim, create=False, readonly=True) as mem:
        st = mem.stats()
        print(f"store:      {path}")
        print(f"nodes:      {st.n_nodes}")
        print(f"edges:      {st.n_edges}")
        print(f"obs:        {st.n_obs}")
        print(f"episodes:   {st.n_episodes}")
        print(f"size:       {st.store_bytes} bytes")
        sample = mem.recent(n=5)
        if sample:
            print("recent nodes:")
            for h in sample:
                c = h.center_xyz
                print(
                    f'  #{h.id} "{h.label}" '
                    f"@[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}] conf={h.confidence:.2f}"
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spatialmem", description="SpatialMem CLI")
    parser.add_argument("--version", action="version", version=f"spatialmem {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_ins = sub.add_parser("inspect", help="show store contents")
    p_ins.add_argument("path")
    p_ins.add_argument("--embedding-dim", type=int, default=512)

    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        return _inspect(args.path, args.embedding_dim)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
