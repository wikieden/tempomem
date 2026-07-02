"""`tempomem` CLI. M0: inspect a store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import TempoMem, __version__


def _inspect(path: str, embedding_dim: int) -> int:
    with TempoMem.open(path, embedding_dim=embedding_dim, create=False, readonly=True) as mem:
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


def _viz(path: str, embedding_dim: int, out: str | None) -> int:
    from . import viz

    with TempoMem.open(path, embedding_dim=embedding_dim, create=False, readonly=True) as mem:
        html = viz.to_html(mem._conn, embedding_dim, title=Path(path).name)
    if out:
        Path(out).write_text(html, encoding="utf-8")
        print(f"wrote {out} ({len(html)} bytes)")
    else:
        sys.stdout.write(html)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tempomem", description="Chronotope CLI")
    parser.add_argument("--version", action="version", version=f"tempomem {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_ins = sub.add_parser("inspect", help="show store contents")
    p_ins.add_argument("path")
    p_ins.add_argument("--embedding-dim", type=int, default=512)

    p_viz = sub.add_parser("viz", help="export a read-only HTML scene viewer")
    p_viz.add_argument("path")
    p_viz.add_argument("-o", "--out", default=None, help="output .html (default: stdout)")
    p_viz.add_argument("--embedding-dim", type=int, default=512)

    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        return _inspect(args.path, args.embedding_dim)
    if args.cmd == "viz":
        return _viz(args.path, args.embedding_dim, args.out)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
