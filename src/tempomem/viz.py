"""Read-only HTML viewer for a .smem store (deferred M1 web viewer).

Produces a single self-contained HTML file: a top-down 2D scatter (world X
horizontal, Z vertical) of node centroids with labels, plus a node table. No
external deps, no network, no JS framework — inline canvas script + embedded
JSON. Render via `tempomem viz store.smem -o scene.html`.
"""

from __future__ import annotations

import json
import sqlite3

from . import serialize

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chronotope — {title}</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #0e1116; color: #e6edf3; }}
  header {{ padding: 12px 18px; border-bottom: 1px solid #30363d; }}
  h1 {{ font-size: 16px; margin: 0; }}
  .meta {{ color: #8b949e; font-size: 12px; margin-top: 2px; }}
  main {{ display: flex; flex-wrap: wrap; gap: 18px; padding: 18px; }}
  canvas {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; font-weight: 600; }}
  .empty {{ color: #8b949e; padding: 18px; }}
</style>
</head>
<body>
<header>
  <h1>Chronotope scene</h1>
  <div class="meta">{n_nodes} nodes · {n_edges} edges
    · {n_obs} observations · embedding_dim {dim}</div>
</header>
<main>
  <canvas id="c" width="560" height="560"></canvas>
  <div id="side"></div>
</main>
<script>
const DATA = {data};
const nodes = DATA.nodes || [];
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const side = document.getElementById("side");
if (!nodes.length) {{
  side.innerHTML = '<div class="empty">Empty store.</div>';
}} else {{
  const xs = nodes.map(n => n.centroid[0]), zs = nodes.map(n => n.centroid[2]);
  const pad = 40, W = cv.width - 2*pad, H = cv.height - 2*pad;
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const spanX = (maxX - minX) || 1, spanZ = (maxZ - minZ) || 1;
  const sx = v => pad + (v - minX) / spanX * W;
  const sz = v => pad + (v - minZ) / spanZ * H;
  ctx.strokeStyle = "#30363d"; ctx.strokeRect(pad, pad, W, H);
  ctx.fillStyle = "#58a6ff"; ctx.font = "12px system-ui";
  for (const n of nodes) {{
    const x = sx(n.centroid[0]), y = sz(n.centroid[2]);
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = "#e6edf3"; ctx.fillText(n.label + " #" + n.id, x + 8, y + 4);
    ctx.fillStyle = "#58a6ff";
  }}
  let rows = nodes.map(n =>
    `<tr><td>#${{n.id}}</td><td>${{n.label}}</td>` +
    `<td>[${{n.centroid.map(c => c.toFixed(2)).join(", ")}}]</td>` +
    `<td>${{n.confidence.toFixed(2)}}</td><td>${{n.n_obs}}</td></tr>`).join("");
  side.innerHTML =
    '<table><thead><tr><th>id</th><th>label</th><th>centroid (x,y,z)</th>' +
    '<th>conf</th><th>n_obs</th></tr></thead><tbody>' + rows + '</tbody></table>';
}}
</script>
</body>
</html>
"""


def to_html(conn: sqlite3.Connection, embedding_dim: int, *, title: str = "scene") -> str:
    """Render the store as a self-contained HTML string."""
    graph = serialize.to_json(conn, embedding_dim)
    return _TEMPLATE.format(
        title=title,
        n_nodes=len(graph["nodes"]),
        n_edges=len(graph["edges"]),
        n_obs=sum(n["n_obs"] for n in graph["nodes"]),
        dim=embedding_dim,
        data=json.dumps(graph),
    )
