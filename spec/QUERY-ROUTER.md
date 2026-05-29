# spec · Query Router

`mem.query(text)` is the high-level door. Internally it routes to one or more **retrievers** and fuses results.

## Intent Detection

Input: user text + optional explicit `intent=`.

Order:
1. **Explicit override** wins.
2. **Pattern heuristic** (fast, no LLM): regexes pinned to common locales (`en`, `zh`).
   - Spatial: `near|next to|on|in|under|above|附近|旁边|上面` → adds `spatial`.
   - Temporal: `last|recent|earlier|yesterday|刚才|最近|上次` → adds `temporal`.
   - Otherwise: `semantic`.
3. **Hybrid** when ≥ 2 intents detected.
4. **LLM fallback** (optional, gated by `config.query.llm_intent`) — only if heuristic returns no signal. Used to extract `subject`, `relation`, `anchor` slots.

## Retrievers

### `semantic(text, k)`
- Embed `text` with the same encoder used for node features (must agree on dim).
- `sqlite-vec` ANN over `node_features`. Linear scan when `n_nodes < 10_000`.
- Rerank top 4k with exact cosine to remove ANN noise.

### `spatial(near, radius | inside_bbox, k)`
- R-tree range query on `node_geom`.
- Sort by Euclidean distance to `near`.

### `temporal(n, episode)`
- Index scan on `nodes(t_last DESC)` filtered by episode.

### `hybrid` (used when intent set has ≥ 2)
- Slot-fill: subject (semantic) + anchor (semantic) + relation (lexical).
- Resolve anchor first; use its centroid + radius derived from relation (`on` → 0.3m; `near` → 1.0m; `inside` → bbox).
- Run spatial retrieval against the resolved anchor.
- Re-score subjects within that region by `0.5 * sem_score + 0.5 * spatial_score`.

## Scoring & Fusion

Per-hit score normalized to `[0, 1]` per retriever, then combined:

```
final = Σ_r  w_r * score_r(hit) * confidence_boost(hit)
confidence_boost(hit) = sqrt(node.confidence)         # damp low-conf
```

Default weights: semantic 0.5, spatial 0.3, temporal 0.2 (only retrievers that fired contribute, weights renormalized).

## Subgraph Extraction

When `k_hops > 0`, expand from each top-k node via BFS over edges. Budget: `cfg.query.max_subgraph_nodes` (default 32). Eviction by lowest `final` score.

## Output Determinism

For fixed store state and identical query text, results must be deterministic. ANN backend must be seeded; ties broken by `node.id` ascending.

## Latency Budget (MVP, MacBook Air M2)

| Step | Target |
|---|---|
| Intent detection | < 1 ms |
| Single retriever | < 20 ms @ 10k nodes |
| Hybrid full path | < 80 ms |
| Subgraph extract (k_hops=2) | < 30 ms |
| Total `query()` p95 | < 150 ms |

CI bench gate fails if any of these regress > 20%.

## Future (M2+)

- **Query rewriter** (LLM) — expand "where did I leave my keys" → multiple semantic probes ("keys", "keychain", "house keys").
- **Negation handling** — "the mug that ISN'T on the table" via constraint filter.
- **Episode scoping** — `mem.query(..., episode="2026-05-29")`.
- **Multi-hop reasoning** — "mug on the table in the kitchen on the second floor" parsed into hierarchical anchor chain.

These all live behind feature flags; MVP routes are sufficient for the launch demo.
