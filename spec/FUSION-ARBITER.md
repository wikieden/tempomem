# spec · Fusion Arbiter

The arbiter decides, for every incoming `Observation`, whether to **merge** it into an existing `Node`, **create** a new node, or **reject** it. This is the single most consequential algorithm in the system; the rest of Chronotope is plumbing around it.

## Inputs

- Pending observation `o = (label, feature, center, bbox, confidence, ts)`.
- Read snapshot of nodes whose bbox intersects an expanded query box around `o.bbox` (R-tree range query, dilation = `cfg.search_dilation_m`).

## Pipeline

```
1. Candidate fetch   ─→  C = nodes within bbox dilation
2. Score each c ∈ C:
       s_geom    = max(0, 1 - dist(o.center, c.centroid) / cfg.dist_norm_m)
       s_iou     = iou3d(o.bbox, c.bbox)
       s_sem     = cos(o.feature, c.feature_centroid)
       s_label   = label_compat(o.label, c.labels_dist)     # ∈ [0, 1]
       s         = w_g * s_geom + w_i * s_iou + w_s * s_sem + w_l * s_label
3. Pick best c* = argmax s
4. Decision:
       if s(c*) ≥ τ_merge          → MERGE into c*
       elif s(c*) ≥ τ_ambig        → DEFER (queue for review at commit)
       elif o.confidence < τ_obs   → REJECT
       else                        → NEW node
```

Defaults (`FusionConfig`):

| param | default | note |
|---|---|---|
| `search_dilation_m` | 0.25 | meters dilation for R-tree query |
| `dist_norm_m` | 0.50 | distance at which `s_geom = 0` |
| `w_g, w_i, w_s, w_l` | 0.2, 0.2, 0.5, 0.1 | sum to 1.0 |
| `τ_merge` | 0.62 | tuned on synthetic kitchen set |
| `τ_ambig` | 0.45 | below this → new node |
| `τ_obs` | 0.30 | reject low-confidence detections |

All four sub-scores are clipped to `[0, 1]`.

## Label Compatibility

`label_compat(o.label, c.labels)`:

1. Exact match against any label in `c.labels` → `max(weight, 0.8)`.
2. CLIP-text cosine between `o.label` and each label in `c.labels`, take max → `1 / (1 + exp(-10*(cos - 0.5)))` (sigmoid centered at 0.5).
3. If both `o.label` and a label in `c.labels` carry known antonyms ("open" vs "closed") → `0`.

Antonym list ships with the package; user can extend via config.

## Merge Operation

Atomic SQL transaction. For node `c*` and incoming observation `o`:

- `c*.n_obs ← c*.n_obs + 1`
- `c*.centroid ← w-mean(c*.centroid * c*.conf_sum, o.center * o.confidence)` (conf-weighted EMA, α = `cfg.centroid_alpha`, default 0.2)
- `c*.bbox ← AABB(c*.bbox ∪ o.bbox)` then optional shrink toward centroid by `cfg.bbox_shrink` to resist outliers
- `c*.feature_centroid ← normalize(c*.feature_centroid + α * (o.feature - c*.feature_centroid))`
- `c*.labels_dist[o.label] += o.confidence`; renormalize
- `c*.confidence ← min(1, c*.confidence + (1 - c*.confidence) * o.confidence * cfg.conf_gain)`
- `c*.t_last ← max(c*.t_last, o.ts)`
- Insert `node_obs(c*.id, o.id, o.ts)`
- Update the `node_vec` index (when the `[vec]` extra is active). *(As built: no `node_geom` R-tree — proximity is a linear AABB scan; see SCHEMA.md.)*

## Split Detection

**As built (M2, V4):** exposed as `mem.resplit()` — an explicit maintenance
sweep (not auto post-commit). For each node it runs a deterministic 2-means
(seeded by the farthest observation pair) over member-observation centroids. A
node splits into two when both clusters have ≥ `cfg.min_split_obs` members and
their centroids are separated by ≥ `cfg.tau_split_m`; member `node_obs` rows are
reassigned and the original node deleted. Returns `(nodes_split, new_nodes)`.

Future: silhouette-scored k>2 splits and auto-trigger on bbox-volume growth.

## Decay

**As built (M2, V2):** `mem.decay(half_life_days=30.0, min_conf=0.1, now=None)`:

```
for node n:
    age_days = (now - n.t_last) / 86400
    n.confidence *= 0.5 ** (age_days / half_life_days)
    if n.confidence < min_conf: forget(n)   # else persist decayed confidence
```

Returns `(n_decayed, n_pruned)`. Edge decay is future work.

## Determinism

For a fixed config + fixed observation stream + fixed feature vectors, the arbiter must produce a byte-identical SQLite file across runs. This is testable: `tests/test_determinism.py` ingests a recorded stream twice into fresh files and SHA-256-diffs the result.

## Logging

Every decision emits one structured log line:

```json
{"event":"fuse","obs_id":123,"decision":"merge","node_id":42,
 "score":0.71,"scores":{"geom":0.82,"iou":0.55,"sem":0.79,"label":0.80},
 "ts":1735.412}
```

`mem.replay_decisions(obs_id)` returns the log line — primary debugging affordance for users complaining "it merged my mug into the kettle."

## Open questions (track in `docs/05-OPEN.md`)

- Should fusion run synchronously per-frame or batched at commit? (current: batched.)
- How to weight static vs dynamic objects (e.g., a mug moves; a fridge doesn't)?
- Per-class threshold overrides — needed at M2 or sooner?
