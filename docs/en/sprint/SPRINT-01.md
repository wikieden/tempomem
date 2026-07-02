> 🌐 **English** · [中文](../../zh/sprint/SPRINT-01.md)

# Sprint 01 · MVP — Real Fusion + Semantic Retrieval (M1)

**Goal:** Replace the M0 node-per-observation stub with the real arbiter from [FUSION-ARBITER.md](../../../spec/FUSION-ARBITER.md). Two sightings of the same object converge to **one** node with aggregated geometry/features. This is the value ConceptGraphs' single-shot pipeline can't deliver: incremental, persistent dedup.

**Exit:** quickstart kitchen (mug seen twice) → mug is ONE node with `n_obs=2`; determinism test green; coverage ≥ 75% on `fusion`.

## Task Breakdown

| ID | Task | Output | Est (CC) |
|---|---|---|---|
| F1 | `FusionConfig` dataclass — thresholds (τ_merge, τ_ambig, τ_obs, weights, dist_norm, search_dilation, centroid_alpha, conf_gain) | tunable config | 20 min |
| F2 | `store.candidates_near` — fetch nodes whose bbox (dilated) overlaps an observation's bbox | candidate set | 25 min |
| F3 | `store.merge_observation_into_node` — atomic update: conf-weighted centroid EMA, bbox union, feature EMA (renormalized), label dist, conf gain, t_last, n_obs++, node_obs link | merge tx | 40 min |
| F4 | `fusion.score` — geom + iou3d + sem(cos) + label_compat, weighted sum, all clipped [0,1] | match scorer | 35 min |
| F5 | `fusion.ingest_observation` — candidate→score→argmax→decide(merge/new/reject) | real arbiter | 30 min |
| F6 | `label_compat` — exact-match + (optional CLIP-text later) + antonym=0; M1 uses exact/substring, CLIP hook stubbed | label scoring | 20 min |
| F7 | Wire `FusionConfig` through `TempoMem.open(config=...)` | configurable | 15 min |
| F8 | Tests: dedup (2 sightings→1 node), distinct objects stay separate, reject low-conf, determinism (same stream twice → identical node count + centroids) | green | 45 min |
| F9 | Update quickstart + SPRINT-00 note; bump version 0.0.1→0.1.0a1 | demo shows dedup | 15 min |

Total ~3.5 h CC.

## Scoring (from spec, M1 defaults)

```
s = w_g*s_geom + w_i*s_iou + w_s*s_sem + w_l*s_label
  w_g,w_i,w_s,w_l = 0.2, 0.2, 0.5, 0.1
  s_geom = max(0, 1 - dist/dist_norm_m)      dist_norm_m=0.50
  s_iou  = iou3d(obs.bbox, node.bbox)
  s_sem  = cos(obs.feat, node.feat_centroid)
  s_label= exact? max(weight,0.8) : substring? 0.5 : 0  (CLIP-text in M2)
decision: s>=τ_merge(0.62) merge | s>=τ_ambig(0.45) new | conf<τ_obs(0.30) reject | else new
```

Note: M1 simplifies label_compat (no CLIP-text encoder yet — that's the `[clip]` extra, wired M2). DEFER state from spec collapses to "new node" in M1 (no commit-time review queue yet).

## Out of scope (this sprint)

- sqlite-vec ANN — semantic retrieval stays linear cosine over BLOB features (fine < 10k nodes)
- CLIP text/image encoder (`[clip]` extra) — M2
- split detection — M2
- decay/forget tuning — `forget()` exists; `decay()` is M2
- ConceptGraphs adapter — M2

## Definition of Done

- [x] `fusion.ingest_observation` merges by score, not blind new-node
- [x] quickstart: 5 detections (mug×2) → 4 nodes, mug merged (conf 0.90→0.95)
- [x] distinct-object test: kettle ≠ mug stay separate
- [x] determinism test: same stream twice → identical centroids
- [x] coverage **99%** on fusion (96% total); ruff clean; 25 tests green
- [x] CHANGELOG + version bump 0.0.1 → 0.1.0a1

**Built 2026-05-29.** Headline: incremental dedup — two sightings of one object converge to a single node with aggregated geometry/feature/label, the value ConceptGraphs' single-shot pipeline can't deliver.
