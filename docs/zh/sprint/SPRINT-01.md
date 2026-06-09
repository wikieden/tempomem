> 🌐 [English](../../en/sprint/SPRINT-01.md) · **中文**

# Sprint 01 · MVP — 真实融合 + 语义检索 (M1)

**目标：** 用 [FUSION-ARBITER.md](../../../spec/FUSION-ARBITER.md) 中的真实仲裁器替换 M0 的"每次观测创建一个节点"占位实现。同一物体的两次观测收敛为**一个**节点，并聚合几何/特征信息。这正是 ConceptGraphs 单次扫描流水线无法实现的价值：增量式、持久化的去重。

**退出条件：** quickstart 厨房场景（马克杯被观测两次）→ 马克杯是唯一一个节点，且 `n_obs=2`；确定性测试通过；`fusion` 模块覆盖率 ≥ 75%。

## 任务拆解

| ID | 任务 | 输出 | 估时 (CC) |
|---|---|---|---|
| F1 | `FusionConfig` 数据类 — 阈值 (τ_merge, τ_ambig, τ_obs, weights, dist_norm, search_dilation, centroid_alpha, conf_gain) | 可调配置 | 20 min |
| F2 | `store.candidates_near` — 获取 bbox（膨胀后）与某次观测 bbox 重叠的候选节点集合 | 候选集 | 25 min |
| F3 | `store.merge_observation_into_node` — 原子更新：置信度加权质心 EMA、bbox 并集、特征 EMA（重归一化）、标签分布、conf gain、t_last、n_obs++、node_obs 链接 | 合并事务 | 40 min |
| F4 | `fusion.score` — geom + iou3d + sem(cos) + label_compat，加权求和，全部截断至 [0,1] | 匹配评分器 | 35 min |
| F5 | `fusion.ingest_observation` — candidate→score→argmax→decide(merge/new/reject) | 真实仲裁器 | 30 min |
| F6 | `label_compat` — 精确匹配 + （后续可选 CLIP-text）+ 反义词=0；M1 使用精确/子串匹配，CLIP 钩子留存占位 | 标签评分 | 20 min |
| F7 | 通过 `SpatialMemory.open(config=...)` 将 `FusionConfig` 接入主接口 | 可配置化 | 15 min |
| F8 | 测试：去重（2 次观测→1 个节点）、不同物体保持独立、拒绝低置信度、确定性（相同流输入两次 → 节点数与质心完全相同） | 绿色通过 | 45 min |
| F9 | 更新 quickstart + SPRINT-00 说明；版本号 0.0.1→0.1.0a1 | demo 展示去重效果 | 15 min |

总计约 3.5 h CC。

## 评分公式（来自规范，M1 默认值）

```
s = w_g*s_geom + w_i*s_iou + w_s*s_sem + w_l*s_label
  w_g,w_i,w_s,w_l = 0.2, 0.2, 0.5, 0.1
  s_geom = max(0, 1 - dist/dist_norm_m)      dist_norm_m=0.50
  s_iou  = iou3d(obs.bbox, node.bbox)
  s_sem  = cos(obs.feat, node.feat_centroid)
  s_label= exact? max(weight,0.8) : substring? 0.5 : 0  (CLIP-text in M2)
decision: s>=τ_merge(0.62) merge | s>=τ_ambig(0.45) new | conf<τ_obs(0.30) reject | else new
```

注：M1 简化了 label_compat（尚无 CLIP-text 编码器——那是 `[clip]` extra，在 M2 接入）。规范中的 DEFER 状态在 M1 中折叠为"新节点"（暂无提交时审查队列）。

## 本 Sprint 范围外

- sqlite-vec ANN — 语义检索暂用 BLOB 特征的线性余弦（节点数 < 10k 时可接受）
- CLIP text/image 编码器（`[clip]` extra）— M2
- split detection — M2
- decay/forget 调优 — `forget()` 已存在；`decay()` 为 M2
- ConceptGraphs 适配器 — M2

## 完成定义

- [x] `fusion.ingest_observation` 按评分合并，而非盲目创建新节点
- [x] quickstart：5 次检测（mug×2）→ 4 个节点，mug 已合并（conf 0.90→0.95）
- [x] 不同物体测试：kettle ≠ mug 保持独立
- [x] 确定性测试：相同流输入两次 → 质心完全相同
- [x] `fusion` 覆盖率 **99%**（总体 96%）；ruff 无报警；25 个测试全部通过
- [x] CHANGELOG + 版本号 0.0.1 → 0.1.0a1

**构建于 2026-05-29。** 核心成果：增量去重 —— 同一物体的两次观测收敛为一个节点，并聚合几何/特征/标签信息，这正是 ConceptGraphs 单次扫描流水线无法实现的价值。
