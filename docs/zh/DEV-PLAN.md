> 🌐 [English](../en/DEV-PLAN.md) · **中文**

# Development Plan — 统一执行追踪（系统级）

这是 SpatialRobot 工作区（三个包：`spatialmem` / `spatialmem-perception` / `spatialmem-brain`）的**唯一执行追踪文件**。

- **战略锚点（WHY / 季度 WHAT）：** [docs/VISION.md](../../../docs/zh/VISION.md)
  §8（P1/P2/P3）。该文件对定位、切入楔、风险、GTM 具有权威性。
- **里程碑视图（单包 M 编号）：** [03-ROADMAP.md](03-ROADMAP.md)
  （M0–M4）及其 M↔P 映射表。
- **本文件（下一步做什么、按什么顺序）：** 具体的、有序的任务列表，
  在一个硬约束条件下推进 VISION §8 P1：**当前开发机无 GPU。**

> **协调说明（2026-06-08）。** 本文件此前与系统愿景在下一步骤上存在分歧（旧文件写的是"先 B1'/B5/A1"；愿景写的是"先出评测证据"）。两者处于不同的粒度层级，且旧文件描述的还是拆分前的架构。此处将两者**合并为一个混合顺序**（修复已验证的缺陷 → 真实数据流 → 评测集 → 发布），挂靠在 VISION §8 P1 之下。冗余的 `docs/00-SYSTEM-VISION.md`（`docs/VISION.md` 的早期子集）已在同一次整理中删除。

## 当前状态（2026-06-08）

- `v0.1.0a1` 已打标签，仓库公开。**155 个测试**（core 120 / brain 19 /
  perception 16），core 安装仅需 numpy，core 上 `pyright` 干净（0 个错误）。
- **架构已完成拆分**（这是已完成的事，而非未来计划）：perception 位于配套的
  `spatialmem-perception`（`BoxDetectorAdapter` + `Detector3D` 接缝 +
  cam→world 几何变换 + `ImageEncoder`）；brain 位于 `spatialmem-brain`
  （`Brain` / `Reasoner` / `CosmosReasonVerbalizer`）。Core 只保留
  `PerceptionAdapter` / `Verbalizer` 协议。
- **M0 ✅ · M1 ✅ · M2 🟡** — 记忆深化 + 检索轨道已完成；M2 的录制 demo
  和习得式（GPU）感知是剩余缺口。

## 范围纪律 — 我们是一个记忆系统，而非感知系统

SpatialMem 存储并查询空间记忆。**识别不是我们的职责** —
正如 Mem0 不做语音转文字，我们不做目标检测。输入是调用方提供的检测结果（BYO perception）。影响如下：

- 具体的习得式感知（ConceptGraphs：SAM 2 + Grounding DINO + OpenCLIP）
  是**配套仓库中可选的 `[perception]` extra，受 GPU 门控，不在
  可工作 demo 的关键路径上**。它是一次*质量提升*，而不是 demo 的阻塞项。
  （这就是 VISION §8 P1 中"首个 ConceptGraphsAdapter"里程碑与无 GPU 约束
  的协调方式：它属于 **P1 范围但 ⛔ CUDA 阻塞**，与 M2 习得式感知行的状态相同。）
- 数据集（`SyntheticScene`、未来的 `ReplicaAdapter`）是**测试/基准夹具**，
  而非产品功能。它们为流水线提供输入，以证明记忆机制的正确性并进行基准测试。

## 关键重构 — M2 demo 不需要 GPU

M2 的退出条件是"流式传输一个 Replica 场景，提出 5 个问题，答对 4 个"。这
**不需要** ConceptGraphs（SAM + Grounding DINO + OpenCLIP → CUDA）。RGB-D
数据集（Replica、ScanNet、ARKitScenes）附带**真值实例分割 + 标签 + 位姿**。
一个*读取这些标注*的 `ReplicaAdapter` 能够以**零模型推理、零 GPU** 生成
`Detection`，并覆盖整条流水线 — `add_frame` → fusion → query → `answer` → eval。
因此：**先用 GT-adapter（无 GPU），再做习得式感知（需 GPU）。** 这一重构
正是消解"GPU 感知是 P1 还是延后？"分歧的关键所在。

## 统一的下一步顺序（混合 — 协调后的序列）

优先级解决了旧冲突：**修复已验证的缺陷 → 真实数据流 → 评测证据 → 可见度。**
全部四项均为纯 CPU，均在关键路径上；GPU 感知置于旁路，等硬件到位后再接入。

| # | 任务 | 包 | GPU | VISION §8 P1 行 | 顺序理由 |
|---|---|---|---|---|---|
| **1** | **检索上下文修复** — `Brain.ask()` 必须先 `query()` 一个相关子图，*再* `serialize`，而不是将整张图 token 截断后直接倾倒 | brain | no | "检索式上下文（先 query 过滤子图再 serialize）" / OQ-6 | 修复一个**已验证的代码缺陷**（见下文），否则任何大场景的评测数字都是谎言。它是评测有效性的前置条件 → 必须最先做。 |
| **2** | **B1' `ReplicaAdapter`** — 将 Replica 场景的 GT 实例掩码 + 深度 + 轨迹解析为现有 `DatasetSource` 格式 | core（+ fixture） | no | （为评测集提供数据 + 完成 M2 demo） | 一条真实数据流同时服务于评测集和录制 demo，一次解锁两个下游任务。 |
| **3** | **评测集 v0（自动化、确定性）** — 在现有 `bench.recall_at_k` 基础上扩展：增加 **`cited_node_ids` 格式合规率**、跨 episode 持久性、decay/forget 正确性。**无人工语义标注**（那是 P2 的事）。 | core | no | "自建空间记忆评测集 v0（自动化、确定性）" | 楔形策略的数字支撑。VISION 将此列为 P1 的头条；依赖 #1（有效检索），结合 #2（真实数据）效果最强。 |
| **4** | **B5 录制 demo + A1 发布** — viz HTML + asciinema 的流式循环；PyPI 发布 | core | no | （可见度） | 在有可信数字支撑之后再做可见度。A1 是**你的操作**（需要 PyPI token，不可逆）。 |

**不在关键路径上（GPU 门控，等 CUDA 到位后再做）：**

| 任务 | 包 | 状态 | VISION §8 P1 行 |
|---|---|---|---|
| `ConceptGraphsAdapter`（SAM 2 + Grounding DINO + OpenCLIP）置于 `[perception]` 后 | perception | ⛔ CUDA | "首个 PerceptionAdapter 具体实现" |
| `Cosmos3PerceptionAdapter`（Cosmos 3 边界框 + ego-pose → world `Detection`） | perception | ⛔ CUDA | （配套 backlog） |
| P3 与 ConceptGraphs demo 场景的性能对齐（recall ±10%） | perception | ⛔ CUDA | "性能 / 质量证明" |

**同样在 VISION §8 P1 中、可并行、无需 GPU（机会性折入）：**

- **Protocol v2 生命周期** — 为 `PerceptionAdapter` 增加 `__enter__`/`__exit__`
  （或 `close()`/`flush()`），以管理有状态编码器（CLIP）的生命周期（HIGH-3）。
- **性能门控** — 在现有"100 obs / 30 ms"基础上，增加"1000 obs / 100 节点
  `commit()` 延迟"的上界验收门控。
- **部署矩阵冒烟** — (a) 无 GPU 纯 numpy core-only 路径（BYO `Detection` →
  query）端到端；(b) +perception GPU 路径。两条路径均须通过。
- **双 Reasoner 后端** — Cosmos-Reason2（本地 RTX PRO 6000，OpenAI-compat
  `/v1`）**和** RoboBrain 均驱动 `Brain.ask()` → `Answer(cited_node_ids)`。
- **[P2 门控尖刺] RoboOS / InternRobotics 场景图能力调研** —
  截止 P1 末；其结论决定 P2 方向（互补 vs 轻量替代）。在结论出来之前，
  将 RoboOS 相关参考视为"未经核实的信号"（OQ-5）。

### 步骤 #1 背后的已验证缺陷

`spatialmem-brain/src/spatialmem_brain/brain.py:61` — 文档字符串写着
"Retrieve memory → reason → answer"，但函数体**并未**执行检索：

```python
def ask(self, question: str) -> Answer:
    """Retrieve memory → reason → answer."""
    context = self._mem.serialize(format="prompt", max_tokens=self._budget)
    return self._reasoner.reason(question, context)
```

它将**整张**图序列化，并按 `_budget` 做 token 截断。在大场景中，
相关对象可能因截断而丢失，导致召回率悄然下降，且**与记忆质量无关**。
修复方案：先从 `question` 出发 `query()`（或 `related()`）一个相关子图，
再序列化*该子图*，并让评测（#3）同时报告两种策略的结果 — 全图截断 vs
查询子图 — 从而**度量**而非假设改进效果（VISION §2.3，OQ-6）。

## M3 延伸（大部分无需 GPU）— P1 退出后

| ID | 任务 |
|---|---|
| D1 | ROS 2 桥接节点（订阅 RGB-D，发布 `/spatialmem/scene_graph`） |
| D2 | 在共享开放词汇数据集上运行 eMEM 基准测试套件（复用 `bench`） |
| D3 | 摘掉 alpha 标签：发布 `v0.1.0`，撰写发布公告，完成首篇外部集成文章 |
| D4 | 3D Web 查看器（升级 2D 版 `spatialmem viz`） |

## 序列图

```
   ┌─ 1 retrieval-context fix (brain, CPU) ─┐
   │                                        ▼
   └────────────────────────► 3 eval set v0 (CPU) ──► 4 B5 record + A1 publish ──► M3 (D-track) ──► v0.1.0
   2 B1' ReplicaAdapter (CPU) ──────────────┘
                                            │
   GPU perception (ConceptGraphs) ──────────┘  (folds in as a quality upgrade when CUDA lands)
```

步骤 1 和步骤 2 相互独立，可并行执行；两者均向步骤 3 输送结果。步骤 4 是
在数字就绪后的可见度行动。GPU 感知从不处于 demo 关键路径上 —
它在硬件可用后作为质量升级接入。

## P1 退出标准（来自 VISION §8）

一句**可复现**的话，证明楔形策略不是 PPT：
"memory holds X objects, survives restart, on a deterministic synthetic scene recall@k = R,
`cited_node_ids` format-compliance = F, 1000-obs `commit()` latency < T, pure-numpy
path smoke passes。"
**注意：recall@k + 格式合规率（可自动度量），不是人工标注的语义准确率** —
后者延至 P2。

## 红线

许可证红线（无例外）和工程红线（RFC 门控）的规范在
[VISION §9](../../../docs/zh/VISION.md) 中。约束本追踪文件的三条工程不变量：
core 保持 **numpy-only**；**fuse-before-persist**（新的 mutator 必须先调用
`_flush_pending()`）；测试 **network-free**（`ScriptedReasoner` / `HashEncoder`）。
