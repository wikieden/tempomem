> 🌐 [English](../en/03-ROADMAP.md) · **中文**

# 03 · 路线图

里程碑以时间为边界，而非功能为边界。若某个里程碑在范围上出现延误，应削减范围，而非推迟时间。

**状态（2026-06-08）：** M0 ✅ · M1 ✅（`v0.1.0a1`）· M2 🟡 进行中（V-track + perception 接缝已完成；具体的 RGB-D 适配器 + 录制演示受阻于 CUDA 机器）。工作区共 155 个测试（core 120 / brain 19 / perception 16），core 安装仅需 numpy。各里程碑冲刺细则：[SPRINT-00](sprint/SPRINT-00.md) · [SPRINT-01](sprint/SPRINT-01.md) · [SPRINT-02](sprint/SPRINT-02.md)。

图例：✅ 已完成 · 🟡 部分完成 · ⛔ 受阻 · ⬜ 未开始。

## M↔P 映射（里程碑视角 ↔ 系统愿景）

本文件使用单包 **M 编号**（M0–M4）。系统愿景文件
[docs/VISION.md](../../../docs/zh/VISION.md) §8 使用按季度的 **P 编号**（P1–P3），
横跨三个包。两者是同一计划的不同视角——下表进行了相互映射。**VISION §8 是战略层面的唯一可信来源；有序的、任务级别的执行细节见 [DEV-PLAN.md](DEV-PLAN.md)。**

| M（本文件） | P（VISION §8） | 备注 |
|---|---|---|
| M0 骨架 ✅ · M1 MVP ✅ | —（P 框架之前的基础） | 在 P 框架形成前已完成；即 numpy-only 的核心 SDK。 |
| M2 真实感知 + 规模 + Verbalizer 🟡 | **P1 · 地基** | P1 = 在*无 GPU* 条件下完成 M2 演示 + 补充 M 视角未命名的证据层（eval set v0、检索上下文修复、性能门控、部署矩阵冒烟测试、双 Reasoner）。 |
| M3 真实机器人演示 + ROS 2 ⬜ | **P2 · 集成验证** | 外部开放友好机体 + LeRobot + RoboOS/InternRobotics spike 结论。 |
| M4 加固 + Mem0 适配器 ⬜ | **P3 · 生态位锁定** | 稳定序列化协议 + 默认采用 + 商业基线（OQ-1/OQ-2）。 |

## M0 · 骨架 ✅

- ✅ 仓库、许可证（Apache-2.0）、CI 矩阵（3.10–3.12 × mac/linux）、包结构
- ✅ `Detection` / `Observation` / `Node` / `Edge` 值对象 + JSON 往返序列化
- ✅ SQLite schema + 只向前迁移（sqlite-vec 接线移至 M2——见备注）
- ✅ `pip install tempomem` 可在 Mac（无 CUDA）上运行，无需真实感知
- ✅ `examples/01_quickstart.py` —— 假 detection 输入，查询输出

**退出条件已达成：** `pytest -q` 绿色通过；在干净的 Python 3.11 venv 中 `import tempomem` 可正常执行。

> 偏差说明：M0 以 BLOB 存储向量（numpy-only）；sqlite-vec ANN 在 M2（V1）通过 `[vec]` extra 加载，并提供线性回退。已记录于 [05-OPEN](05-OPEN.md)。

## M1 · MVP · "Detections-In" SDK ✅（`v0.1.0a1`）

- ✅ `add_detections([Detection(...)])` 摄取（BYO perception）
- ✅ 融合仲裁器 v1：候选搜索 + 3D IoU + 特征余弦 + 标签打分，确定性合并/新增/拒绝——增量去重（这是 ConceptGraphs 单次流水线所欠缺的核心价值）
- ✅ `query(...)` → 空间 + 时序 + 关键字检索，返回节点
- ✅ **语义查询**（从 M2 提前）：BYO `Encoder` 协议 + `OpenClipEncoder`（`[clip]` extra），对节点特征做余弦计算
- ✅ `serialize(format="prompt"|"json")` → 紧凑图文本
- ✅ Web 查看器（只读）—— `tempomem viz` 导出自包含 HTML 场景（构建于 M2）
- ⬜ Replica/ScanNet RGB-D 演示——需要 M2 perception 适配器（CUDA）

**退出条件：** README 快速入门可在干净机器上原样运行（`examples/01` + `02`）。Replica 演示并入 M2。

## M2 · 真实感知 + 规模 + Verbalizer 🟡

已完成（无需 GPU）：
- ✅ V1 sqlite-vec ANN 索引（`[vec]`），写入时维护，线性回退
- ✅ V2 `decay(half_life_days, min_conf)` + `forget()` —— 记忆清理
- ✅ V3 LLM verbalizer：`Verbalizer` 协议 + `answer()`（BYO OpenAI/Anthropic/Ollama）
- ✅ V4 分裂检测——`resplit()`（对成员观测做确定性 2-means 分割）
- ✅ V5 评估框架——`bench.recall_at_k`
- ✅ V6 `[clip]` + `[vec]` CI 通道
- ✅ P0/P2 `PerceptionAdapter` 协议 + `add_frame(rgb, depth, pose)` 接缝（桩测试已覆盖）

记忆深化 + 检索（V-track 之后交付，无需 GPU）：
- ✅ 层级/房间——`define_region(...)` + `contents(region)`（物体嵌套于区域之下）
- ✅ 空间关系——`relate()` 推断 `near`/`on`/`under` 边 + `related(node, rel=)`
- ✅ 原位修正——`update(node_id, ...)` + `history(node_id)` 观测轨迹
- ✅ 关系感知 `serialize(format="prompt")` —— 为每个节点附加关系边
- ✅ 多会话合并——`merge(other_smem)` 通过融合折叠另一个 store 的对象
- ✅ 关系型自然语言查询——`query("what's on the table")` 遍历关系边
- ✅ 变更检测——`moved()` / `changes(since_ts)` / `stale(before_ts)`
- ✅ Token 预算控制的 `serialize(format="prompt", max_tokens=N)` —— 有界 LLM payload
- ✅ `consolidate()` + `salient(n)` —— 合并遗漏的重复项，按近期度·置信度·证据排序
- ✅ 数据集流式处理——`DatasetSource` + `stream(mem, source)` + `SyntheticScene`（+ `bench.recall_at_k`）

受阻于 CUDA 开发机：
- ⛔ P1 具体的 `ConceptGraphsAdapter`（SAM + Grounding DINO + OpenCLIP），置于 `[perception]` extra 之后
- ⛔ P3 在 ConceptGraphs 演示场景上的基准对比（物体召回率误差 ±10%）
- ⛔ 录制演示："流式传输 Replica 场景，提出 5 个问题，答对 4 个"

**退出条件（不变）：** 录制的 Replica 演示。协议接缝 + 接线已就绪，因此一旦有 GPU，P1 即可即插即用。

## M3 · 真实机器人演示 + ROS 2 适配器 ⬜

- ROS 2 桥接节点（订阅 RGB-D topic，发布 `/tempomem/scene_graph`）
- 公开演示：移动机器人或 AR 会话，多日持久化
- 在共享数据集上与 eMEM 对比基准测试（开放词汇查询）——复用 `bench.recall_at_k`
- 首个外部集成技术文章（目标：Brain2Robot / L3-planner 参考循环）
- v0.1.0 PyPI 发布 + 发布帖
- 3D Web 查看器（M2 的 `tempomem viz` 是 2D 俯视只读起点）

**退出条件：** GitHub ★ 达 100，Discord 中 3 位外部用户，1 篇被引用的集成文章。

## M4 · 加固 + Mem0 适配器 ⬜（Q3）

- gRPC 服务器门面，支持多进程 / 语言无关使用
- Mem0 适配器 shim（`Mem0SpatialBackend`）
- Vision Pro / Quest 场景网格摄取适配器（草图）
- nvblox（Apache-2.0）几何基底适配器——可选，来自 NVIDIA 调研
- 托管层原型（托管 `.smem` 存储）——仅在社区有需求时推进

**退出条件：** v0.2.0，至少一个生产用户。

## 剪切项（12 个月计划内故意不做的事）

- 训练自有 VLM 或 SLAM
- 闭源纯云端层
- 动作 / 规划层（由 L3 planner 等消费端负责）
- 多智能体 / 共享地图联邦（未来有趣，现在是干扰）

## 跟踪

每个里程碑在 `docs/sprint/SPRINT-NN.md` 中细化分解。有序执行计划（GPU 感知、GT 适配器优先）见 [DEV-PLAN.md](DEV-PLAN.md)。已解决的设计问题记录于 [05-OPEN.md](05-OPEN.md)；已交付的接口由 [CHANGELOG.md](../../CHANGELOG.md) 和 [spec/API.md](../../spec/API.md) 跟踪。
