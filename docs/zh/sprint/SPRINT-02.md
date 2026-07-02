> 🌐 [English](../../en/sprint/SPRINT-02.md) · **中文**

# Sprint 02 · 真实感知 + 规模化 + Verbalizer（M2）

**目标：** 闭合从原始 RGB-D 到问题回答的完整回路。当前 Chronotope 以 detections 为输入（BYO 感知）。M2 交付真实的 `add_frame(rgb, depth, pose)` 感知适配器，使用户可以流式传输数据集场景并直接提问，无需手动提供 detections。此外还包括使其具备生产形态的各个部件：规模化 ANN 检索、decay/forget，以及 LLM verbalizer。

**退出条件（来自 [roadmap](../03-ROADMAP.md)）：** 已录制"流式传输 Replica 场景，提问 5 个，答对 4 个"的 demo。

## 决策 2026-05-29 — 感知后端 = 选项 C（Protocol + ConceptGraphs 优先）

已完成 NVIDIA 开放模型调研（详见 [05-OPEN.md](../05-OPEN.md) P1/P3 日志）。结论：**没有 NVIDIA 模型适合作为捆绑默认后端** —— FoundationStereo/Pose/RADIO 为非商业授权（NSCL），Isaac Perceptor 为专有软件，NV-CLIP 仅限企业使用，C-RADIO 商业可用但无原生文本编码器（破坏我们的文本查询 ANN）。因此：**选项 C** —— 定义 `PerceptionAdapter` Protocol（P0），以 **ConceptGraphs（SAM + Grounding DINO + OpenCLIP，Apache/MIT）** 作为第一个具体适配器，固定 commit 并软 fork。NVIDIA 部件后续再议：nvblox 几何（M3/M4）、C-RADIO `[radio]` 编码器（M2 之后，需要文本空间原型验证）、Cosmos-Reason2（在文档中作为具名 BYO verbalizer）。**P 系列任务已解锁。**

`add_frame` 需要一个能将 RGB-D 转换为开放词汇表 3D detections（标签 + 3D 边界框 + 特征）的后端。已考虑的选项：

| 选项 | 后端 | 许可证 | 安装成本 | 备注 |
|---|---|---|---|---|
| **A** | ConceptGraphs（SAM + open-CLIP），固定 commit | Apache/MIT-ish | conda + torch + faiss | 学术性，单次推理；我们的增量融合是新的价值所在 |
| **B** | NVIDIA 技术栈（nvblox 几何 + RADIO/NV-CLIP 特征 + 开放词汇表分割） | ⚠️ 许多 NVIDIA 模型许可证为非商业 / 仅限评估 | 仅支持 CUDA | 研究中，正在评估适配性与许可证风险 |
| **C** | 混合方案：我们的适配器 Protocol，优先交付 ConceptGraphs，NVIDIA 作为可选 extra | 按组件决定 | 混合 | 最灵活，工作量最大 |

→ **已决策：C**（见上）。`PerceptionAdapter` Protocol + ConceptGraphs 优先。

## 任务拆分

### 不受门控 — 立即启动

| ID | 任务 | 产出 | 估时（CC） |
|---|---|---|---|
| V1 | sqlite-vec 接线：`[vec]` extra 背后的 `obs_features`/`node_features` vec0 表；迁移 002；`query.semantic_vec` 中的 ANN 路径，扩展不存在时回退到线性 | >10k 节点时的 ANN | 60 分钟 |
| V2 | `decay(half_life_days, min_conf)` —— 按年龄进行置信度衰减，低于下限时剪枝；`forget()` 已存在 | 内存卫生 API | 40 分钟 |
| V3 | LLM verbalizer：`Verbalizer` Protocol + `answer(query, k)` = 检索 → serialize(prompt) → BYO LLM（OpenAI/Anthropic/Ollama）；不捆绑密钥 | 自然语言回答，而非仅节点 | 50 分钟 |
| V4 | 分裂 detection：一个节点漂移为两个聚类 → 分裂（从 M1 延期） | 融合正确性 | 45 分钟 |
| V5 | 评估工具：场景 → 摄入 → N 个脚本化查询 → recall@k vs 真实标注；可复用于 demo 指标 | 基准数值 | 50 分钟 |
| V6 | `[clip]` CI 任务：安装 extra，对真实 torch 验证 `OpenClipEncoder` 文本嵌入的 shape/dim（M1 中已构建，CI 中未测试） | clip 通道绿色 | 25 分钟 |

### 受感知后端决策门控

| ID | 任务 | 产出 | 依赖 |
|---|---|---|---|
| P0 | `PerceptionAdapter` Protocol：`process_frame(rgb, depth, pose) -> list[Detection]` | 后端无关的接缝 | — |
| P1 | 第一个具体适配器（A/B/C 决策赢家），固定 commit，打包为 extra | `add_frame` 可用 | P0 + 决策 |
| P2 | `TempoMem.add_frame(rgb, depth, pose)` 通过适配器接入 → 融合 | RGB-D 摄入 | P1 |
| P3 | 在后端自带的 demo 场景上对适配器进行基准测试；目标物体召回率与参考值偏差在 ±10% 以内 | 一致性证明 | P1, V5 |

## 超出范围（M3+）

- ROS 2 桥接 → M3
- 真实机器人 / AR 多日持久化 → M3
- eMEM 基准测试 → M3
- gRPC 外观、Mem0 适配器 → M4

## 完成定义

- [x] `PerceptionAdapter` Protocol + `add_frame` 摄入 RGB-D **(P0/P2 — 桩测试通过)**；ConceptGraphs 具体适配器（P1）等待 CUDA 开发机
- [x] 分裂 detection —— `resplit()` **(V4 — 已于 2026-05-29 构建)**
- [x] sqlite-vec ANN 路径绿色（含线性回退）；`[vec]` extra 可安装 **(V1 — 已于 2026-05-29 构建)**
- [x] `decay()` + `answer()`（verbalizer）API 已落地并有测试 **(V2, V3 — 已于 2026-05-29 构建)**
- [x] 评估工具报告 recall@k **(V5 — `tempomem.bench.recall_at_k`)**
- [x] `[clip]` CI 通道绿色 **(V6 — 烟雾测试，不下载权重)**
- [ ] Demo 已录制：流式传输 Replica 场景 → 5 个问题 → ≥4 个正确
- [ ] 新模块覆盖率 ≥ 80%；ruff 检查通过

## 风险

| 风险 | 缓解措施 |
|---|---|
| NVIDIA 模型许可证为非商业 → 无法作为默认交付 | Protocol 接缝（P0）保持后端可替换；默认使用 Apache 兼容的 ConceptGraphs，NVIDIA 作为可选项 |
| 感知后端将 conda/CUDA 拖入安装过程 | 隔离在 `[perception]`/`[nvidia]` extras 中；核心保持纯 numpy |
| sqlite-vec wheel 在各 OS/Python 版本下存在缺口 | 线性余弦回退已存在；ANN 是机会性的，非必需 |
| Verbalizer 需要密钥/网络 | 仅 BYO 模型；无捆绑依赖，可通过 Ollama 走离线路径 |
