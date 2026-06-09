> 🌐 [English](../en/02-ARCHITECTURE.md) · **中文**

# 02 · 架构

## 分层栈

```
┌──────────────────────────────────────────────────────────────┐
│  L5  Agent / LLM           ← 消费者（LangChain、Mem0、原生） │
├──────────────────────────────────────────────────────────────┤
│  L4  Query Router          自然语言 → (语义 | 空间 | 时序)   │
│       └─ serializer        图 → 提示文本                     │
├──────────────────────────────────────────────────────────────┤
│  L3  Memory Store          对象 · 地点 · 房间 · 事件         │
│       ├─ Fusion Arbiter    去重 / 合并 / 拆分                │
│       └─ Decay / Forget    置信度 + TTL                      │
├──────────────────────────────────────────────────────────────┤
│  L2  Persistence           SQLite + sqlite-vec + R-tree      │
├──────────────────────────────────────────────────────────────┤
│  L1  Ingest Adapters       conceptgraphs · hydra · custom    │
├──────────────────────────────────────────────────────────────┤
│  L0  Perception (external) RGB-D · pose · CLIP/SigLIP feats  │
└──────────────────────────────────────────────────────────────┘
```

每一层都是一个独立的 Python 模块，并配有文档化的 Protocol。替换其中一层不会产生级联影响。

## 核心概念

- **Frame** — 原子摄入单元：`(rgb, depth, intrinsics, pose, ts)` *或* `(detections, ts)`。
- **Observation** — 单帧对单个对象的观测结果：2D 掩码 / 3D 点云 + 特征 + 标签分布 + 边界框 + 时间戳。
- **Node** — 稳定的场景图实体。类型：`Object`、`Place`、`Room`、`Floor`。保存滚动聚合的几何信息 + 特征中心 + 观测 id 历史记录。
- **Edge** — 有类型的关系：`on`、`inside`、`near`、`part_of`、`same_room_as`、`temporal_before`。Edge 携带置信度 ∈ [0,1] 及 last_seen 时间戳。
- **Episode** — 共享同一 session id 的连续帧序列。Episode 可提交并支持回放。

## 数据流（正常路径）

```
add_frame(rgb,d,pose) ──▶ Ingest Adapter (ConceptGraphs)
                              │
                              ▼
                       observations[]（每帧）
                              │
                              ▼
                       Fusion Arbiter
                       ├─ 候选匹配（质心 KNN + IoU3D）
                       ├─ 评分（几何 + CLIP cos + 标签一致性）
                       ├─ 决策：merge | new node | reject
                       ▼
                       Memory Store 变更事务
                              │
                              ▼
                       持久化写入（每次 commit 一次 fsync）
```

查询路径：

```
query("where is the red mug?") ──▶ Query Router
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        空间意图？               语义意图？               时序意图？
        (near/in/under)         ("red mug")           ("last seen")
                │                      │                      │
                ▼                      ▼                      ▼
        R-tree 范围扫描       sqlite-vec ANN          时间戳索引
                └──────────────────────┼──────────────────────┘
                                       ▼
                              候选节点集合
                                       │
                                       ▼
                          k-hop 子图提取
                                       │
                                       ▼
                       LLM verbalizer  ──▶  Answer + 引用节点 id
```

## 模块映射（规划中）

| 模块 | 职责 | v0 代码行数预算 |
|---|---|---|
| `spatialmem.frame` | Frame / Observation 数据类 | <200 |
| `spatialmem.adapters.conceptgraphs` | 通过 ConceptGraphs 后端将 RGB-D 转换为 observations | <600 |
| `spatialmem.adapters.detections` | 预检测 observations（自带感知） | <200 |
| `spatialmem.fusion` | Arbiter + 匹配评分 | <500 |
| `spatialmem.store` | 基于 SQLite 的 Node / Edge / Episode CRUD | <600 |
| `spatialmem.persist` | Schema 迁移、sqlite-vec、R-tree | <400 |
| `spatialmem.query` | Router + 检索器 + verbalizer | <600 |
| `spatialmem.serialize` | 图 → 提示文本 / JSON / DOT | <300 |
| `spatialmem.llm` | 自带 LLM Protocol + 轻量封装 | <200 |
| `spatialmem.viz` | Web 查看器（独立可选依赖） | <800 |
| `spatialmem.bridges.ros2` | 可选 ROS 2 节点 | <400 |
| `spatialmem.bridges.mem0` | 可选 Mem0 空间后端 shim | <200 |

核心总计约 3.5k 行代码，刻意保持极小规模。

## 持久化布局（单个 `.smem` 文件）

```
SQLite file
├── meta(table)                schema version, created_at, embedding_dim
├── episodes(table)            id, session, start_ts, end_ts, label
├── observations(table)        id, episode, ts, pose, bbox, label, conf
├── obs_features (vec)         sqlite-vec virtual table (obs_id → vec)
├── nodes(table)               id, type, label, centroid, bbox, conf, t_first, t_last
├── node_features (vec)        sqlite-vec (node_id → centroid embedding)
├── node_geom (rtree)          node_id → (xmin,xmax,ymin,ymax,zmin,zmax)
├── edges(table)               id, src, dst, type, conf, t_last
└── node_obs(table)            node_id ↔ observation_id (many:many)
```

单文件 = 可移植、可 diff 的演示状态，易于备份。v0 不依赖任何外部服务。

## 线程与并发

- v0：单写者，异步摄入队列。`add_frame` 立即返回；arbiter 在 worker 中运行；`commit()` 进行 join。
- 查询路径持有只读快照（SQLite WAL）。
- v1：通过 gRPC server 外观支持多进程。

## 硬依赖（v0）

`numpy`、`scipy.spatial`、`sqlite-vec`、`pillow`、`pydantic`。CLIP/SigLIP 通过 `[clip]` extra 可选启用。ConceptGraphs 适配器通过 `[conceptgraphs]` extra 安装，并固定到已测试的提交版本。

其他所有内容（Torch、ROS、CUDA）均为**可选 extra**。在无 CUDA 的 Mac 上执行 `pip install spatialmem`，使用 `detections` 适配器必须能端到端正常工作。

## 可观测性

- 结构化 JSON 日志（`spatialmem.events`），每次融合决策对应一条事件。
- `mem.stats()` 返回 `{n_nodes, n_edges, n_obs, last_commit_ms, store_bytes}`。
- `mem.dump(path)` 将完整图导出为 JSON 以供调试。
