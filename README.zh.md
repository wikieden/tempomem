> 🌐 [English](README.md) · **中文**

# Chronotope

[![CI](https://github.com/wikieden/tempomem/actions/workflows/ci.yml/badge.svg)](https://github.com/wikieden/tempomem/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tempomem)](https://pypi.org/project/tempomem/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

AI agent 的空间记忆层。只需一次 pip 安装，即可将 RGB-D / 点云转换为持久化、可查询、LLM 原生的 3D 场景图。

> **定位：** 面向 3D 空间的 Mem0。ConceptGraphs / Hydra 负责感知；Mem0 负责文本记忆；而目前没有任何工具能将感知层与持久化、Agent 可查询的空间记忆打通。我们填补的正是这个空缺。

## 快速入门

```python
from tempomem import SpatialMemory, Detection

# 输入检测结果（自带感知模块）：提供标签 + 3D 包围盒 + 特征向量。
mem = SpatialMemory.open("kitchen.smem", embedding_dim=512)
mem.add_detections([
    Detection("mug", feat, center_xyz=(1.2, 0.3, 0.9),
              bbox_min=(1.15, 0.25, 0.85), bbox_max=(1.25, 0.35, 0.95)),
])
mem.commit()                       # 执行融合仲裁器 —— 增量去重

mem.recent(n=5)                                    # 时序查询
mem.spatial(near=(1.0, 0.0, 1.0), radius=2.0)      # 空间查询
prompt = mem.serialize(format="prompt")            # 图 -> 紧凑 LLM 文本

# 自然语言语义搜索与问答需要 encoder/verbalizer：
mem = SpatialMemory.open("kitchen.smem", embedding_dim=512,
                         encoder=my_clip, verbalizer=my_llm)
mem.semantic("coffee mug")                         # 基于节点特征的余弦相似度检索
mem.answer("where is the mug?")                    # 检索 -> 提示词 -> 自带 LLM

# 流式 RGB-D（需要 PerceptionAdapter；ConceptGraphs adapter 开发中）：
# mem = SpatialMemory.open(..., adapter=MyAdapter())
# mem.add_frame(rgb, depth, pose); mem.commit()
```

无需 GPU 即可运行：`python examples/01_quickstart.py` 和 `examples/02_query_and_answer.py`。

## 当前可用功能

| 能力 | 状态 |
|---|---|
| 单文件 `.smem` SQLite 存储，持久化/重新打开 | ✅ |
| `add_detections` + 增量融合（去重、合并、拒绝） | ✅ |
| 空间 / 时序 / 关键词查询 | ✅ |
| 通过自带 `Encoder` 进行语义查询（`[clip]` 中的 `OpenClipEncoder`） | ✅ |
| sqlite-vec ANN 索引（`[vec]`），线性回退 | ✅ |
| 通过自带 `Verbalizer`（OpenAI / Anthropic / Ollama）执行 `answer()` | ✅ |
| `decay()` + `forget()` + `resplit()` 记忆清理 | ✅ |
| `consolidate()` + `salient()` 记忆整合 | ✅ |
| 层级 / 房间 —— `define_region()` + `contents()` | ✅ |
| 空间关系 —— `relate()` + `related()`（`near`/`on`/`under`） | ✅ |
| 关系型自然语言查询 —— `query("what's on the table")` 遍历边 | ✅ |
| `update()` + `history()`（Mem0 风格的更正 + 观测记录） | ✅ |
| 多会话合并 —— `merge(other.smem)` | ✅ |
| 变化检测 —— `moved()` / `changes()` / `stale()` | ✅ |
| 用于 LLM 交接的 `serialize(format="prompt"/"json")` | ✅ |
| Token 预算提示词 —— `serialize(format="prompt", max_tokens=N)` | ✅ |
| `recall_at_k` 评估框架 | ✅ |
| 只读 HTML 查看器 —— `tempomem viz store.smem -o scene.html` | ✅ |
| 通过 `PerceptionAdapter` 协议支持 RGB-D `add_frame` | ✅ 接口已就绪；ConceptGraphs adapter 开发中（需 CUDA） |

核心安装**仅依赖 numpy**。重型后端位于扩展选项之后：`[clip]`、`[vec]`、`[perception]`。

## 仓库结构

```
docs/        产品与工程决策（建议优先阅读）
spec/        规范性 API / schema 规格文档
src/tempomem/   Python 包（库）
examples/    可运行示例（真实数据 + 仿真数据）
tests/
```

## 当前状态

Pre-alpha，公开设计阶段。建议按序阅读：

- [docs/00-VISION.md](docs/zh/00-VISION.md) —— 是什么 & 为什么
- [docs/01-POSITIONING.md](docs/zh/01-POSITIONING.md) —— 竞品分析与差异化切入点
- [docs/02-ARCHITECTURE.md](docs/zh/02-ARCHITECTURE.md) —— 分层结构与数据流
- [docs/03-ROADMAP.md](docs/zh/03-ROADMAP.md) —— 里程碑规划
- [docs/04-MVP-SCOPE.md](docs/zh/04-MVP-SCOPE.md) —— 首个可交付成果
- [spec/API.md](spec/API.md) —— Python SDK 契约
- [spec/SCHEMA.md](spec/SCHEMA.md) —— 场景图 schema
- [spec/FUSION-ARBITER.md](spec/FUSION-ARBITER.md) —— 节点合并算法
- [spec/QUERY-ROUTER.md](spec/QUERY-ROUTER.md) —— 查询 → 检索路由
- [spec/ENGINEERING.md](spec/ENGINEERING.md) —— 编码规范 / CI

## 许可证

Apache-2.0 —— 参见 [LICENSE](LICENSE)。
