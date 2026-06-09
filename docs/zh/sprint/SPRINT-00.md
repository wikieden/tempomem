> 🌐 [English](../../en/sprint/SPRINT-00.md) · **中文**

# Sprint 00 · 骨架搭建 (M0)

**目标：** 在 Mac（无 CUDA）上全新安装后 `import spatialmem` 可正常运行，schema 与 dataclass 可往返序列化，50 行伪检测数据演示跑通。无真实感知，无融合智能——仅搭好框架。

**完成标准（来自 [roadmap](../03-ROADMAP.md)）：** `pytest -q` 全绿；在干净的 Python 3.11 venv 中 `import spatialmem` 正常；演示 notebook 生成可查询的 store。

## 任务分解

| ID | 任务 | 输出 | 依赖 | 估时 (CC) |
|---|---|---|---|---|
| T1 | `pyproject.toml` — hatch、依赖项（numpy/scipy/sqlite-vec/pillow/pydantic）、extras 桩、ruff+pyright 配置 | 可安装的骨架 | — | 20 min |
| T2 | 包骨架 `src/spatialmem/__init__.py`，含 `__all__` + version | `import spatialmem` 可用 | T1 | 10 min |
| T3 | `frame.py` — `Detection`、`Observation` 冻结 dataclass + JSON 往返 | 有类型的值对象 | T2 | 30 min |
| T4 | `persist/schema.sql` + `persist/migrations/001_init.py` — [SCHEMA.md](../../../spec/SCHEMA.md) 中的所有表 | 空 store 可创建 | T2 | 40 min |
| T5 | `persist/__init__.py` — 打开/创建、sqlite-vec + rtree 加载、迁移运行器、WAL | `SpatialMemory.open()` 返回活跃 store | T4 | 40 min |
| T6 | `store.py` — Node/Edge/Episode CRUD + `stats()` | 读写图行 | T5, T3 | 50 min |
| T7 | 融合**桩** — 每条观测 = 新节点（尚无合并逻辑） | 观测落地为节点 | T6 | 20 min |
| T8 | `query.py` 最小实现 — `recent()` + `spatial()`（R-tree），暂无语义 | 节点可检索 | T6 | 30 min |
| T9 | `serialize.py` — `format="json"` + 基础 `format="prompt"` | 图 → 文本 | T6 | 30 min |
| T10 | `cli.py` — `spatialmem inspect <file>` | 计数 + 样本节点 | T6 | 20 min |
| T11 | `examples/01_quickstart.py` — 输入合成厨房检测数据，输出查询结果 | 可运行演示 | T7, T8, T9 | 20 min |
| T12 | 测试：schema 往返、store CRUD、JSON 往返、演示即测试 | `pytest -q` 全绿 | T3–T9 | 40 min |
| T13 | CI：`.github/workflows/ci.yml` — lint + 单元测试，覆盖 3.10/3.11/3.12 × mac/linux | 绿色徽章 | T12 | 30 min |

总计约 6 h CC 时间。

## 执行顺序

```
T1 → T2 → T3 ─┐
              ├→ T4 → T5 → T6 → T7 ─┐
              │                T8 ──┤
              │                T9 ──┤→ T11 → T12 → T13
              │               T10 ──┘
```

T3 与 T4 可在 T2 之后并行执行。T7/T8/T9/T10 均从 T6 扇出。

## 本 sprint 不包含

- 真实融合评分（geom/iou/sem/label）→ M1
- 语义检索 / CLIP → M1
- ConceptGraphs 适配器 → M2
- LLM verbalizer → M2

## 完成定义

- [x] 在干净的 Python 3.12 venv（无 CUDA，仅 numpy 依赖）上 `pip install -e .`
- [x] `python examples/01_quickstart.py` 打印查询命中结果
- [x] `spatialmem inspect kitchen.smem` 显示节点计数
- [x] `pytest -q` 全绿（18 个测试）；覆盖率 **95%** 总体，核心模块 93–100%
- [ ] CI 在两种 OS 上均为绿色 — 首次推送后验证
- [x] B2 + B4 已解决（名称清晰，Apache-2.0）— 见 [05-OPEN.md](../05-OPEN.md)

**构建于 2026-05-29。** 偏差：M0 将特征向量以 BLOB float32 格式存储；`sqlite-vec` ANN 推迟至 M1（语义检索不在 M0 范围内）。默认依赖保持仅 numpy。

## 风险

| 风险 | 缓解措施 |
|---|---|
| sqlite-vec wheel 在某些 OS/Python 组合下缺失 | 在 T1 中提交依赖前先验证 wheel 矩阵 |
| R-tree（`pysqlite` rtree 模块）未编译进标准库 sqlite | 在 `open()` 时检测，抛出清晰错误；记录最低 sqlite 版本 |
| T7 期间范围蔓延至真实融合 | T7 明确为桩——合并逻辑在 M1，在评审中强制执行 |
