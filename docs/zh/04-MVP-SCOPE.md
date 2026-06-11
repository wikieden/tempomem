> 🌐 [English](../en/04-MVP-SCOPE.md) · **中文**

# 04 · MVP 范围 (M1)

## 目标

交付最小可用产品，让真实用户能够 `pip install` 后立即获得价值。推迟感知层。赢得 API。

## 范围内

1. **摄取：** 仅 `add_detections(List[Detection])`。`Detection` = `{label, conf, center_xyz, bbox3d, feature_vec, mask=None, ts}`。
2. **融合：** 确定性仲裁器——KNN 候选（质心距离 ≤ τ_d）、3D IoU ≥ τ_iou、CLIP-cos ≥ τ_s、标签兼容性检查。
3. **存储：** SQLite + sqlite-vec + R-tree。单个 `.smem` 文件。
4. **查询：**
   - `mem.query(text)` → 路由至空间 / 语义 / 时间检索器，返回排序后的节点（暂无 LLM——纯检索）。
   - `mem.spatial(near=(x,y,z), radius=r)` → R-tree 范围扫描。
   - `mem.semantic(text)` → CLIP 文本嵌入 + 节点特征 ANN。
   - `mem.recent(n=10)` → 时间扫描。
5. **序列化：** `mem.serialize(format="prompt", k_hops=2, root=None)` → 受 token 预算约束的缩进文本。
6. **CLI：** `tempomem inspect demo.smem`（计数、样本节点、schema 版本）。

## 范围外（MVP）

- 真实 RGB-D 感知（M2）
- LLM 语言化器（M2——MVP 返回原始节点；用户自行封装 LLM 调用）
- ROS 2 桥接（M3）
- Web 查看器（延伸——仅在时间允许时）
- 衰减 / 遗忘（M2）
- gRPC（M4）
- 多房间层级推断（M2）

## 演示脚本（发布时录制）

```python
import json, numpy as np
from tempomem import SpatialMemory, Detection

mem = SpatialMemory.open("kitchen.smem")

# Simulate 3 passes through a kitchen
for det in load_synthetic_kitchen_detections():
    mem.add_detections([det])
mem.commit()

# Query
hits = mem.query("mug near the sink")
print(hits[0].label, hits[0].center_xyz, hits[0].confidence)

# Prompt-ready text for any LLM
print(mem.serialize(format="prompt", k_hops=1, root=hits[0].id))
```

在笔记本电脑上运行时间 <30 秒，无需 GPU，无需网络。

## 验收测试（打标签 v0.1.0 前必须全部通过）

| ID | 检查项 | 工具 |
|---|---|---|
| A1 | 在 macOS arm64 + Linux x86_64（Python 3.10/3.11/3.12）上 `pip install tempomem` | CI 矩阵 |
| A2 | README 中的快速入门无需修改即可运行 | `pytest tests/test_readme.py` |
| A3 | 插入 10k 条检测记录 + 100 次查询在 2024 款 MacBook Air 上 <60 秒完成 | benchmark 门控 |
| A4 | v0.1.0 生成的 `.smem` 文件可被 v0.1.x 读取——schema 迁移已覆盖 | migration test |
| A5 | `mem.stats()` 数值与从头重新计算的基准一致 | invariant test |
| A6 | `fusion`、`store`、`query` 模块覆盖率 ≥ 75% | `pytest --cov` |
| A7 | 默认安装中无对 Torch / CUDA / ROS 的硬依赖 | `pip show tempomem` 依赖审计 |

## 风险登记册（MVP）

| 风险 | 严重程度 | 缓解措施 |
|---|---|---|
| 仲裁器错误合并不该合并的对象 | 高 | 确定性 + 配置中的阈值；黄金集回归测试；记录决策日志 |
| 小数据量下 sqlite-vec ANN 召回率低 | 中 | 混合策略：10k 向量以下线性扫描；以上切换至 ANN |
| v0.1.0 锁定后 API 频繁变更 | 高 | 保持 API 接口极小；将非核心内容标记为 `experimental` |
| ConceptGraphs 安装痛点渗入核心 | 高 | 硬性规则：ConceptGraphs 置于 `pip install tempomem[conceptgraphs]` 之后 |
