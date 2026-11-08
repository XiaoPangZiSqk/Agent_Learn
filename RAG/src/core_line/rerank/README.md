# Rerank — 交叉编码精排模块

## 项目简介

本模块是 RAG（检索增强生成）流水线中的 **重排序（Rerank）** 环节，负责对多路召回 + RRF 融合后的候选文档进行二次精准排序。通过交叉编码器（Cross-Encoder）对 `(query, text)` 成对联合编码并打相关性分，精度显著高于双塔召回模型，最终输出「原始排序 vs 精排排序」对比结果，供生成层选取 Top-N。

模块隶属于 `core_line` 核心流水线，位于召回融合层之后、生成层之前。

---

## 主要功能特性

| 特性 | 说明 |
|------|------|
| **交叉编码精排** | 对每条候选的 `text` 与查询组成 `(query, text)` 对，联合编码打相关性分 |
| **可插拔打分器** | `CrossEncoderReranker` 只负责编排；`BaiLianRerankScorer` 只负责打分。自定义打分器只需提供 `score(query, texts) -> list[float]` |
| **批量可配** | `batch_size` 控制单次请求文档数（qwen3-rerank 单次上限 500 条），支持构造参数与环境变量切换 |
| **异常兜底** | 空候选返回空结果；非法入参显式抛错；API 限流/5xx 指数退避重试后仍失败抛 `RuntimeError` |
| **顺序保证** | 打分返回按 `index` 还原顺序，保证与输入候选一一对应；条数不一致直接报错，不静默对齐 |
| **对比输出** | 同时保留 `original_rank` / `original_score`（RRF 融合分）与 `rerank_rank` / `rerank_score`（交叉编码分），便于观察位次变化 |

---

## 整体工作流程

```
多路召回候选（HybridRetriever.search）
        │
        ▼
  校验并归一化候选（_normalize）
  · 校验 text 非空、类型正确
  · 抽取 id / metadata / score / channels
  · 记录 original_rank（输入顺序，从 1 开始）
        │
        ▼
  交叉编码器成对打分（BaiLianRerankScorer.score）
  · (query, text) 对批量送入 qwen3-rerank
  · 分批请求 + 指数退避重试
  · 按 index 还原打分顺序
        │
        ▼
  按 rerank_score 降序重排
  · 同分按 original_rank 升序（稳定排序）
  · 赋 rerank_rank（从 1 开始）
        │
        ▼
  top_k 截断（打分与排序之后，全体候选均已打分）
        │
        ▼
  输出对比结果
  {
    "query":   str,
    "model":   str,
    "count":   int,       # 参与精排的候选总数（top_k 截断前）
    "results": [          # 按 rerank_score 降序
      { "id", "text", "metadata",
        "original_rank", "rerank_rank",
        "original_score", "rerank_score",
        "channels" },
      ...
    ]
  }
```

---

## 安装与配置

### 依赖安装

```bash
pip install dashscope python-dotenv
```

> 若不使用默认百炼打分器（注入自定义 scorer），可跳过 `dashscope` 安装。

### API Key 配置

默认打分器走百炼（DashScope）TextReRank 接口，需要 API Key。通过以下任一方式提供：

**方式一：环境变量**

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
# 或
export BAILIAN_RERANK_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

**方式二：构造参数**

```python
reranker = CrossEncoderReranker(api_key="sk-xxxxxxxxxxxxxxxx")
```

### 可选环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `BAILIAN_RERANK_MODEL` | 重排模型名 | `qwen3-rerank` |
| `DASHSCOPE_API_KEY` | 百炼 API Key | — |
| `BAILIAN_RERANK_API_KEY` | 百炼 API Key（优先级更高） | — |

### 模型限制（qwen3-rerank）

- 单条文档上限 **4,000 token**，超长服务端返回 400（不会截断）
- 单次请求上限 **500 条文档 / 120,000 token**
- 可用模型名：`qwen3-rerank` / `qwen3.7-text-rerank` / `gte-rerank-v2`（`gte-rerank` 已于 2026-05-30 下线）

---

## 基本使用示例

### 完整流水线：召回 + 融合 + 精排

```python
from core_line.retrieve.retriever import HybridRetriever, RecallChannel
from core_line.rerank.reranker import CrossEncoderReranker

# 1. 构建混合召回器（多路召回 + RRF 融合）
hybrid = HybridRetriever([
    RecallChannel("vector", store),
    RecallChannel("fts5", fts5),
])

# 2. 构建精排器
reranker = CrossEncoderReranker(model="qwen3-rerank", batch_size=50)

# 3. 召回 + 融合
fused = hybrid.search("如何配置 HNSW 参数", top_k=20)

# 4. 精排
result = reranker.rerank("如何配置 HNSW 参数", fused, top_k=5)

# 5. 查看结果
for item in result["results"]:
    print(item["rerank_rank"], item["original_rank"], item["rerank_score"], item["id"])
```

### 注入自定义打分器

```python
from core_line.rerank.reranker import CrossEncoderReranker

class MyScorer:
    model = "my-local-reranker"

    def score(self, query: str, texts: list[str]) -> list[float]:
        # 返回与 texts 等长的浮点列表
        return [0.9, 0.7, 0.5]

reranker = CrossEncoderReranker(scorer=MyScorer())
result = reranker.rerank("查询文本", candidates, top_k=3)
```

### 仅打分器单独使用

```python
from core_line.rerank.reranker import BaiLianRerankScorer

scorer = BaiLianRerankScorer(batch_size=100, max_retries=5)
scores = scorer.score("如何配置 HNSW 参数", ["HNSW 参数调优指南", "向量索引原理"])
print(scores)  # [0.92, 0.41]
```

---

## 输出结构

```json
{
  "query": "如何配置 HNSW 参数",
  "model": "qwen3-rerank",
  "count": 20,
  "results": [
    {
      "id": "doc_001::chunk_0",
      "text": "HNSW 参数包括 ef_construction、M 等...",
      "metadata": {"doc_id": "doc_001", "chunk_id": "chunk_0", "heading_path": ["索引", "HNSW"]},
      "original_rank": 3,
      "rerank_rank": 1,
      "original_score": 0.0317,
      "rerank_score": 0.9342,
      "channels": ["vector", "fts5"]
    }
  ]
}
```

> **注意**：`rerank_score`（相关性分）与 `original_score`（RRF 融合分）量纲不同，禁止互相比较或相加；对比只看 `original_rank` 与 `rerank_rank` 的位次变化。

---

## 项目目录结构

```
RAG/src/core_line/
├── chunk/              # 文本切块
├── embedding/          # 向量嵌入
├── generate/           # 生成层
├── load_file/          # 文件加载
├── retrieve/           # 多路召回 + RRF 融合
├── vector_store/       # 向量存储
└── rerank/             # ← 本模块（交叉编码精排）
    ├── reranker.py     # 核心实现（CrossEncoderReranker + BaiLianRerankScorer）
    └── README.md       # 本文件
```

### 核心类一览

| 类 / 协议 | 职责 |
|-----------|------|
| `RerankScorer` | 打分器协议（Protocol），定义 `score(query, texts) -> list[float]` |
| `BaiLianRerankScorer` | 默认打分器，走百炼 DashScope TextReRank，含分批、重试、顺序还原 |
| `CrossEncoderReranker` | 精排编排器，负责候选校验、打分调度、排序、对比输出 |

---

## 注意事项

- **本层不去重**：同一 `id` 多次出现会各自成条，跨通道去重是融合层的职责。
- **top_k 截断时机**：截断发生在打分与排序之后，全体候选都会被打分，`count` 字段反映截断前总数。
- **空候选**：候选列表为空时返回 `count=0` 的空结果，不调用打分器。
- **非法入参**：`query` 为空、`candidates` 类型错误、`top_k < 1` 等均显式抛出 `ValueError` / `TypeError`。
