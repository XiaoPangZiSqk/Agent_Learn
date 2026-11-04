# 向量库存储与检索（Vector Store）

该模块位于 RAG 核心链路：

```
加载数据(load_file) -> 切分(chunk) -> 向量化(embedding) -> 索引构建/存储(vector_store) -> 召回/检索 -> 重排 -> 生成
```

`chromadb_store.py` 中的 `ChromaVector` 负责把切分后的文本块写入本地向量库，并提供相似度检索，
是 embedding 模块的下游、retrieve/rerank 模块的上游。

## 一、设计要点

| 关注点 | 取值 / 做法 | 说明 |
|---|---|---|
| 存储模式 | `chromadb.PersistentClient` 本地持久化 | 索引落盘，进程重启后直接复用，无需重新向量化 |
| 距离度量 | `cosine` | 检索结果同时给出 `distance` 与 `score = 1 - distance` |
| 记录 id | `{doc_id}::{chunk_id}`（或显式传入 `id`） | 稳定且可推导，便于按文档定位与覆盖 |
| 写入语义 | `upsert` | 同 id 覆盖旧记录，重复构建索引不产生副本 |
| 维度绑定 | 写入/查询前显式校验 | 维度不一致直接报错，换模型需 `reset()` 重建 |
| embedding 获取 | 懒加载 `BaiLianEmbeder` | 只读/管理操作不要求 `DASHSCOPE_API_KEY` |

## 二、主要流程

### 2.1 索引构建（写入）

```
to_records() 产出的 records
            │
            ▼
┌───────────────────────────┐
│ 阶段 1 记录校验            │  必须是 dict 列表；text 非空
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 2 生成 id 与 metadata │  id = id 或 doc_id::chunk_id
│                           │  text 存文档，其余字段存 metadata
│                           │  过滤 None / 非标量转 str
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 3 分批               │  默认每批 64 条
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 4 向量化             │  embedder.embed_documents(batch)
│                           │  内部再按模型上限（10 条）分批
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 5 维度校验           │  条数与维度双重校验，失败即终止
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 6 upsert 落盘        │  ids / embeddings / documents / metadatas
└─────────────┬─────────────┘
              ▼
        返回写入的 id 列表
```

关键逻辑关系：**id 生成规则是幂等的前提**，同一份文档重复建索引不会产生重复记录；
向量化失败时该批次不会写入，避免“向量与文本错位”的脏数据。

### 2.2 相似度检索（查询）

```
query(text, top_k, where, ...)
            │
            ▼
┌───────────────────────────┐
│ 阶段 1 查询文本校验        │  非空 str
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 2 查询向量化          │  embed_query(text)，text_type=query
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 3 向量维度校验        │  与集合维度不一致直接报错
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 4 集合查询            │  n_results=top_k
│                           │  where / where_document 下推过滤
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 5 组装命中            │  id / text / metadata
│                           │  distance、score = 1 - distance
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 阶段 6 阈值过滤            │  score_threshold 可丢弃低分命中
└─────────────┬─────────────┘
              ▼
    list[dict]（按 distance 升序）
```

空集合查询返回 `[]`，不抛异常，调用方无需额外判空。

## 三、核心功能列表

| 分类 | 方法 | 说明 | 返回 |
|---|---|---|---|
| 写入 | `add_records(records, batch_size=64)` | 写入 `to_records()` 产出的 chunk 记录 | `list[str]` 写入的 id |
| 写入 | `add_texts(texts, metadatas=None, ids=None, batch_size=64)` | 直接写入纯文本，id 缺省用 uuid | `list[str]` 写入的 id |
| 查询 | `query(text, top_k=5, where=None, where_document=None, score_threshold=None)` | 文本检索 | `list[dict]` |
| 查询 | `query_by_vector(vector, top_k=5, ...)` | 用已有向量检索，参数与返回同上 | `list[dict]` |
| 读取 | `get(ids=None, where=None, limit=None, offset=None, with_embeddings=False)` | 按 id / 元数据读取，不做向量检索 | `list[dict]` |
| 管理 | `count()` / `len(store)` | 集合记录总数 | `int` |
| 管理 | `delete(ids=None, where=None)` | 按 id 或条件删除，二者至少给一个 | `None` |
| 管理 | `reset()` | 删除并重建集合（换维度 / 全量重建） | `None` |
| 属性 | `collection` | 底层 `chromadb.Collection`，用于未封装的能力 | `Collection` |

## 四、数据结构

### 4.1 写入记录

字段来自 `chunker.markdown_chunker.to_records()`：

| 字段 | 类型 | 去向 |
|---|---|---|
| `doc_id` | `str` | metadata，同时参与生成 id |
| `chunk_id` | `int` | metadata，同时参与生成 id |
| `text` | `str` | document（原文，检索时原样返回） |
| `heading_path` | `str` | metadata |
| `char_start` / `char_end` | `int` | metadata |
| `tokens` / `overlap_tokens` | `int` | metadata |
| `id` | `str`（可选） | 显式指定时优先于 `doc_id::chunk_id` |

### 4.2 检索结果

```python
{
    "id": "intro.md::3",
    "text": "RAG 的检索流程...",
    "metadata": {"doc_id": "intro.md", "chunk_id": 3, "heading_path": "A > B", ...},
    "distance": 0.137,
    "score": 0.863,          # 1 - distance，越大越相似
}
```

`get()` 返回 `{"id", "text", "metadata"}`，`with_embeddings=True` 时额外带 `embedding`。

### 4.3 metadata 约束

Chroma 只接受 `str / int / float / bool`，且不接受空 dict，因此写入前统一处理：

- `None` 值直接丢弃；
- 列表、字典等非标量值转成 `str`；
- `add_records` 中 metadata 为空时补 `{"id": record_id}`；
- `add_texts` 的 `metadatas` 要么整体为 `None`（不存元数据），要么每条清洗后都非空。

## 五、配置项

### 5.1 构造参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `persist_dir` | `RAG/data/chroma_data` | 本地持久化目录，不存在会自动创建 |
| `collection_name` | `rag_chunks` | 集合名，需满足 Chroma 命名规则（3~512 位，仅 `a-zA-Z0-9._-`，首尾为字母或数字） |
| `embedder` | `None` | 自定义向量化器（提供 `embed_documents` / `embed_query`，可选 `dimension` 属性）；缺省懒加载 `BaiLianEmbeder` |
| `dimension` | embedder 维度，再缺省 `1024` | 向量维度，需与 embedding 模型输出一致 |

### 5.2 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_PERSIST_DIR` | `RAG/chroma_data` | 覆盖持久化目录 |
| `CHROMA_COLLECTION_NAME` | `rag_chunks` | 覆盖集合名 |
| `DASHSCOPE_API_KEY` | 无 | 懒加载 `BaiLianEmbeder` 时必需，缺失会在首次写入/查询时报错 |

> 环境变量在类定义（模块导入）时读取，运行期间修改不生效，需在导入前设置。

### 5.3 类常量

| 常量 | 值 | 含义 |
|---|---|---|
| `DEFAULT_TOP_K` | `5` | 默认返回条数 |
| `WRITE_BATCH_SIZE` | `64` | 单批 upsert 条数 |
| `DISTANCE_SPACE` | `cosine` | 集合距离度量 |

## 六、使用示例

```python
from core_line.vector_store.chromadb_store import ChromaVector

store = ChromaVector()          # 默认 chroma_data/ + rag_chunks

# 建索引：records 来自 MarkdownChunker + to_records
ids = store.add_records(records)
print("写入", len(ids), "条，当前共", store.count(), "条")

# 检索
hits = store.query("什么是 RAG？", top_k=5)
for hit in hits:
    print(hit["score"], hit["metadata"]["heading_path"], hit["text"][:60])

# 只在某篇文档内检索，并过滤低分
hits = store.query(
    "重排策略",
    top_k=5,
    where={"doc_id": "retrieve.md"},
    score_threshold=0.5,
)

# 读取与删除
records = store.get(where={"doc_id": "retrieve.md"})
store.delete(["retrieve.md::0"])

# 换 embedding 维度后重建
store.reset()
```

依赖：

```bash
pip install chromadb   # 当前环境 1.5.9
```

## 七、异常与边界

| 场景 | 行为 |
|---|---|
| `collection_name` 为空 | 初始化时 `ValueError` |
| `collection_name` 不符合 Chroma 命名规则 | 初始化时 `chromadb.errors.InvalidArgumentError` |
| 记录不是 dict、缺少 `text` 或文本为空 | `add_records` 抛 `TypeError` / `ValueError`，带出错下标 |
| 记录缺少 `doc_id`/`chunk_id` 且未给 `id` | `ValueError`，提示无法生成 id |
| 同一次写入出现重复 id | `ValueError`（跨次写入是覆盖，不算重复） |
| embedding 返回条数不符 | `RuntimeError` |
| 向量维度与集合维度不一致 | `ValueError`，提示需 `reset()` 重建 |
| `top_k < 1`、查询文本/向量为空 | `ValueError` |
| 集合为空时查询 | 返回 `[]` |
| `delete()` 未给 `ids` 也未给 `where` | `ValueError`，提示用 `reset()` 清空 |

设计原则：**配置与入参错误尽早暴露；写入幂等可重试；数据不一致宁可报错也不写入错位结果。**

## 八、附加说明

1. **维度一致性**：`dimension` 与 embedding 模型强绑定（`text-embedding-v4` 默认 1024 维）。换模型或换维度等价于重建索引，需先 `reset()` 再重新写入。
2. **已有集合的度量**：若集合由旧版本/其他代码创建为 `l2` 或 `ip`，本模块会复用该集合并打印 warning，此时 `score` 仍按 `1 - distance` 计算，语义可能失真；建议用新集合名或 `reset()`。
3. **持久化目录**：默认 `RAG/chroma_data` 已在 `.gitignore` 中忽略，属于可随时重建的产物，不要提交到仓库。
4. **并发与性能**：`PersistentClient` 为进程内单例式使用；写入按批串行，避免触发 embedding 限流。多进程同时写同一目录需由调用方自行加锁或串行化。
5. **当前边界**：只提供单条查询（未封装批量 query）、未内置 BM25/混合检索、未接入 rerank；`collection` 属性可直接调用 Chroma 原生能力补齐。
6. **与上下游衔接**：上游 `to_records()` 的字段直接作为 metadata 落库；下游检索模块可直接消费 `score` 排序，或把 `text` 交给 rerank 模块精排。
