# 关键词召回（Retrieve / BM25）

该模块位于 RAG 核心链路：

```
加载数据(load_file) -> 切分(chunk) -> 向量化(embedding) -> 索引构建/存储 -> 召回/检索(retrieve) -> 重排 -> 生成
```

`bm25_retriever.py` 与 `fts5_retriever.py` 是**关键词召回**通道的两个实现：前者用「本地分词缓存 + 内存 BM25 统计量」，
后者用「SQLite FTS5 单文件倒排索引」。两者都对 chunk 文本做词项匹配，与 `vector_store` 的稠密向量召回互为补充，
最终由融合层（尚未实现）合并各路结果。两者消费同一批 records、返回同一结构，可以互换或并列使用。

- 第一至十章：`bm25_retriever.py`（BM25 本地缓存方案）
- 第十一章：`fts5_retriever.py`（SQLite FTS5 方案）

## 一、设计目标与边界

| 目标 | 做法 | 边界 |
|---|---|---|
| 不依赖 Chroma | 构建期直接消费 `to_records()` 的 records | 两边由构建脚本保证同源 |
| 启动快 | 缓存分词结果，加载时只重建统计量 | 缓存体积大于原文 |
| 更新可控 | 按 `doc_id` 先删后写 | 不支持真正的增量 IDF 更新 |
| 结果可解释 | 返回命中的词项分数与原始 metadata | 分数无上界，不能与 cosine 直接比较 |
| 可独立部署 | 只依赖本地文件 + jieba | pickle 文件仅限本项目生成 |

当前**不包含**：RRF 融合、rerank、混合检索编排、向量通道封装。这些属于后续工作。

## 二、实现流程

### 2.1 构建期（离线）

```
to_records() 的 records
        │
        ▼
┌───────────────────────────┐
│ 1. 校验记录                │  text 非空、doc_id 必须、id 可推导
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 2. 按 doc_id 分组          │  同批重复 id 直接报错
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 3. 删除同文档旧 chunk      │  解决 chunk_id 重排后的幽灵记录
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 4. 逐条分词                │  tokenize(text)（见 embedding/tokenizer.py）
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 5. 重建 BM25 统计量        │  倒排表 + df/idf + doc_len/avgdl
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 6. 原子落盘                │  docs.pkl -> manifest.json -> meta.json
└───────────────────────────┘
```

增量构建（`rebuild=False`）在缓存已存在时会**先自动 `load()`**，
再按 `doc_id` 覆盖，因此新进程可以直接做单文档更新，不需要手动加载。

### 2.2 查询期（在线）

```
search(query, top_k, where)
        │
        ▼
┌───────────────────────────┐
│ 1. 索引就绪检查            │  未 build/load -> RuntimeError
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 2. 入参校验                │  query 非空、top_k >= 1、where 结构合法
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 3. 查询分词并去重          │  避免子词扩展造成重复加权
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 4. BM25 打分               │  只遍历包含查询词的倒排项
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 5. 元数据过滤              │  打分后、排序前筛选
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 6. 排序取 top_k            │  分数降序，同分按 id 稳定排序
└─────────────┬─────────────┘
              ▼
[{id, text, metadata, score}]
```

## 三、缓存结构

默认目录：`RAG/data/bm25_index`（可用构造参数覆盖）。

```
data/bm25_index/
├── docs.pkl          # 分词结果与文档身份信息
├── manifest.json     # 每篇文档的 chunk 数与内容指纹
└── meta.json         # 版本/参数/统计量，作为提交标记
```

### 3.1 docs.pkl

```python
{
    "ids":    ["a.md::0", "a.md::1"],
    "tokens": [["安装", "chroma", "步骤"], ["检索", "流程"]],
    "metas":  [{"doc_id": "a.md", "chunk_id": 0, ...}],
    "texts":  ["安装 Chroma 的步骤", "检索流程"],
}
```

四个列表严格等长、按下标对齐。存 `texts` 是为了让 BM25 通道能独立返回原文，
查询期不访问 Chroma。

### 3.2 manifest.json

```json
{
  "a.md": {"chunks": 2, "hash": "cef6cd4239c55434", "updated_at": "2026-11-04T22:02:09"}
}
```

`hash` 为 `blake2b(id + text)`，用于发现“chunk 数不变但内容变了”。

### 3.3 meta.json

```json
{
  "schema_version": 1,
  "tokenizer_version": 1,
  "k1": 1.5,
  "b": 0.75,
  "chunk_count": 2,
  "doc_count": 1,
  "token_count": 12,
  "built_at": "2026-11-04T22:02:09"
}
```

### 3.4 落盘与校验

1. 写 `docs.pkl.tmp` → `os.replace`；
2. 写 `manifest.json.tmp` → `os.replace`；
3. 最后写 `meta.json.tmp` → `os.replace`。

`meta.json` 是提交标记：它存在且版本匹配，才认为缓存完整。
加载时校验字段长度、`schema_version`、`tokenizer_version`，任一不符直接报错要求重建。

## 四、核心功能

| 方法 / 属性 | 说明 | 返回 |
|---|---|---|
| `build(records, *, rebuild=False)` | 构建或增量更新索引 | `{records, chunks, docs, tokens, removed}` |
| `load()` | 加载本地缓存 | `bool`（缓存不存在返回 `False`） |
| `search(query, *, top_k=20, where=None)` | 关键词检索 | `list[{id, text, metadata, score}]` |
| `clear()` | 删除缓存文件并清空内存 | `None` |
| `len(retriever)` | chunk 总数 | `int` |
| `doc_ids` | 已建索引的文档列表 | `list[str]` |
| `is_ready` | 是否已构建或加载 | `bool` |

## 五、BM25 计算与参数

打分公式（与 Lucene 一致）：

```
idf(t)     = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(D, Q) = Σ idf(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · |D| / avgdl))
```

实现细节：

- **倒排表**：`term -> [(doc_idx, tf)]`，查询时只遍历命中词项的文档，而不是全库扫描；
- **IDF 恒为正**：避免极高频词产生负分；
- **查询词去重**：`err_conn_refused` 会扩展出多个子词，去重防止同一词项被重复加权。

| 参数 | 默认 | 作用 | 调整影响 |
|---|---|---|---|
| `k1` | 1.5 | 词频饱和速度 | 越大越强调重复出现的词 |
| `b` | 0.75 | 文档长度归一化 | 0 不归一化，1 完全归一化 |
| `top_k` | 20 | 返回条数 | 融合前建议取 20~50 |
| `cache_dir` | `data/bm25_index` | 缓存目录 | 每个集合建议独立目录 |

## 六、元数据过滤

`search(where=...)` 支持的操作符：

| 操作符 | 示例 | 说明 |
|---|---|---|
| 直接比较 | `{"doc_id": "a.md"}` | 隐式 `$eq` |
| `$eq` / `$ne` | `{"chunk_id": {"$ne": 0}}` | 相等 / 不等 |
| `$in` / `$nin` | `{"doc_id": {"$in": ["a.md", "b.md"]}}` | 集合包含 |
| `$gt/$gte/$lt/$lte` | `{"tokens": {"$gte": 200}}` | 数值/字符串比较 |
| `$and` / `$or` | `{"$and": [{"doc_id": "a.md"}, {"chunk_id": {"$gte": 1}}]}` | 组合条件 |

设计说明：

- **在打分后、排序前过滤**，因此过滤条件越严格，返回条数可能少于 `top_k`；
- `where` 在检索入口统一校验，非法操作符即使没有命中也会立即报错，不会静默忽略；
- 不支持 `$contains`：与 Chroma 本地引擎对字符串 metadata 的行为保持一致，避免两条通道语义分叉。

## 七、使用示例

### 7.1 构建

```python
from core_line.retrieve.bm25_retriever import BM25Retriever
from core_line.vector_store.chromadb_store import ChromaVector

store = ChromaVector()
bm25 = BM25Retriever()          # 默认 data/bm25_index

for document in md_load.load_file(file_path):
    records = to_records(md_chunker.chunk(document.page_content), document.metadata["file_name"])
    store.add_records(records)
    bm25.build(records)         # 与向量写入并列，两边消费同一批 records
```

### 7.2 查询

```python
bm25 = BM25Retriever()
if not bm25.load():
    raise RuntimeError("BM25 缓存缺失，请先运行索引构建")

hits = bm25.search("配置 HNSW 参数", top_k=20)
hits = bm25.search("召回策略", top_k=20, where={"doc_id": "retrieve.md"})
```

### 7.3 与向量通道并列

```python
dense = store.query(query, top_k=20)
sparse = bm25.search(query, top_k=20)

# 两路结果结构对齐（id/text/metadata/score），可直接交给 RRF 融合
```

## 八、可扩展性

| 扩展点 | 当前状态 | 建议做法 | 注意事项 |
|---|---|---|---|
| 打分器 | 内置 `_BM25Index` | 替换为 `rank_bm25` 适配器，或 `bm25s` | 保持 `get_scores(tokens) -> list[float]` 接口 |
| 分词器 | 直接导入 `core_line.embedding.tokenizer` | 改为构造参数注入 | 版本号必须与缓存 meta 同步 |
| 融合层 | 未实现 | 新增 `fusion.py`，用 RRF 按排名融合 | BM25 与 cosine 分数量纲不同，不要直接相加 |
| 字段加权 | 未实现 | 写入时把 `heading_path` 词项重复计入，或拆多字段 BM25 | 会改变 tf 分布，需重新评估 |
| 增量更新 | 按文档覆盖 | 已支持 `build(records)` 单文档更新 | 真正的增量 IDF 更新未实现，规模大时仍建议批量重建 |
| 服务化 | 单机本地缓存 | 多实例时改为共享检索服务（ES/OpenSearch） | 本地 pickle 无法跨节点共享 |
| 缓存格式 | pickle | 可换 msgpack/json 提升可移植性 | 体积与加载速度会变差 |

扩展原则：

1. **保持接口稳定**：`search` 的返回结构不要随意增删字段，融合层依赖 `id` 与 `score`；
2. **版本先行**：任何影响词项或统计量的改动，都先递增 `TOKENIZER_VERSION` 或 `SCHEMA_VERSION`；
3. **单一路径**：不要在 `search` 内部偷偷降级为纯向量，失败应显式抛出。

## 九、限制与注意事项

1. **pickle 安全**：`docs.pkl` 只能加载本项目生成的文件，不要接收外部输入。
2. **内存占用**：分词结果与倒排表都常驻内存；万级 chunk 通常在百 MB 量级，十万级需评估。
3. **文本存两份**：Chroma 与 BM25 缓存各存一份原文，换来查询期零 Chroma 依赖。
4. **分数不可比**：BM25 分数无上界，不能与 `cosine` 的 `score` 直接混用；用 RRF。
5. **过滤后不足 top_k**：这是预期行为，调用方不能假设一定返回 `top_k` 条。
6. **全量更新成本**：每次 `build` 都会重建整份 BM25 统计量（O(总词项数)），百万级需改用 `bm25s`。
7. **两边同源靠脚本**：BM25 不会自动感知 Chroma 的写入，构建脚本必须同时调用两边。

## 十、运维与验证

### 10.1 何时必须重建

| 变更 | 需要重建 |
|---|---|
| 新增 / 删除文档 | 是（`build` 增量即可） |
| 文档内容或切分参数变化 | 是（同 `doc_id` 覆盖） |
| `tokenizer.py` 规则变化 | 是（版本号会拦截） |
| 修改 `k1` / `b` | 否，加载时按新参数重建统计量 |
| 更换缓存目录 | 是（缓存不通用） |

### 10.2 验证清单

| 检查项 | 期望 |
|---|---|
| `len(retriever)` 与写入 chunk 数一致 | 相等 |
| `doc_ids` 覆盖全部文档 | 无缺失 |
| 重启后 `load()` | 返回 `True`，`count` 不变 |
| 重复 `build` 同一批 records | `count` 不翻倍 |
| 文档变短后重建 | 旧 `chunk_id` 不再被召回 |
| `search` 抽查专有名词 | 命中对应 chunk，且排名靠前 |
| 删除缓存后 `load()` | 返回 `False`，不静默降级 |

### 10.3 最小自检脚本

```python
r = BM25Retriever("data/bm25_index")
assert r.load(), "缓存缺失"
assert r.count() > 0

hits = r.search("HNSW 参数", top_k=5)
assert hits, "关键词召回为空"
assert all({"id", "text", "metadata", "score"} <= set(h) for h in hits)
assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)
```

## 十一、FTS5 关键词通道

`fts5_retriever.py` 把分词结果写进 **SQLite FTS5** 倒排索引：匹配、打分（`bm25()`）、元数据过滤、截断
全部在 SQL 内完成，查询期不需要把语料加载进 Python 内存。

### 11.1 与 BM25 通道的差异

| 维度 | `BM25Retriever` | `FTS5Retriever` |
|---|---|---|
| 索引载体 | `docs.pkl` + 两个 json | 单个 `fts5.db` |
| 分词结果 | 全量常驻内存 | 存表，按需读取 |
| 打分 | 手写 `_BM25Index` | FTS5 内置 `bm25()`，支持列权重 |
| 元数据过滤 | Python 侧（打分后、排序前） | 下推 SQL（先过滤、再排序截断） |
| 返回条数 | 过滤严格时可能少于 `top_k` | 有足够命中时总是满 `top_k` |
| 增量成本 | 每次 `build` 重建全部统计量 | 只重写变动文档对应的倒排行 |
| 构建原子性 | 依赖 `meta.json` 提交标记 | 单事务，DDL + 写入一起提交或回滚 |
| 并发读 | 进程内各持一份内存快照 | WAL 模式下多进程可同时读 |

共同点：复用同一套 `core_line.embedding.tokenizer`、支持同一套 Chroma 风格 `where`、
返回同样的 `[{id, text, metadata, score}]`，分数同样**不能**与 cosine 比较（融合请用 RRF）。

### 11.2 为什么先分词再入库

FTS5 自带的 `unicode61` 会把连续汉字当成一个整词（`向量检索` 只产出 1 个 token），子查询 `检索` 命中不了；
按字符切又会丢失词义。因此沿用项目已有的 jieba 分词：

```
text ──tokenize()──► ["安装", "chroma", "hnsw", "参数"]
                          │ 空格拼接
                          ▼
                 "安装 chroma hnsw 参数"  ──► body_tokens 列
```

FTS5 侧只需 `unicode61` 按空白切分，**不依赖自定义 tokenizer**（Python 的 `sqlite3` 也无法注册）。
代价是分词规则一变就必须重建，这一点由 `TOKENIZER_VERSION` 校验兜住。

### 11.3 表结构

```
data/fts5_index/fts5.db
├── chunks       正表：id / doc_id / chunk_id / heading_path / text / 分词列 / meta(JSON)
├── chunks_fts   FTS5 虚拟表：body_tokens、heading_tokens（content='chunks' 外部内容）
├── docs         文档清单：doc_id / chunks / hash / updated_at
└── meta         单行 JSON：结构与分词器版本、列权重、统计量、构建时间
```

- **外部内容表**：`content='chunks'` 让分词结果只存一份，索引不重复存正文；
- **触发器**：insert / delete / update 三个触发器保持正表与倒排索引同步，Python 侧无需手工维护；
- **指纹**：`docs.hash` 为 `blake2b(id + text)`，与 BM25 的 manifest 同算法，用于跳过未变动文档。

### 11.4 查询表达式

| 环节 | 处理 |
|---|---|
| 分词 | `tokenize(query)`，与建库完全同一套规则 |
| 去重 | 保序去重，避免子词扩展重复加权 |
| 组装 | 每个词加双引号后 `OR` 连接：`"配置" OR "hnsw"` |
| 加引号的原因 | `err_conn_refused`、`v1.2.3` 里的 `-` `.` `_` 会被 FTS5 当成语法符号 |
| 用 OR 而非 AND | 与 BM25「多词命中累加分」对齐，命中词多的排前面 |
| 列权重 | `bm25(chunks_fts, 1.0, 0.4)`：正文 1.0、标题 0.4 |

### 11.5 召回 SQL

```sql
SELECT c.id, c.text, c.meta, -bm25(chunks_fts, ?, ?) AS score
FROM chunks_fts
JOIN chunks AS c ON c.rowid = chunks_fts.rowid
WHERE chunks_fts MATCH ?
  AND <where 编译出的条件>
ORDER BY score DESC, c.id ASC
LIMIT ?
```

`bm25()` 返回「越小越好」的负值，取负号后与其它通道「分数越大越相关」的约定一致。

### 11.6 where 下推

| 字段 | SQL 表达式 |
|---|---|
| `doc_id` / `chunk_id` / `heading_path` | 独立列，直接比较 |
| 其它字段 | `json_extract(c.meta, ?)` |

操作符与 `BM25Retriever` 一致：`$eq/$ne/$in/$nin/$gt/$gte/$lt/$lte/$and/$or`。
`$ne` 用 `IS NOT`、`$nin` 额外放行 `NULL`，保证「字段缺失」时的判断与 Python 侧相同。
字段名需匹配 `^[A-Za-z_][A-Za-z0-9_]*$`：`json_extract` 的路径无法参数化绑定，先约束再拼接。

### 11.7 使用示例

```python
from core_line.retrieve.fts5_retriever import FTS5Retriever

fts = FTS5Retriever("data/fts5_index")     # 传目录时自动拼 fts5.db
fts.build(records)                         # 构建期：与 store.add_records、bm25.build 并列
fts.build(records)                         # 内容未变：命中指纹，stats["skipped"] 列出全部文档

fts = FTS5Retriever()
if not fts.load():
    raise RuntimeError("FTS5 索引缺失，请先构建")

hits = fts.search("配置 HNSW 参数", top_k=20)
hits = fts.search("召回策略", top_k=20, where={
    "$and": [{"doc_id": "retrieve.md"}, {"tokens": {"$gte": 200}}],
})
```

### 11.8 运维与验证

| 变更 | 是否需要 `rebuild=True` |
|---|---|
| 新增 / 删除文档 | 否，`build(records)` 按 `doc_id` 覆盖 |
| 文档内容变化 | 否，指纹不一致会自动重写 |
| `tokenizer.py` 规则变化 | 是（`TOKENIZER_VERSION` 拦截） |
| 表结构变化（`SCHEMA_VERSION`） | 是 |
| 调整 `body_weight` / `heading_weight` | 否，权重只在查询期生效 |

| 检查项 | 期望 |
|---|---|
| 重复 `build` 同一批 records | `len(fts)` 不增长，`skipped` 覆盖全部文档 |
| 文档变短后重建 | 旧 `chunk_id` 不再被召回 |
| 新进程 `load()` | 返回 `True`，检索结果与构建进程一致 |
| 不存在的词 | 返回 `[]`，不抛异常 |
| 非法 `where` | 立即 `ValueError`，即使无命中也要报错 |
| `fts.clear()` 后 `load()` | 返回 `False`，`.db` / `-wal` / `-shm` 一并删除 |

注意：`sqlite3` 连接不跨线程，多线程场景请每个线程创建实例；并发写靠 `busy_timeout=5000` 兜底，
批量重建仍建议单进程执行。
