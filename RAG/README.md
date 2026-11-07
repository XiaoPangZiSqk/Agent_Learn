# RAG 项目业务流程说明

本文档梳理 `RAG/` 的完整业务流程，覆盖各功能模块的职责与关系、主流程的先后顺序与关键节点、模块间的数据流向与依赖，以及每个阶段的关键输入与输出。

内容依据当前 `src/` 下的实现与各模块 README 整理；标注为“待实现”的部分，对应当前目录为空或入口尚未编排的能力，便于团队区分“已落地”与“规划中”。

---

## 1. 项目概览

### 1.1 目标

把本地 Markdown 文档构建为可检索的知识库，并提供两条互补的召回通道：

- **向量通道**：语义相似度召回（Chroma + 百炼 `text-embedding-v4`）；
- **关键词通道**：词项精确匹配召回（BM25 + jieba 分词）。

两条通道消费同一批切分记录，各自独立建索引、独立检索，最终在融合层按排名合并。

### 1.2 能力现状

| 能力 | 状态 | 说明 |
|---|---|---|
| 文档加载 | 可用 | 支持 `.md` / `.markdown`；`.pdf` / `.docx` 已注册但解析抛 `NotImplementedError` |
| Markdown 结构化切分 | 可用 | `MarkdownChunker`，边界感知 + 重叠 + 小块合并 |
| 记录标准化 | 可用 | `to_records()` 产出双通道共用的记录结构 |
| 向量索引与检索 | 可用 | `ChromaVector`，本地持久化、cosine 距离 |
| 关键词索引与检索 | 可用 | `BM25Retriever`，本地分词缓存 + 内存 BM25 统计 |
| 构建流程编排 | 部分可用 | `RagApp.py` 目前只执行“加载 → 切分 → 打印”，尚未写入两个索引 |
| 双路融合（RRF） | 待实现 | 仅在文档中约定接口与算法 |
| 重排（rerank） | 待实现 | `core_line/rerank/` 为空目录 |
| 生成（generate） | 待实现 | `core_line/generate/` 为空目录 |

---

## 2. 模块职责与相互关系

### 2.1 分层视图

```
入口/编排层      RagApp.py
                     │
构建层           文档加载 → 结构化切分 → 记录标准化
                     │
索引层           向量索引（Chroma）      关键词索引（BM25 缓存）
                     │                        │
召回层           向量召回（dense）        关键词召回（sparse）
                     └──────────┬─────────────┘
融合/重排/生成层          RRF → rerank → generate（均待实现）
```

### 2.2 模块清单

| 层次 | 模块 / 文件 | 关键类或函数 | 职责 | 主要输入 → 输出 |
|---|---|---|---|---|
| 入口 | `src/RagApp.py` | 脚本 | 串联加载与切分，用于调试和流程演示 | 目录路径 → 控制台输出 |
| 加载 | `src/core_line/load_file/load_data.py` | `LoadData`、`Document` | 扫描文件、按后缀分发解析、构建元数据 | 文件/目录 → `Iterator[Document]` |
| 解析工具 | `src/core_line/chunk/chunk_tool.py` | `parse_markdown`、`Block`、`predic_token`、`split_sentences` | Markdown 行扫描成结构块；token 估算；句子切分 | Markdown 文本 → `list[Block]` |
| 切分 | `src/chunker/markdown_chunker.py` | `MarkdownChunker`、`Chunk`、`to_records` | 按标题分节、块装箱、重叠、小块合并；产出标准记录 | 单篇 Markdown → `list[Chunk]` / `list[dict]` |
| 向量化 | `src/core_line/embedding/bai_lian_embeder.py` | `BaiLianEmbeder` | 文档批量向量化、查询向量化，含分批、重试、顺序还原 | `list[str]` → `list[list[float]]` / `list[float]` |
| 分词 | `src/core_line/embedding/tokenizer.py` | `tokenize`、`tokenize_batch`、`TOKENIZER_VERSION` | 供 BM25 建库与查询共用的中英文切词 | 文本 → `list[str]` |
| 向量库 | `src/core_line/vector_store/chromadb_store.py` | `ChromaVector` | 记录写入、向量检索、读取与集合维护 | `records` / 查询文本 → 命中列表 |
| 关键词召回 | `src/core_line/retrieve/bm25_retriever.py` | `BM25Retriever`、`_BM25Index` | BM25 缓存构建、加载、打分与元数据过滤 | `records` / 查询文本 → 命中列表 |
| 重排 | `src/core_line/rerank/` | 无 | 预留：对召回结果精排 | 待定义 |
| 生成 | `src/core_line/generate/` | 无 | 预留：拼装上下文并调用 LLM | 待定义 |

### 2.3 模块间协作关系

```
RagApp
 ├── core_line.load_file.load_data        （加载）
 └── chunker.markdown_chunker             （切分 + 记录标准化）
         └── core_line.chunk.chunk_tool   （Markdown 解析 / token 估算）

ChromaVector
 ├── core_line.embedding.bai_lian_embeder （文档与查询向量化）
 └── chromadb                             （持久化 + 相似度检索）

BM25Retriever
 └── core_line.embedding.tokenizer        （建库与查询分词，必须同版本）
```

要点：

1. **两条索引通道互不依赖**：向量通道不读 BM25 缓存，BM25 也不从 Chroma 拉语料，二者只在构建期共享同一批 `records`。
2. **`embedding` 目录承担两种职责**：`bai_lian_embeder.py` 服务于向量通道，`tokenizer.py` 服务于关键词通道；二者共享同一套文本归一约定，但输出与用途完全不同。
3. **`chunker` 位于 `core_line` 之外**：它复用 `core_line/chunk/chunk_tool.py`，属于当前目录结构上的特例，后续可考虑归并到 `core_line/chunk` 下。

---

## 3. 端到端总流程

整个系统分为**构建期（离线）**与**查询期（在线）**两个阶段，二者通过落盘索引解耦。

```
构建期（离线，可重复执行）
────────────────────────────────────────────────────────────
  文档目录 / 单个文件
        │
        │ LoadData.load_file()            阶段 B1 加载
        ▼
  Document(page_content, metadata)
        │
        │ MarkdownChunker.chunk()         阶段 B2 切分
        ▼
  list[Chunk]
        │
        │ to_records(chunks, doc_id)      阶段 B3 记录标准化
        ▼
  records[]  ────────────────┬──────────────────────────────
        │                     │
        │ ChromaVector        │ BM25Retriever
        │ .add_records()      │ .build()
        ▼                     ▼
  data/chroma_data/      data/bm25_index/      阶段 B4 双通道落库
  （cosine 向量索引）     （docs/manifest/meta）

查询期（在线）
────────────────────────────────────────────────────────────
  query
    ├── embed_query ──► ChromaVector.query()  ──► dense hits
    └── tokenize    ──► BM25Retriever.search() ──► sparse hits
                                   │
                                   ▼
                         RRF 融合（待实现）
                                   ▼
                         rerank 精排（待实现）
                                   ▼
                         generate 生成（待实现）
```

---

## 4. 构建期流程（离线）

### 4.1 阶段 B1：文档加载

**入口**：`LoadData.load_file(path, recursive=True)`

**处理步骤**

1. 路径判定：指向文件则只加载该文件；指向目录则按 `**/*`（递归）或 `*`（不递归）惰性遍历。
2. 过滤隐藏文件与隐藏目录（路径任一部分以 `.` 开头）。
3. 按后缀分发：`.md` / `.markdown` → Markdown 解析；`.pdf` / `.docx` → 当前抛 `NotImplementedError`；其它类型静默跳过。
4. 文件级预检：
   - 大小为 0 → 跳过并告警；
   - 超过 50 MB（`DEFAULT_MAX_FILE_SIZE`）→ 跳过并告警。
5. 文本读取：按 `utf-8-sig` → `gb18030` → `utf-8(errors="replace")` 顺序尝试，统一将 `\r\n` / `\r` 归一为 `\n`。
6. 构建元数据：`source`（绝对路径）、`file_name`、`file_type`、`size_bytes`，Markdown 额外记录 `encoding`。

**关键输入 → 输出**

| 项 | 内容 |
|---|---|
| 输入 | 文件或目录路径；`recursive` 开关 |
| 输出 | `Iterator[Document]`，每项含 `page_content` 与 `metadata` |
| 失败行为 | 单文件异常只记日志并继续，不中断整批；路径不存在抛 `FileNotFoundError` |

**设计要点**：返回生成器，同一时刻内存中只有一份文档；目录边扫描边产出，不预先收集全部路径。

### 4.2 阶段 B2：结构化切分

**入口**：`MarkdownChunker.chunk(md)`，一篇文档对应一次调用。

**主流程与关键节点**

```
md 文本
  │ ① parse_markdown：行扫描 → Block[]
  ▼
Block[]（含 type / path / start / end / atomic）
  │ ② _split_sections：按标题路径分节
  ▼
[(heading_path, Block[]), ...]
  │ ③ _pack_blocks：节内装箱成 Chunk[]
  ▼
Chunk[]（无重叠）
  │ ④ _apply_overlap：句尾重叠，吸附句子边界
  ▼
Chunk[]（带 prefix）
  │ ⑤ _merge_small：小块并入前块
  ▼
Chunk[]（重编号 index）
```

**① 结构解析（Markdown 文本 → `Block`）**

- 逐行扫描并按固定优先级识别边界：YAML frontmatter → 围栏代码块 → ATX 标题 → GFM 表格 → 水平分割线 → 列表 → 引用 → 段落（兜底）。
- 维护标题栈生成面包屑 `path`，使每个块携带“自己在文档哪个章节下”。
- 代码块、表格、分割线、frontmatter 标记为**原子块**（`atomic=True`），默认不从中间切开。
- Setext 标题（文字行后跟 `===` / `---`）会被识别为标题而非段落。
- 每个 `Block` 记录原文偏移 `start` / `end`，供后续溯源。

**② 按标题分节（`_split_sections`）**

遇到标题即开启新节；非标题块归入当前节。每个节携带自己的 `heading_path`。切分不会跨节合并，保证块的章节归属唯一。

**③ 节内装箱（`_pack_blocks`）**

以“软预算 `target` + 硬上限 `max_size`”为约束，把同一节的 `Block` 依次装入 `Chunk`：

| 块类型 | 处理策略 |
|---|---|
| 原子块，且 `tokens <= max_size` | 允许把当前块撑到 `target ~ max_size`，不切中 |
| 原子块，且 `tokens > max_size` | 降级拆分：代码按空行分组并补围栏、表格按行分组并补表头、其它按行 |
| 普通块，累加后 `<= target` | 直接追加 |
| 普通块，单块 `> max_size` | 先 flush，再按句子切分后重新装箱 |
| 普通块，累加超过 `target` | flush 当前块，以该块为新块起点 |

超长内容的降级顺序：**原子块保护 → 结构内分组（代码/表格/行）→ 句子级切分 → 行级兜底**。

**④ 重叠（`_apply_overlap`）**

- 仅当**上一块**包含 `paragraph` / `list` / `quote` 类型时，才从上一块尾部取不超过 `overlap` 预算的内容作为当前块 `prefix`。
- 重叠区按句子边界吸附，不会从半句话中间截断。
- 重叠在**每个标题节内部**应用，不跨节。

**⑤ 小块合并（`_merge_small`）**

- 块小于 `min_size` 时，尝试并入前一块；
- 合并条件：与前一块同根标题（`heading_path[:1]` 相同）且合并后不超过 `target`；
- 合并后清空 `prefix` 与 `overlap_tokens`，避免重复计入。

**切分参数**

| 参数 | 默认值 | 作用 |
|---|---|---|
| `CHUNK_SIZE` / `target` | 400 token | 目标块大小（软预算） |
| `MAX_CHUNK_SIZE` / `max_size` | 560 token | 硬上限，超过必须切开 |
| `MIN_CHUNK_SIZE` / `min_size` | 120 token | 小块合并阈值 |
| `OVERLAP` | 60 token | 相邻块重叠预算 |

token 统一用 `tiktoken` 的 `o200k_base` 估算。

**关键输入 → 输出**

| 项 | 内容 |
|---|---|
| 输入 | 单篇 Markdown 全文（`str`） |
| 输出 | `list[Chunk]`，每块含 `core`、`prefix`、`heading_path`、`start`、`end`、`block_types`、`index` |
| 关键约束 | 原子块优先不切；块长受 `target` / `max_size` 约束；重叠只发生在同节正文块之间 |

### 4.3 阶段 B3：记录标准化

**入口**：`to_records(chunks, doc_id)`

把 `Chunk` 转成两条索引通道都能直接消费的扁平字典：

| 字段 | 类型 | 来源 / 含义 |
|---|---|---|
| `doc_id` | `str` | 调用方传入，当前 `RagApp` 使用 `document.metadata["file_name"]` |
| `chunk_id` | `int` | `Chunk.index`，同一文档内从 0 递增 |
| `heading_path` | `str` | 标题面包屑，以 `" > "` 连接 |
| `text` | `str` | `prefix + "\n\n" + core`（无 prefix 时即 core） |
| `char_start` / `char_end` | `int` | 在源文档中的字符偏移 |
| `tokens` | `int` | 该块 token 数 |
| `overlap_tokens` | `int` | 重叠区 token 数 |

**关键输入 → 输出**：`list[Chunk]` + `doc_id` → `list[dict]`（无落盘，纯内存结构）。

该结构是构建期的**唯一数据契约**：向量通道与关键词通道都只认这 8 个字段。

### 4.4 阶段 B4：双通道落库

两条通道**并列**执行，互不读取对方产物。

#### 4.4.1 向量通道：`ChromaVector.add_records(records)`

```
records
  │ 1. 校验：必须是 dict 列表，text 非空
  ▼
  │ 2. 生成 id 与 metadata
  │    id = record["id"] 或 "{doc_id}::{chunk_id}"
  │    text 作为 document，其余字段清洗后作为 metadata
  ▼
  │ 3. 按 batch_size=64 分批
  ▼
  │ 4. embedder.embed_documents(batch)  （内部再按模型上限 10 条分批）
  ▼
  │ 5. 校验向量条数与维度（必须等于集合维度，默认 1024）
  ▼
  │ 6. collection.upsert(...)  同 id 覆盖
  ▼
写入的 id 列表
```

- **幂等性来源**：id 由 `doc_id + chunk_id` 稳定推导，重复构建不产生副本。
- **失败行为**：任一批次校验或向量化失败即抛错，该批不写入，避免“文本与向量错位”。

#### 4.4.2 关键词通道：`BM25Retriever.build(records)`

```
records
  │ 1. 校验：text 非空、doc_id 必需、id 可由 chunk_id 推导
  ▼
  │ 2. 按 doc_id 分组；同批重复 id 直接报错
  ▼
  │ 3. 删除同 doc_id 的旧 chunk（防止 chunk_id 重排留下幽灵记录）
  ▼
  │ 4. 逐条 tokenize(text)
  ▼
  │ 5. 重建 BM25 统计量（倒排表 + df/idf + doc_len/avgdl）
  ▼
  │ 6. 原子落盘 docs.pkl → manifest.json → meta.json
  ▼
统计信息 {records, chunks, docs, tokens, removed}
```

- 增量构建时若缓存已存在，会先自动 `load()` 再按 `doc_id` 覆盖。
- `meta.json` 最后写入，作为“缓存完整”的提交标记；加载时校验结构版本与分词器版本。

**落盘产物**

```
RAG/data/
├── chroma_data/                 # Chroma 持久化（向量 + 原文 + metadata）
├── bm25_index/
│   ├── docs.pkl                 # ids / tokens / metas / texts，四列表等长对齐
│   ├── manifest.json            # 每篇文档的 chunk 数与内容指纹
│   └── meta.json                # schema/tokenizer 版本、k1/b、统计量、构建时间
└── load_file/                   # 源文档
```

---

## 5. 查询期流程（在线）

### 5.1 总览

查询期主要包含三步：**向量召回 → 关键词召回 → 融合**（融合尚未实现）。

向量通道与关键词通道**可独立调用**，调用顺序不影响结果；融合层按 `id` 对齐两路命中。

### 5.2 阶段 Q1：向量召回

**入口**：`ChromaVector.query(text, top_k=5, where=None, where_document=None, score_threshold=None)`

| 步骤 | 处理 |
|---|---|
| 1 | 校验查询文本非空 |
| 2 | `embed_query(text)`，固定 `text_type="query"` |
| 3 | 校验查询向量维度与集合维度一致 |
| 4 | `collection.query(n_results=top_k)`，`where` / `where_document` 下推过滤，cosine 距离 |
| 5 | 组装命中：`id`、`text`、`metadata`、`distance`、`score = 1 - distance` |
| 6 | 按 `score_threshold` 丢弃低分命中 |

**输出**：按相似度从高到低排列的命中列表；空集合返回 `[]`，不抛异常。

### 5.3 阶段 Q2：关键词召回

**入口**：`BM25Retriever.search(query, top_k=20, where=None)`

前置条件：必须先 `build()` 或 `load()`，否则抛 `RuntimeError`。

| 步骤 | 处理 |
|---|---|
| 1 | 索引就绪检查 |
| 2 | 入参校验：query 非空、`top_k >= 1`、`where` 结构合法 |
| 3 | 查询分词并去重（避免子词扩展重复加权） |
| 4 | 用倒排表计算 BM25 分数，只遍历命中词项 |
| 5 | 按 `where` 过滤 metadata（打分后、排序前） |
| 6 | 分数降序、同分按 id 稳定排序，取 `top_k` |

**输出**：`[{id, text, metadata, score}]`。

**注意**：BM25 分数无固定上界，与 cosine `score` 量纲不同，**不能直接比较或相加**。

### 5.4 阶段 Q3：融合（待实现）

约定接口：

| 项 | 约定 |
|---|---|
| 输入 | 两路各 `top_k` 条（建议各 20~50 条） |
| 对齐键 | `id`（两通道 id 规则均为 `doc_id::chunk_id`） |
| 算法 | RRF：`score(d) = Σ 1 / (k + rank_i(d))`，常用 `k = 60` |
| 输出 | 统一结构的去重命中列表，保留两路原始分数供排查 |

由于两路结果结构已对齐（`id` / `text` / `metadata` / `score`），融合层只需关注排名，不需要理解两路分数量纲。

### 5.5 阶段 Q4 / Q5：重排与生成（待实现）

- **rerank**：接收融合后的候选块，按 query 与块内容做精排，输出重排后的 Top-N；
- **generate**：用重排结果拼装上下文（受“`top_k × chunk_size × (1 + 重叠率)` ≤ 上下文预算 50%”约束），调用 LLM 生成答案并附引用（`doc_id` + `heading_path` + `char_start/end`）。

---

## 6. 数据流与依赖关系

### 6.1 数据形态演进

| 阶段 | 数据形态 | 关键字段 | 载体 |
|---|---|---|---|
| 加载后 | `Document` | `page_content`、`metadata.source/file_name/file_type/size_bytes/encoding` | 内存（生成器逐条产出） |
| 切分后 | `Chunk` | `core`、`prefix`、`heading_path`、`start`、`end`、`block_types`、`index` | 内存 |
| 标准化后 | `record` | `doc_id`、`chunk_id`、`heading_path`、`text`、`char_start`、`char_end`、`tokens`、`overlap_tokens` | 内存（双通道共用契约） |
| 向量落库后 | Chroma 记录 | `id`、`embedding`、`document(text)`、`metadata` | `data/chroma_data/` |
| 关键词落库后 | BM25 缓存 | `ids`、`tokens`、`metas`、`texts` | `data/bm25_index/` |
| 召回后 | hit | `id`、`text`、`metadata`、`score`（向量通道另有 `distance`） | 内存 |

### 6.2 依赖方向

```
load_data            （无内部依赖）
   │
   ▼
markdown_chunker ──► chunk_tool ──► tiktoken
   │
   ▼
to_records
   │
   ├──────────────► chromadb_store ──► bai_lian_embeder ──► dashscope
   │                      │
   │                      ▼
   │                   chromadb
   │
   └──────────────► bm25_retriever ──► tokenizer ──► jieba
```

依赖原则：

1. **单向依赖**：加载 → 切分 → 索引 → 召回，反向不成立；索引层不 import 召回层。
2. **索引层互不依赖**：`chromadb_store` 与 `bm25_retriever` 之间没有 import 关系。
3. **外部依赖最小化**：只有 `bai_lian_embeder` 需要网络与 API Key；其余模块均可离线运行。

### 6.3 外部依赖

| 依赖 | 使用位置 | 必需性 |
|---|---|---|
| `tiktoken` | `chunk_tool.predic_token` | 必需（切分） |
| `dashscope` + `python-dotenv` | `bai_lian_embeder` | 向量通道必需 |
| `chromadb`（当前环境 1.5.9） | `chromadb_store` | 向量通道必需 |
| `jieba` | `tokenizer` | 关键词通道必需 |
| 百炼 API Key | `bai_lian_embeder` | 向量通道必需 |

### 6.4 配置项汇总

**环境变量**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BAILIAN_EMBEDDING_API_KEY` | 无 | 百炼 API Key（代码实际读取的变量名） |
| `BAILIAN_EMBEDDING_MODEL` | `text-embedding-v4` | 向量模型名 |
| `CHROMA_PERSIST_DIR` | `RAG/data/chroma_data` | 向量库持久化目录 |
| `CHROMA_COLLECTION_NAME` | `rag_chunks` | 集合名 |

> 注意：`bai_lian_embeder.py` 的注释与报错信息写的是 `DASHSCOPE_API_KEY`，但代码实际读取 `BAILIAN_EMBEDDING_API_KEY`；部署时以代码为准，建议在 `.env` 中统一。环境变量在模块导入时读取，运行期修改不生效。

**关键运行参数**

| 模块 | 参数 | 默认值 |
|---|---|---|
| 切分 | `target` / `max_size` / `min_size` / `overlap` | 400 / 560 / 120 / 60 token |
| 向量化 | `dimension` / `batch_size` / `max_retries` | 1024 / 10 / 3 |
| 向量库 | `DEFAULT_TOP_K` / `WRITE_BATCH_SIZE` / 距离度量 | 5 / 64 / cosine |
| BM25 | `k1` / `b` / `DEFAULT_TOP_K` | 1.5 / 0.75 / 20 |

---

## 7. 关键输入与输出汇总

| 阶段 | 入口 | 关键输入 | 关键输出 | 是否落盘 | 失败行为 |
|---|---|---|---|---|---|
| B1 加载 | `LoadData.load_file` | 路径、`recursive` | `Iterator[Document]` | 否 | 单文件跳过；路径不存在抛错 |
| B2 切分 | `MarkdownChunker.chunk` | Markdown 全文 | `list[Chunk]` | 否 | 结构异常时降级为行级切分 |
| B3 标准化 | `to_records` | `chunks`、`doc_id` | `list[record]` | 否 | 不做额外校验 |
| B4a 向量入库 | `ChromaVector.add_records` | `records` | 写入 id 列表 | 是 | 校验失败即抛错，该批不写 |
| B4b 关键词入库 | `BM25Retriever.build` | `records` | 统计字典 | 是 | 重复 id / 缺少 doc_id 抛错 |
| Q1 向量召回 | `ChromaVector.query` | 查询文本、`top_k`、过滤条件 | `[{id,text,metadata,distance,score}]` | 否 | 维度不符 / 空查询抛错 |
| Q2 关键词召回 | `BM25Retriever.search` | 查询文本、`top_k`、`where` | `[{id,text,metadata,score}]` | 否 | 未加载索引抛错；无命中返回 `[]` |
| Q3 融合 | 待实现 | 两路命中 | 统一排序列表 | 否 | — |
| Q4 重排 | 待实现 | 融合结果、查询 | 精排 Top-N | 否 | — |
| Q5 生成 | 待实现 | 精排结果、查询 | 答案 + 引用 | 否 | — |

---

## 8. 关键节点与一致性约束

### 8.1 必须保持一致的约定

| 约束 | 内容 | 违反后果 |
|---|---|---|
| 记录同源 | 两条通道必须消费同一批 `records` | 两路召回结果对不上，融合失效 |
| id 规则 | `{doc_id}::{chunk_id}` | 无法去重、无法按文档覆盖更新 |
| 向量维度 | 集合维度 = embedding 输出维度（默认 1024） | 写入/查询直接报错 |
| 分词器版本 | BM25 建库与查询共用 `TOKENIZER_VERSION` | 词项对不齐，分数整体退化为 0 |
| 缓存版本 | `SCHEMA_VERSION` + `TOKENIZER_VERSION` 与 `meta.json` 一致 | 加载被拒绝，需重建 |
| 距离度量 | 集合使用 cosine，`score = 1 - distance` | 分数语义失真（会告警但不阻断） |

### 8.2 索引重建触发条件

| 变更 | 向量通道 | 关键词通道 |
|---|---|---|
| 新增文档 | 重新写入即可（upsert） | `build(records)` 增量覆盖 |
| 删除文档 | 需显式 `delete(where={"doc_id": ...})` | 不自动删除，需 `clear()` 后全量重建或扩展删除接口 |
| 文档内容或切分参数变化 | 重新写入，块数变少时需清理旧 chunk | 按 `doc_id` 先删后写，无残留 |
| 更换 embedding 模型或维度 | `reset()` 后全量重建 | 不受影响 |
| `tokenizer.py` 规则变化 | 不受影响 | 必须重建（版本号拦截） |
| 修改 `k1` / `b` | 不受影响 | 无需重建，加载时按新参数重算统计量 |
| 更换缓存目录 / 集合名 | 需全量重建 | 需全量重建 |

### 8.3 当前实现缺口与风险

1. **构建编排未完成**：`RagApp.py` 只打印切分结果，未调用 `ChromaVector.add_records` 与 `BM25Retriever.build`，因此两条索引实际上不会生成。
2. **后段链路为空**：RRF 融合、`rerank/`、`generate/` 均未实现，系统当前只能返回两路原始召回结果。
3. **`doc_id` 取值偏弱**：当前使用文件名，跨目录同名文件会互相覆盖；建议改用相对路径或内容哈希。
4. **向量侧缺少按文档清理**：`add_records` 只做 upsert，文档变短或切分参数调小后，旧的高位 `chunk_id` 不会被删除，会残留“幽灵块”。BM25 侧已按 `doc_id` 先删后写，向量侧需调用方显式 `delete(where={"doc_id": ...})` 后再写入。
5. **多格式解析缺失**：PDF / Word 仅注册了后缀，解析方法仍抛 `NotImplementedError`。
6. **备选切分器未接入且有缺陷**：`fixed_len_split` 以 token 总数作为字符下标使用（长文本块数偏多）；`recursive_character_split` 的外层返回值写在内层函数中，调用返回 `None`。两者均未进入主链路。
7. **目录归属不统一**：`chunker/` 在 `core_line/` 之外，通过导入根复用 `core_line/chunk`，迁移目录时需同步修改导入路径。

---

## 9. 运行与验证

### 9.1 运行方式

`RagApp.py` 以 `src` 为导入根，需在 `src` 目录下执行：

```bash
cd RAG/src
python RagApp.py
```

### 9.2 环境准备

```bash
pip install tiktoken dashscope python-dotenv chromadb jieba
```

`.env` 建议放在项目根目录：

```ini
BAILIAN_EMBEDDING_API_KEY=sk-xxx
BAILIAN_EMBEDDING_MODEL=text-embedding-v4
# 可选
CHROMA_PERSIST_DIR=./data/chroma_data
CHROMA_COLLECTION_NAME=rag_chunks
```

### 9.3 构建期自检清单

| 检查项 | 期望 |
|---|---|
| 切分结果 | 每块带 `heading_path`，代码块围栏成对，无跨节合并 |
| 块长 | 常规块落在 120~560 token；原子块仅在超限时拆分，无法合并的短块可低于 120 |
| 重复构建 | `store.count()` 不翻倍，BM25 `len(retriever)` 不增长 |
| 两路条数 | 同一批 records 写入后，向量与 BM25 的 chunk 数一致 |
| 缓存加载 | 重启后 `bm25.load()` 返回 `True`，`count` 不变 |

### 9.4 查询期自检清单

| 检查项 | 期望 |
|---|---|
| 向量召回 | `query()` 返回字段完整，`score` 随相关性下降 |
| 关键词召回 | 专有名词能命中，分数降序，同分按 id 稳定 |
| 过滤条件 | `where` 生效后条数可少于 `top_k`（预期行为） |
| 融合前对齐 | 两路 `id` 规则一致，可直接按 id 去重 |

---

## 10. 后续演进顺序建议

1. **补齐构建编排**：新增独立构建脚本，串联 `load → chunk → to_records → add_records + build`，两路同一批 records 一次写入。
2. **实现融合层**：新增 `core_line/retrieve/fusion.py`，用 RRF 合并两路排名，统一 `id` 去重。
3. **接入重排**：在 `core_line/rerank/` 实现精排，输入融合候选、输出 Top-N。
4. **接入生成**：在 `core_line/generate/` 实现上下文拼装、LLM 调用与引用回填。
5. **补全加载器**：实现 PDF / Word 解析，并统一多格式的 `doc_id` 生成规则。
