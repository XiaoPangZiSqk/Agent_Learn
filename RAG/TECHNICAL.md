# RAG 系统技术说明（开发者导读）

本文档面向开发者，逐节说明各核心模块的**功能职责、技术方案与代码实现位置**，专题覆盖
SC 自适应、语义缓存、快速模式、批量流水、构建冷启动等机制，并给出工程实施注意事项
与当前不足。业务流程视角请先读 `README.md`；本文聚焦"怎么实现的、去哪里改"。

**定位约定**：实现位置统一写为 `文件路径 → 类/方法`（不写行号，避免代码演进后失真）；
全局常量写为 `文件 → 常量名`。快速跳转可查文末[§7 符号索引](#7-符号快速定位索引)。

---

## 目录

1. [系统总览](#1-系统总览)
2. [核心模块逐节说明](#2-核心模块逐节说明)
3. [关键机制专题（性能与一致性）](#3-关键机制专题性能与一致性)
4. [模块职责边界与相互关系](#4-模块职责边界与相互关系)
5. [工程实施注意事项](#5-工程实施注意事项)
6. [当前不足与缺点](#6-当前不足与缺点)
7. [符号快速定位索引](#7-符号快速定位索引)

---

## 1. 系统总览

### 1.1 目标与能力边界

把本地 Markdown 文档构建为可检索知识库，提供"双通道召回 → RRF 融合 → 交叉重排 →
LLM 生成（带引用与自校验）"的完整 RAG 链路。

- **输入**：Markdown 文档目录（`.md`/`.markdown`；`.pdf`/`.docx` 仅注册未实现解析）。
- **输出**：结构化答案（JSON：answer / reasoning / citations / confidence /
  evidence_sufficient）+ 自校验状态（nli_checked / nli_passed）+ SSE 流式事件。

### 1.2 分层架构与数据流

```
配置/基建层   config.py（Key 链·预检）      cache.py（三级语义缓存）
                    │                            │
构建期        加载 → 切分 → 记录标准化 → 嵌入 → 双通道落库（Chroma + BM25）
查询期        召回(向量+BM25) → RRF 融合 → 交叉重排 → 上下文组装 → 生成(自校验)
                    └────────── pipeline.py（编排/批量并发/SSE）──────────┘
```

### 1.3 目录结构速览

```
RAG/
├── .env / .env.example          # 环境配置（模板逐项标注读取方）
├── README.md                    # 业务流程说明
├── TECHNICAL.md                 # 本文
├── data/
│   ├── load_file/               # 源文档
│   ├── chroma_data/             # Chroma 持久化
│   ├── bm25_index/              # BM25 缓存（docs.pkl/manifest/meta）
│   └── cache/rag_cache.db       # 语义缓存（sqlite）
└── src/
    ├── RagApp.py                # 入口：启动预检 → 加载 → 切分 → 打印
    ├── chunker/markdown_chunker.py          # 结构化切分主链路
    └── core_line/
        ├── config.py            # 统一配置 + 启动预检
        ├── cache.py             # 语义缓存
        ├── pipeline.py          # 查询流水线
        ├── chunk/chunk_tool.py  # Markdown 解析/token 估算/备选切分
        ├── load_file/load_data.py
        ├── embedding/{bai_lian_embeder.py, tokenizer.py}
        ├── vector_store/chromadb_store.py
        ├── retrieve/{retriever.py, bm25_retriever.py, fts5_retriever.py}
        ├── rerank/reranker.py
        └── generate/{generator.py, context_assembler.py}
```

### 1.4 全局数据契约

| 形态 | 定义位置 | 关键字段 |
|---|---|---|
| `Document` | `load_file/load_data.py` | `page_content`、`metadata.source/file_name/file_type/size_bytes/encoding` |
| `Block` | `chunk/chunk_tool.py` | `type/heading/paragraph/list/quote/code/table/hr/frontmatter`、`content/path/start/end/atomic` |
| `Chunk` | `chunker/markdown_chunker.py` | `core/prefix/heading_path/start/end/block_types/overlap_tokens/fragment` |
| `record`（双通道共用） | `chunker/markdown_chunker.py → to_records` | `doc_id/chunk_id/heading_path/text/char_start/char_end/tokens/overlap_tokens/fragment` |
| 召回/融合命中 | `retrieve/retriever.py` | `{id, text, metadata, score, channels[]}` |
| 精排结果 | `rerank/reranker.py` | 含 `original_rank/rerank_rank/original_score/rerank_score/degraded` |
| `AssembledContext` | `generate/context_assembler.py` | 已编号 chunk 列表（`[n]` 引用锚点）+ token 预算信息 |
| `GenerationResult` | `generate/generator.py` | answer/citations/confidence/**nli_checked**/nli_passed/degraded/cached/error |
| `PipelineResult` | `pipeline.py` | 各阶段产物 + `timings_ms` + `degraded` 聚合 |

**id 规则**：`{doc_id}::{chunk_id}`（向量与 BM25 两通道一致，是去重/融合/引用溯源的对齐键）。

---

## 2. 核心模块逐节说明

> 每节按 **职责 / 技术方案 / 实现位置 / 边界** 四要素组织。

### 2.1 配置管理 `core_line/config.py`

- **职责**：百炼 Key/模型/端点、缓存与生成模式等配置的**单一读取出口**；启动预检。
- **技术方案**：
  - 统一 Key 读取链（`BAILIAN_KEY_ENVS`）：`BAILIAN_RERANK_API_KEY →
    BAILIAN_EMBEDDING_API_KEY → DASHSCOPE_API_KEY`，`BaiLianEmbeder` 与
    `BaiLianRerankScorer` 共用 `find_bailian_api_key()`；
  - `BAILIAN_BASE_URL` 规范化：兼容模式地址（`.../compatible-mode/v1`）自动转换为
    dashscope 原生基址（`.../api/v1`），经 `base_address` 按次下发；
  - 启动预检 `preflight()`：一次性校验 key/model/base_url（含 `DS_*`、
    `RAG_CACHE_TTL_DAYS`、`GENERATOR_FAST_MODE` 合法性），缺失**报出具体变量名**；
  - `ResolvedConfig.summary()` 输出配置快照（Key 来源/模型/缓存/模式）。
- **实现位置**：`core_line/config.py → find_bailian_api_key / get_embedding_model /
  get_rerank_model / get_bailian_base_url / get_cache_enabled / get_generator_fast_mode /
  get_generator_sc_model / collect_config_problems / preflight`；
  调用方：`src/RagApp.py`（启动预检）、`embedding/bai_lian_embeder.py`、`rerank/reranker.py`。
- **边界**：只管"读配置与校验"，不持有运行态；各消费模块的构造参数可覆盖 env 值。

### 2.2 语义缓存 `core_line/cache.py`

- **职责**：嵌入向量、rerank 打分、生成结果三级缓存（sqlite 持久化），热点请求零成本命中。
- **技术方案**：
  - 单文件 sqlite（`data/cache/rag_cache.db`），`(namespace, key)` 主键 +
    `created_at` TTL；命名空间 `embedding / rerank / generation`；
  - 稳定键：`make_key(*parts)`（sha256）+ `normalize_query()`（NFKC/小写/去空白标点）+
    `text_fingerprint()`（文本 sha256 前 32 位）；
  - 线程安全（写锁 + `check_same_thread=False`）；`RAG_CACHE_ENABLED=false` 时
    `get_default_cache()` 返回 `None`，调用方零成本跳过；
  - 维护：`clear(namespace)`、`purge_expired()`、`stats()`。
- **实现位置**：`core_line/cache.py → SemanticCache / get_default_cache /
  normalize_query / make_key / text_fingerprint`。
- **边界**：只存 JSON 可序列化值；不做业务语义判断（键的构成由各消费方决定，见 §3.2）。

### 2.3 数据加载 `core_line/load_file/load_data.py`

- **职责**：扫描目录、按后缀分发解析、构建元数据，惰性产出文档流。
- **技术方案**：`load_file(path, recursive=True)` 返回**生成器**（内存同时只驻留一份文档）；
  编码三级回退 `utf-8-sig → gb18030 → utf-8(errors=replace)`；统一 `\r\n→\n`；
  50MB 上限、空文件/隐藏文件跳过；单文件异常记日志不中断。
- **实现位置**：`core_line/load_file/load_data.py → LoadData.load_file / _load_markdown /
  _read_text / _build_metadata`。
- **边界**：只负责"文本 + 元数据"，不做切分；`.pdf/.docx` 抛 `NotImplementedError`。

### 2.4 文本切分 `chunk/chunk_tool.py` + `chunker/markdown_chunker.py`

- **职责**：Markdown 结构感知切分 → `Chunk` → `to_records()` 标准记录（构建期唯一入口）。
- **技术方案**：
  - **解析**：`chunk_tool.parse_markdown` 行扫描成 `Block[]`（边界优先级：frontmatter →
    围栏代码 → ATX 标题 → GFM 表格 → 分割线 → 列表 → 引用 → 段落；代码/表格等为原子块）；
  - **装箱**：按标题分节（不跨节）→ 节内软预算 `target=400`/硬上限 `max_size=560`；
    超长原子块降级切（代码按空行分组、表格按行分组补表头、段落句子级）；
  - **重叠**：`_apply_overlap` 按上一块类型取前缀——段落句尾（句子边界吸附）、
    heading 行（上下文锚点）、代码末行（`_code_tail` 剔除围栏标记行）；预算 `overlap=60`；
  - **合并**：`_merge_small` 放宽版——跨根标题可并、前块放不下并后块（`_absorb`）、
    上限放宽到 `max_size`；阈值 `min_size=100`（target 的 1/4）；
  - **碎块兜底**：合并后仍 `< FRAGMENT_TOKENS=60` 的块 `_mark_fragments` 打
    `fragment=true`（记录字段随 metadata 入库）；
  - **形态自适应**：`_is_code_dense` 按围栏/表格字符占比（阈值 0.25）判定代码密集文档，
    自动切 `target=600/max=840/min=80`；显式传参的调用方不自适应；
  - **备选切分器**（已修复，未接入主链路）：`fixed_len_split` 按 token 定长硬切、
    `recursive_character_split` 分隔符逐级降级递归。
- **实现位置**：
  - 常量：`core_line/chunk/chunk_tool.py → CHUNK_SIZE/MAX_CHUNK_SIZE/MIN_CHUNK_SIZE/
    OVERLAP/FRAGMENT_TOKENS/CODE_DENSE_*`；
  - 解析与工具：`chunk_tool.py → parse_markdown / predic_token / split_sentences`；
  - 主链路：`chunker/markdown_chunker.py → MarkdownChunker.chunk / _pack_blocks /
    _apply_overlap / _merge_small / _mark_fragments / _apply_doc_profile`、`to_records`。
- **边界**：只产文本块与记录，不调 API；`to_records` 是切分层与索引层的唯一契约。

### 2.5 向量嵌入 `core_line/embedding/bai_lian_embeder.py`

- **职责**：文档/查询向量化（百炼 DashScope `text-embedding-v4`，默认 1024 维）。
- **技术方案**：
  - 批量分批（`MAX_BATCH_SIZE=10`）、按 `text_index` 还原顺序；指数退避重试
    （`{429,500,502,503,504}` 可重试）；
  - `text_type` 区分 document/query，查询可带英文 `instruct`；
  - **嵌入缓存**：键 `(模型, 维度, text_type, instruct, 文本指纹)`——chunk 内容不变则
    向量不变，重建索引时全量命中；未命中才走 API（`_embed_batch → _request_api`）；
  - `base_address` 按次下发（由 `config.get_bailian_base_url()` 规范化）。
- **实现位置**：`core_line/embedding/bai_lian_embeder.py → BaiLianEmbeder.embed_documents /
  embed_query / _embed_batch / _request_api`。
- **边界**：只做向量化，不建索引；Key 缺失在构造时即抛错（提示具体变量名）。

### 2.6 分词 `core_line/embedding/tokenizer.py`

- **职责**：BM25/FTS5 通道共用的中英文切词（与向量通道无关）。
- **技术方案**：jieba `lcut_for_search`（长词+子词）；英文/代码标识符整体保留并按
  `_ . -` 补子词；NFKC 归一化、小写、停用词过滤；`TOKENIZER_VERSION` 版本闸
  （规则变更必须重建 BM25 缓存）。
- **实现位置**：`core_line/embedding/tokenizer.py → tokenize / tokenize_batch / normalize`。
- **边界**：分词规则变更 → 调用方负责清缓存重建（BM25 侧有版本校验拦截）。

### 2.7 向量存储 `core_line/vector_store/chromadb_store.py`

- **职责**：Chroma 本地持久化向量库（写入/检索/维护）。
- **技术方案**：`chromadb.PersistentClient`；cosine 距离（`score = 1 - distance`）；
  id 由 `doc_id::chunk_id` 稳定推导，`upsert` 幂等（重复构建无副本）；
  批写 `WRITE_BATCH_SIZE=64`；embedding 懒加载（只读操作不需 Key）；
  支持 `where`/`where_document` 过滤下推、`score_threshold`。
- **实现位置**：`core_line/vector_store/chromadb_store.py → ChromaVector.add_records /
  query / query_by_vector / reset`。
- **边界**：维度强绑定（换模型需 `reset()` 全量重建）；**无按文档删除**（见 §6）。

### 2.8 关键词索引与检索 `retrieve/bm25_retriever.py`（备选 `fts5_retriever.py`）

- **职责**：BM25 关键词召回（与向量召回并列的 sparse 通道）。
- **技术方案**：
  - 手写 BM25（`k1=1.5, b=0.75`），Lucene 同款 idf；倒排表 + 内存统计量 `_BM25Index`；
  - 落盘 `data/bm25_index/`（`docs.pkl → manifest.json → meta.json` 三级提交）；
    `SCHEMA_VERSION` + `TOKENIZER_VERSION` 双版本校验；**增量构建**按 `doc_id`
    先删后写；
  - 备选 FTS5：`fts5_retriever.py`（SQLite 虚拟表，标题列降权 0.4），可与 BM25 互换。
- **实现位置**：`retrieve/bm25_retriever.py → BM25Retriever.build / load / search`、
  `_BM25Index`；`retrieve/fts5_retriever.py → FTS5Retriever`。
- **边界**：只认 `to_records` 契约；分数无上界，禁止与 cosine 分直接相加（融合只看排名）。

### 2.9 多路召回融合 `retrieve/retriever.py`

- **职责**：多通道并行召回 + RRF 按排名融合，输出统一候选（交重排层）。
- **技术方案**：
  - `RecallChannel(name, retriever, weight)` 通道抽象（duck-type `search()`/`query()`）；
  - RRF：`score(d) = Σ weight_c / (rrf_k + rank_c(d))`，`rrf_k=60`；同通道重复 id
    取最优排名；`channels[]` 保留各路 rank/原始分供溯源；
  - **碎块降权**：`fragment=true` 候选融合分 × `FRAGMENT_PENALTY=0.85`；
  - 排序 `score 降序、id 升序` 稳定 tie-break。
- **实现位置**：`retrieve/retriever.py → HybridRetriever.search / RecallChannel`。
- **边界**：只看排名不做分数换算；任一通道异常原样上抛（不做降级，降级在重排/生成层）。

### 2.10 重排 `rerank/reranker.py`

- **职责**：交叉编码精排（query×文本 成对打分），并提供失败降级。
- **技术方案**：
  - `BaiLianRerankScorer`（百炼 `gte-rerank-v2`，代码默认 `qwen3-rerank`）：
    `MAX_DOC_TOKENS=4000`、`batch_size=50`、指数退避重试；
  - **rerank 缓存**：键 `(模型, query 归一化, 文本指纹)`，批量部分命中；
  - **降级兜底**：打分失败/超时 → 保持 RRF 排序直进生成，结果带 `degraded=true` +
    `error`（`rerank → _degraded_result`）；打分器可注入自定义实现（Protocol）。
- **实现位置**：`rerank/reranker.py → CrossEncoderReranker.rerank / _degraded_result、
  BaiLianRerankScorer.score / _score_batch`。
- **边界**：不去重（融合层职责）；截断发生在打分排序之后。

### 2.11 上下文组装 `generate/context_assembler.py`

- **职责**：把精排结果组装成生成可用的、带引用编号的上下文。
- **技术方案**：按 id 去重 → token 预算截断（`CONTEXT_MAX_TOKENS=4096`，从后往前丢，
  优先保高分）→ 统一编号 `[1]..[n]`（生成层引用锚点）→ `citation_map()` 供引用核验；
  单条超长按 `CONTEXT_MAX_CHUNK_CHARS=2000` 截断；token 计数 tiktoken。
- **实现位置**：`generate/context_assembler.py → ContextAssembler.assemble /
  _truncate_and_number、AssembledContext.to_prompt_block / citation_map`。
- **边界**：不调 API；输入必须是 rerank 结构（含 rerank_score/rank 字段）。

### 2.12 生成与自校验 `generate/generator.py`

- **职责**：LLM 生成（DeepSeek，OpenAI 兼容 SDK）+ 三级自校验 + SSE 流式。
- **技术方案**（详细机制见 §3）：
  - **Prompt 层**：严格 JSON 输出（answer/reasoning/citations/confidence/
    evidence_sufficient）、"仅依据上下文"约束、`[n]` 引用强制；
    `PromptBuilder` 提供 CoT / SC / Reflexion / NLI / 润色五种模板；
  - **SC 投票**：并发采样 + 波次提前终止 + 次数自适应（§3.1）；`_aggregate` 多数投票；
  - **模型分层**：`GENERATOR_SC_MODEL` 轻量模型采样、主模型 `_polish` 最终润色；
  - **自校验链**：`CitationVerifier`（`[n]` → chunk 映射核验）→ `NLIChecker`
    （LLM-as-Judge 蕴含判定，鲁棒性见 §3.7）→ `Reflexion`（低置信/无效引用/明确矛盾
    时反思修正，上限 `GENERATOR_MAX_REFLEXION_ROUNDS=2`）；
  - **降级兜底**：生成失败 → "上下文摘要 + 明确未生成"（`degraded=True`，不写缓存）；
  - **SSE**：`SSEStreamer` 首包优先（conclusion → reasoning → citations →
    verification → done / error）。
- **实现位置**：`generate/generator.py → Generator.generate / generate_stream /
  _self_consistency / _sample_parallel / _sc_target / _polish / _reflexion /
  _degraded_result；DeepSeekClient.chat；NLIChecker.check / _call；
  CitationVerifier.verify；PromptBuilder；SSEStreamer`。
- **边界**：消费 `AssembledContext`；不感知检索细节；`nli_checked` 语义见 §3.7。

### 2.13 查询流水线 `core_line/pipeline.py`

- **职责**：编排"召回→融合→重排→组装→生成"全链；批量并发与 SSE 阶段进度。
- **技术方案**：
  - `run(query)`：分阶段计时（`timings_ms`），返回 `PipelineResult`；
  - `run_stream(query)`：每阶段推 `progress` 事件，生成阶段复用首包优先 SSE；
  - `run_batch(queries, max_workers=3)`：各查询整链跑线程池，天然形成
    "查询 i 的重排 与 查询 i+1 的召回并行"的阶段重叠（实测重叠系数 ~2.2x）。
- **实现位置**：`core_line/pipeline.py → RagPipeline.run / run_stream / run_batch、
  PipelineResult`。
- **边界**：组件构造注入（不负责组件生命周期）；降级标记聚合到 `PipelineResult.degraded`。

### 2.14 入口 `src/RagApp.py`

- **职责**：启动预检 → 加载 → 切分 → 打印（演示/调试入口）。
- **实现位置**：`src/RagApp.py`（`preflight()` 失败即报变量名并退出）。
- **边界**：**不编排落库与查询**——完整链路请用 `RagPipeline`（见 §6）。

---

## 3. 关键机制专题（性能与一致性）

### 3.1 SC 自适应（Self-consistency 并行 + 次数自适应 + 波次提前终止）

| 机制 | 说明 | 实现位置 |
|---|---|---|
| 次数自适应 | 简单问题 n=1、复杂问题用满 `GENERATOR_SELF_CONSISTENCY_N`（默认 3）；快速模式强制 1 | `Generator._sc_target` |
| 复杂度启发 | 长度 ≤25 字、疑问词（为什么/对比/分析/如何…）与多问检测 | `Generator._is_simple_query`（`_COMPLEX_MARKERS`） |
| 并行采样 | 线程池并发 n 次独立采样，单次失败仅告警跳过，结果按提交序收集 | `Generator._sample_parallel / _sample_once` |
| 波次提前终止 | 第一波并发 2 次；归一化答案 2/2 一致（`_answers_agree`）即收束，否则补发剩余 | `Generator._self_consistency` |
| 多数投票 | 按归一化 answer 聚票，合并 reasoning | `Generator._aggregate` |

**延迟特征**：串行 3 次 ≈ 3×单次 → 并行后 ≈ 1~2×单次（实测 3 并发较串行提速约 2.2x）；
简单问题直接 1 次采样，省 2/3 成本。

### 3.2 语义缓存与一致性（三级缓存键设计）

| 命名空间 | 键构成 | 一致性含义 |
|---|---|---|
| `embedding` | `(模型, 维度, text_type, instruct, 文本指纹)` | chunk 内容不变 → 向量不变，重建索引全量命中 |
| `rerank` | `(模型, query 归一化, 文本指纹)` | 同 query×文本×模型分数可复用 |
| `generation` | `("gen", "v2", 主模型, sc_model, fast_mode, n, query 归一化, 索引版本)` | (query, 索引版本) 不变 → 答案可直接返回 |

**一致性保障**：
1. **索引版本参与键**：`Generator.generate(index_version=)` 显式传入，缺省用
   `_context_fingerprint`（上下文 chunk id 列表哈希）——检索结果变化自然失效；
2. **缓存键加版本位**（`"v2"`）：结果 schema 变更（如新增 `nli_checked`）时 bump，
   旧条目自动失访（`Generator.generate` 中 `make_key("gen","v2",...)`）；
3. **降级结果不入缓存**：`if self._cache and not result.degraded`，避免污染；
4. **TTL**：`RAG_CACHE_TTL_DAYS`（默认 30，≤0 永不过期），过期即未命中并惰性删除
   （`SemanticCache._expired`），另有 `purge_expired()` 主动清理；
5. **开关**：`RAG_CACHE_ENABLED=false` 全局旁路；单测传 `SemanticCache(enabled=False)`
   获得冷结果。

**实测**：重复查询 3ms→2ms（cached=True）；缓存库 embedding/rerank/generation 分域增长。

### 3.3 快速模式（GENERATOR_FAST_MODE）

- **定位**：延迟最低档（简单问答/调试/批量化初筛）。
- **行为**：SC 固定 1 票、跳过 NLI、禁用 Reflexion 与润色；跳过**显式标注**
  `nli_issues=["nli_check_skipped: fast_mode：快速模式跳过 NLI 检查"]`、
  `nli_checked=False`（不做假阴性）。
- **实现位置**：`config.py → get_generator_fast_mode`；`Generator.__init__(fast_mode=)`、
  `Generator.generate`（fast_mode 分支）。
- **实测**：~2.9s/问（对比完整链 6~29s）。

### 3.4 批量流水与 SSE 流式

- **批量阶段重叠**：`RagPipeline.run_batch` 把每条查询的整链投递线程池（默认 3 worker），
  不同查询自然处于不同阶段——查询 A 在生成、B 在重排、C 在召回；
  结果按输入序返回。实测 3 查询 wall 19s vs 各链之和 42.7s（重叠系数 2.25x）。
- **SSE 流式**：`run_stream` 先推 3 个 `progress` 事件（recall_fusion/rerank/assemble
  各带 count 与 elapsed_ms），再接 `generate_stream` 的首包优先事件流；
  检索阶段异常 → `error` 事件（stage=retrieve）。
- **实现位置**：`pipeline.py → run_batch / run_stream`；`generator.py →
  generate_stream / SSEStreamer`。
- **注意**：嵌套线程池（batch worker × SC 采样池）总量可控（3×3=9 并发上限量级）。

### 3.5 构建冷启动与增量更新

| 步骤 | 冷启动行为 | 复用/幂等机制 |
|---|---|---|
| 加载 | 生成器惰性遍历，编码回退 | 单文件异常跳过不中断 |
| 切分 | 形态自适应选参；碎块标注 | 纯本地，参数可显式覆盖 |
| 嵌入 | API 批量调用（分批 10） | **嵌入缓存**：重建时 chunk 不变即命中 |
| 向量落库 | upsert 批写 | id 稳定（`doc_id::chunk_id`）重复构建无副本 |
| BM25 落库 | 分词 + 倒排重建 | 按 `doc_id` 先删后写；双版本校验拒绝脏缓存 |

**冷启动成本**（28 chunk 量级）：切分 0.2s + 嵌入 1.3s + 落库 0.6s ≈ 2s；
二次构建嵌入层全缓存命中。**触发全量重建**：换 embedding 模型/维度（`reset()`）、
分词规则变更（版本拦截）、换缓存目录（详见 `README.md` §8.2）。

### 3.6 降级容错矩阵

| 故障点 | 行为 | 标记 | 实现位置 |
|---|---|---|---|
| rerank 打分失败/超时 | 保持 RRF 排序直进生成 | `rerank["degraded"]=true` + error | `CrossEncoderReranker._degraded_result` |
| 生成全流程失败 | 返回"上下文摘要 + 明确未生成" | `GenerationResult.degraded=true` + error | `Generator._degraded_result` |
| NLI 解析/接口失败 | 修复→重调→标注跳过 | `nli_checked=false` + `nli_check_skipped:*` | `NLIChecker.check` |
| SC 单次采样失败 | 跳过该样本（并行池容忍） | votes 减少 | `Generator._sample_parallel` |
| 单文件加载失败 | 跳过继续 | 日志 | `LoadData.load_file` |
| 召回通道异常 | 原样上抛（不降级） | — | `HybridRetriever.search` |

原则：**降级必须可见**（degraded/nli_checked/issues 三处标记），且降级结果不写缓存。

### 3.7 NLI 自校验鲁棒性（修复 → 重调 → 标注 三段式）

1. **输出契约收窄**：prompt 只要求单一标签 `{"overall": "entailment|neutral|contradiction"}`
   （杜绝长 JSON 截断），`max_tokens=2048`，`response_format=json_object`
   （SC 采样同样显式开启 json_mode）；
2. **容错修复**：`_repair_json` = 直接解析 → 平衡括号扫描提取首个完整 JSON 对象
   （`_extract_first_json_object`）→ 补全截断引号/括号（`_auto_close_json`）→
   逐级截回结构边界丢弃悬空键值；
3. **带错重调**：修复失败则把上次错误附在 prompt 里重调一次（`retry_note`）；
4. **失败不静默**：仍失败 → `nli_checked=False`、`nli_passed=False`、
   `nli_issues=["nli_check_skipped: parse_error: ..."]`；**不变量：未检查 ⇒ 不得
   nli_passed=True**（消除假阴性）；Reflexion 仅在"明确判定不一致"时触发
   （`_need_reflexion` 要求 `nli_checked and not nli_passed`）；
5. **消费约定**：判"NLI 通过"应取 `nli_checked and nli_passed`；`nli_check_skipped:`
   前缀即校验缺口（建议上报告警）。

---

## 4. 模块职责边界与相互关系

### 4.1 依赖关系（自上而下，无环）

```
RagApp / 调用方
   └── pipeline.RagPipeline
         ├── retrieve.retriever.HybridRetriever ──┬── vector_store.ChromaVector ── embedding.BaiLianEmbeder
         │                                        └── retrieve.BM25Retriever ──── embedding.tokenizer
         ├── rerank.CrossEncoderReranker ──────── rerank.BaiLianRerankScorer
         ├── generate.ContextAssembler
         └── generate.Generator ───────────────── generate.NLIChecker / CitationVerifier / SSEStreamer
               │
               ├── config（Key 链/模式）        cache（三级缓存，被 embedder/reranker/generator 共用）
```

- `chunker.markdown_chunker` 仅依赖 `core_line.chunk.chunk_tool`（构建期，不在查询链上）。
- `config.py` 与 `cache.py` 是横向基建：被嵌入/重排/生成三方共用。

### 4.2 职责边界速查

| 问题 | 负责模块 | 不负责的模块 |
|---|---|---|
| 缺 API Key 怎么报 | `config.preflight`（启动即报变量名） | 各 API 模块（仅兜底二次校验） |
| 跨通道去重 | `HybridRetriever`（同 id 取最优排名） | `CrossEncoderReranker`（显式不去重） |
| 分数换算/相加 | 不做（RRF 只看排名） | 任何层都禁止直接加 cosine+BM25 分 |
| 重复写入幂等 | `ChromaVector`（upsert）/ `BM25Retriever`（先删后写） | — |
| 失败降级 | rerank 层（RRF 序）、生成层（摘要）、NLI 层（标注） | 融合层（异常上抛） |
| 引用编号一致性 | `ContextAssembler`（编号）↔ `CitationVerifier`（核验） | — |
| 缓存键构成 | 各消费方（embedding/rerank/generator） | `cache.py`（只存取） |
| 记录契约 | `to_records`（唯一出口） | 索引层不自造字段 |

### 4.3 一致性约束

1. **id 规则** `{doc_id}::{chunk_id}`：向量/BM25/引用/缓存共同对齐键，改动需全链同步；
2. **双通道同源**：两通道必须消费同一批 `records`，否则融合失效；
3. **分词/缓存版本**：`TOKENIZER_VERSION` / `SCHEMA_VERSION` 变更必须重建 BM25；
4. **向量维度**：集合维度 = embedding 输出维度（默认 1024），换模型需 `reset()`；
5. **`nli_checked` 不变量**：unchecked ⇒ passed=False（代码断言过的不变量）。

---

## 5. 工程实施注意事项

> 来自实际工程与历次调试的经验教训，按主题分组。

### 5.1 配置与外部服务

1. **配置读取单一出口**：历史 bug——注释写 `DASHSCOPE_API_KEY`、代码实际读
   `BAILIAN_EMBEDDING_API_KEY`，rerank 默认构造直接失败。所有 env 读取收敛到
   `config.py`，`.env.example` 逐项标注"哪个模块读取"，注释与代码必须一致；
2. **启动预检优先**：宁可在启动时报"缺 `DS_API_KEY`"，也不要拖到第一次调用才炸；
3. **端点转换要验证**：兼容模式 URL（`/compatible-mode/v1`）与原生协议
   （`/api/v1`）路径不同，接线后必须实测（本项目曾 404/400 各踩一次）；
4. **模型能力边界**：embedding `dimension` 有合法枚举（2048…64），测试时用非法值
   （如 8）会 400，别误判为链路故障；
5. **云 API 配额与延迟**：三个外部服务（百炼 embedding/rerank、DeepSeek）叠加，
   必备三件套——重试退避、降级路径、缓存。

### 5.2 LLM 输出契约

6. **JSON 契约越窄越稳**：NLI 曾因长 JSON 截断静默跳过检查（假阴性）；改为单一
   标签输出后未再复现。能只给标签就不给理由；
7. **容错解析 + 显式失败**：解析失败要"修复 → 重调 → 标注"三段式，绝不静默吞掉；
   状态字段（`nli_checked`）与原因字段（`nli_issues`）分开记；
8. **json_mode 全链统一**：所有要求 JSON 的调用（SC/NLI/Reflexion/润色）显式开启；
9. **改结果 schema 记得 bump 缓存键版本**（如 `"v2"`），否则旧缓存以旧结构回灌。

### 5.3 切分与索引

10. **单位一致性**：token 与字符别混用——历史 bug 把 token 数当字符下标（长文本
    尾部丢失）。参数命名/注释明确单位，本项目统一以 token 计量（`o200k_base`）；
11. **小块合并阈值要配比**：`min_size = target/4`（100/400）；过紧会产出 9~30 token
    碎块；合并上限放宽到 `max_size`；无法合并的碎块必须标注（`fragment`）并在检索侧
    降权，而不是硬塞；
12. **代码密集文档要单独装箱预算**：围栏/表格占比高时 target 400 不够装相邻代码块，
    按文档形态自适应（600/840/80）；显式传参的用户应自动退出自适应；
13. **BM25 与向量分数不可相加**：量纲不同，融合层只看排名（RRF）是刻意设计；
14. **向量侧清理要显式**：`add_records` 仅 upsert，文档变短会留"幽灵块"，
    切分参数调小后应 `delete(where={"doc_id":...})` 再写（见 §6）。

### 5.4 工程配套与验证

15. **文档与代码同步**：本项目 README 曾两次滞后（能力表写"待实现"、env 注释
    名实不符），代码改动应同 PR 更新文档/`.env.example`；
16. **测试三板斧**：假客户端验重试/降级路径（零成本）、不变量断言（unchecked⇒
    非 passed）、小样本真实调用回归（3 问：2 答 1 拒答）；
17. **基准测试先清缓存**：冷/热差异巨大（生成 6s vs 0ms），计时脚本用
    `SemanticCache().clear()` 或禁用缓存；
18. **计时工具统一**：混用 `time.time()` 与 `perf_counter` 遇系统时钟调整会出
    倒挂（曾观测到 2.5s 偏差），同一测量链用同一时钟源。

---

## 6. 当前不足与缺点

| # | 不足 | 影响 | 建议方向 |
|---|---|---|---|
| 1 | **生成延迟仍是主瓶颈**（6~29s/问）：NLI/Reflexion 为链式 LLM 调用 | 在线体验 | 校验环节并行化、分层模型加深（SC 已并行） |
| 2 | **SC 复杂度启发式偏粗**：如"原理和特点是什么"被判简单（1 票） | 多样性/置信 | 轻量分类器或按检索信号（top1 断层）动态定票 |
| 3 | **向量侧无按文档删除**：文档变短残留"幽灵块"（实测 count 29 vs 28） | 检索污染 | `ChromaVector` 补 `delete_doc(doc_id)`，构建前先清 |
| 4 | **`doc_id` 用文件名**：跨目录同名文件互相覆盖 | 正确性 | 相对路径或内容哈希做 doc_id |
| 5 | **PDF/Word 未实现**：后缀注册但抛 `NotImplementedError` | 可用性 | 接 `pypdf`/`python-docx` 或预转 Markdown |
| 6 | **`RagApp.py` 不编排落库与查询**：仅加载→切分→打印 | 易用性 | 补 ingest/query 子命令，或入口直接用 `RagPipeline` |
| 7 | **无 `requirements.txt`/`pyproject.toml`**：依赖只写在 README | 环境复现 | 补版本清单（tiktoken/dashscope/chromadb/jieba/openai/python-dotenv） |
| 8 | **备选切分器未接入主链路**：`fixed_len_split`/`recursive_character_split` 已修复但闲置 | 维护成本 | 作为降级切分接入，或移入独立工具包 |
| 9 | **overlap 对"单块节"文档天然为 0**：节内仅一个 chunk 时无相邻块可重叠 | 语义衔接 | 跨节重叠选项（尾部摘要式）可按需开启 |
| 10 | **token 计数编码器不统一**：切分用 `o200k_base`、上下文预算用 `cl100k_base` | 预算精度 | 统一为 `o200k_base`（约 25~40% 中文效率差） |
| 11 | **强依赖 3 个云 API**：无本地模型降级 | 可用性/成本 | rerank/生成可插拔本地实现（Protocol 已预留） |
| 12 | **无正式评测集**：质量评估靠 3 个手写问题 | 回归保障 | 建小评测集（正例+拒答例），量化命中率/引用有效率/拒答正确率 |

---

## 7. 符号快速定位索引

| 想找… | 去哪里 |
|---|---|
| 统一 Key 链 / 启动预检 | `core_line/config.py → find_bailian_api_key / preflight` |
| 语义缓存（增删查/TTL） | `core_line/cache.py → SemanticCache` |
| Markdown 解析成块 | `core_line/chunk/chunk_tool.py → parse_markdown` |
| 切分主流程/重叠/合并/碎块 | `chunker/markdown_chunker.py → MarkdownChunker` |
| 记录契约（record） | `chunker/markdown_chunker.py → to_records` |
| 切分参数（400/560/100/60） | `core_line/chunk/chunk_tool.py → CHUNK_SIZE 等常量` |
| 文档形态自适应 | `chunker/markdown_chunker.py → _apply_doc_profile / _is_code_dense` |
| 嵌入调用与嵌入缓存 | `core_line/embedding/bai_lian_embeder.py → _embed_batch` |
| 中文分词（BM25 用） | `core_line/embedding/tokenizer.py → tokenize` |
| 向量库写入/检索 | `core_line/vector_store/chromadb_store.py → ChromaVector` |
| BM25 建库/增量/检索 | `core_line/retrieve/bm25_retriever.py → BM25Retriever` |
| RRF 融合/碎块降权 | `core_line/retrieve/retriever.py → HybridRetriever.search` |
| 交叉重排/降级 | `core_line/rerank/reranker.py → CrossEncoderReranker.rerank` |
| 上下文编号/预算截断 | `core_line/generate/context_assembler.py → ContextAssembler.assemble` |
| SC 并行/自适应 | `core_line/generate/generator.py → _self_consistency / _sc_target` |
| NLI 容错修复/重调/标注 | `core_line/generate/generator.py → NLIChecker / _repair_json` |
| Reflexion / 润色 | `core_line/generate/generator.py → _reflexion / _polish` |
| 生成降级摘要 | `core_line/generate/generator.py → _degraded_result` |
| SSE 事件 | `core_line/generate/generator.py → SSEStreamer` |
| 批量流水/阶段重叠 | `core_line/pipeline.py → RagPipeline.run_batch` |
| 配置模板（变量→读取方） | `RAG/.env.example` |
| 业务流程/缺口清单 | `RAG/README.md`（§8.3 缺口与风险） |
