# 文本向量化与 BM25 分词（Embedding）

该目录位于 RAG 核心链路：

```
加载数据(load_file) -> 切分(chunk) -> 向量化(embedding) -> 索引构建/存储 -> 召回/检索 -> 重排 -> 生成
```

目录内包含两块彼此独立的能力，分别服务于两条召回通道：

| 文件 | 大类 | 职责 | 下游 |
|---|---|---|---|
| `bai_lian_embeder.py` | 文本向量化 | 把文本块编码成稠密向量 | `vector_store`（向量召回） |
| `tokenizer.py` | BM25 分词 | 把文本切成关键词词项 | `retrieve/bm25_retriever.py`（关键词召回） |

两者的相同点是输入都是切分后的文本、都提供“文档侧”和“查询侧”两个入口；
不同点在于输出与依据：

| 维度 | 文本向量化 | BM25 分词 |
|---|---|---|
| 输出 | 1024 维稠密向量 | 词项列表 |
| 匹配依据 | 语义相似 | 词项精确匹配 |
| 依赖 | DashScope SDK + 网络 | 本地 jieba |
| 产物存放 | Chroma（向量库） | `data/bm25_index`（缓存） |
| 失败影响 | 该批 chunk 无法写入 | 该文档关键词召回缺失 |
| 可重建性 | 可重建，但会产生调用费用 | 可重建，纯本地计算 |

下面按这两大类分别说明。

## 一、文本向量化（bai_lian_embeder.py）

基于百炼（DashScope）SDK，模型为 `text-embedding-v4`。

### 1.1 模型约束（决定实现方式的硬条件）

| 约束项 | 取值 | 对实现的影响 |
|---|---|---|
| 单次请求文本条数 | 最多 10 条 | 必须分批，`batch_size` 不能超过 10 |
| 单条文本长度 | 最多 8192 token | 不在模块内截断，依赖切分阶段控制块大小 |
| 输出维度 | 2048/1536/1024/768/512/256/128/64，默认 1024 | 初始化时校验，非法维度直接拒绝 |
| 文档 / 查询差异 | 支持 `text_type` = document / query | 文档与查询走不同的入口和参数 |
| 输出类型 | 默认 `dense` 稠密向量 | 本模块只处理稠密向量，返回 `list[float]` |

### 1.2 整体流程

```
                ┌─────────────────────────────┐
                │ 阶段 1 初始化 BaiLianEmbeder │
                │ 取 Key / 校验维度与批量参数  │
                └──────────────┬──────────────┘
                               │ 校验不通过 -> 立即抛错（fail-fast）
                               ▼
             ┌─────────────────┴─────────────────┐
             │                                   │
   embed_documents(texts)                 embed_query(text)
   （文档批量入口）                        （查询单条入口）
             │                                   │
             ▼                                   ▼
    ┌──────────────────┐                ┌──────────────────┐
    │ 阶段 2 入参校验   │                │ 阶段 2 入参校验   │
    │ 非空/类型/分条    │                │ 非空/类型         │
    └────────┬─────────┘                └────────┬─────────┘
             ▼                                   │
    ┌──────────────────┐                         │
    │ 阶段 3 分批切片   │  batch_size <= 10       │
    │ 逐批串行处理      │                         │
    └────────┬─────────┘                         │
             └───────────────┬─────────────────┘
                             ▼
                ┌───────────────────────────┐
                │ 阶段 4 组装请求参数 payload│
                │ model/input/api_key       │
                │ dimension/text_type/instruct │
                └─────────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │ 阶段 5 调用 SDK 并分类处理 │
                │ 200 -> 成功               │
                │ 429/5xx -> 退避后重试     │
                │ 其它 -> 抛错终止          │
                └─────────────┬─────────────┘
                              │ 成功
                              ▼
                ┌───────────────────────────┐
                │ 阶段 6 解析响应并还原顺序  │
                │ 按 text_index 落位         │
                │ 校验条数/越界/空缺         │
                └─────────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │ 阶段 7 汇总返回            │
                │ documents: list[list[float]]│
                │ query: list[float]         │
                └───────────────────────────┘
```

### 1.3 阶段说明与逻辑关系

#### 阶段 1：初始化与配置校验

入口：`BaiLianEmbeder.__init__`

按顺序完成三件事，任一失败即终止，不产生任何网络请求：

1. **解析 API Key**：优先使用传入的 `api_key`，否则读取环境变量 `DASHSCOPE_API_KEY`；两者都没有则抛出 `ValueError`。
2. **校验维度**：`dimension` 为 `None` 时使用模型默认值，否则必须落在支持列表中。
3. **校验批量与重试参数**：`1 <= batch_size <= 10`、`max_retries >= 1`。

配置完成后，模型名、维度、批大小、重试策略都成为实例状态，后续所有请求复用，避免每次调用重复解析。

这一阶段是后续所有阶段的前置条件：Key 缺失或参数非法时，后面的分批与调用都不会发生。

#### 阶段 2：入口分流与入参校验

模块对外暴露两个语义不同的入口，先分流再共用同一套内部实现：

| 入口 | 面向对象 | `text_type` | 返回 |
|---|---|---|---|
| `embed_documents(texts)` | 文档块，批量 | `document`（可改为 `query` 或 `None`） | `list[list[float]]` |
| `embed_query(text)` | 用户查询，单条 | `query` | `list[float]` |

校验规则：

- `embed_documents` 收到 `str` 时报 `TypeError`，提示改用 `embed_query`，避免“把字符串当字符列表逐字向量化”的隐性 bug。
- 空列表直接返回 `[]`，不发起请求。
- 每条文本必须是非空 `str`，否则抛出 `ValueError`/`TypeError`，并带上出错下标，便于定位是哪个 chunk 有问题。

查询侧固定使用 `text_type="query"`，可选传入英文 `instruct` 进一步优化检索向量；文档侧默认 `document`，因为底库文本需要偏正文式的完整表达。

#### 阶段 3：分批

文档批量入口在此阶段把 `texts` 按 `batch_size` 切成若干批：

- 切分用切片完成，保持原始顺序；
- 批次之间**串行**发送，不并发，配合 `request_interval` 起到主动限流作用；
- 每完成一批就追加到结果列表，内存中只需保留“待处理文本 + 已生成向量”；
- 每批记录一条 debug 日志（已处理 / 总量），便于长文档观察进度。

分批是“模型条数上限”与“结果顺序”两个约束的交汇点：先分批才能调通接口，按序追加才能保证最终向量与输入一一对应。

#### 阶段 4：组装请求参数

入口：`_embed_batch`

集中构造 `payload`，避免参数散落在调用处：

```
必填: model / input / api_key
可选: dimension（None 时不传，用模型默认）
      text_type（None 时不传，模型不区分文本角色）
      instruct（仅查询侧，配合 text_type="query" 使用）
```

采用“非空才传”的方式，保证不把 `None` 透传给服务端，也兼容不同版本的 SDK。

#### 阶段 5：调用与重试

同一批请求在一个循环内完成“调用 — 判定 — 退避”：

1. **网络/客户端异常**：捕获异常；未达 `max_retries` 则退避后重试，达到上限则包装成 `RuntimeError` 抛出，并保留原始异常链。
2. **HTTP 200**：视为成功，按 `request_interval` 等待后进入解析阶段。
3. **HTTP 429 / 5xx**：限流或服务端临时故障，属于可恢复错误，退避后重试。
4. **其它状态码**（鉴权失败、参数错误等）：重试无意义，立即抛出带 `status/code/message/request_id` 的 `RuntimeError`。

退避公式：

```
delay = min(retry_backoff * 2^(attempt-1), max_retry_delay)
```

即 1s、2s、4s……逐次翻倍并封顶，既给服务端恢复时间，也避免长时间阻塞主流程。

关键逻辑关系：**重试只发生在单批内部**，重试成功仍返回该批完整结果；重试耗尽则整次调用失败，不会产生“部分批次成功、部分缺失”的静默结果。

#### 阶段 6：响应解析与顺序还原

入口：`_parse`

服务端返回 `output.embeddings`，每项含 `text_index` 与 `embedding`。解析分三步：

1. **条数校验**：返回条数与请求条数不一致直接报错，防止向量与文本错位。
2. **按下标落位**：预分配长度等于请求条数的列表，用 `text_index` 把向量放回对应位置；缺失 `text_index` 时退化为返回顺序。
3. **完整性校验**：下标越界或任一位仍为空，说明响应异常，抛错而不是返回带空洞的结果。

这一步是“接口返回顺序不可依赖”的兜底，保证 `vectors[i]` 恒对应 `texts[i]`，向量库写入与召回才能对齐。

#### 阶段 7：汇总返回

- `embed_documents`：把各批结果按批序拼接，返回长度与入参一致的二维列表；
- `embed_query`：单条请求直接取第 0 个元素，返回一维向量。

返回结构直接对接向量库写入（文档）与检索（查询）两类调用。

### 1.4 异常分类

| 场景 | 处理方式 | 是否重试 |
|---|---|---|
| 缺少 API Key | 初始化即抛 `ValueError` | 否 |
| 维度/批大小/重试次数非法 | 初始化即抛 `ValueError` | 否 |
| 文本为空、类型错误 | 调用前抛 `ValueError`/`TypeError` | 否 |
| 网络异常、SDK 抛出异常 | 退避重试，耗尽后抛 `RuntimeError` | 是 |
| 429 限流、500/502/503/504 | 退避重试，耗尽后抛 `RuntimeError` | 是 |
| 鉴权失败、参数错误等 4xx | 立即抛 `RuntimeError` | 否 |
| 返回条数不符、下标越界、向量缺失 | 抛 `RuntimeError` | 否（视为数据异常） |

设计原则：**配置与入参错误尽早暴露，可恢复错误有限重试，数据不一致宁可报错也不返回错位结果。**

### 1.5 参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `api_key` | 环境变量 `DASHSCOPE_API_KEY` | 百炼 API Key |
| `model` | `text-embedding-v4` | 向量模型名 |
| `dimension` | `1024` | 输出维度，`None` 表示用模型默认 |
| `batch_size` | `10` | 单次请求条数，上限 10 |
| `max_retries` | `3` | 单批最大尝试次数（含首次） |
| `retry_backoff` | `1.0` | 退避基数（秒） |
| `max_retry_delay` | `8.0` | 单次退避上限（秒） |
| `request_interval` | `0.0` | 请求间隔（秒），用于主动限流 |

### 1.6 使用示例

```python
from core_line.embedding.bai_lian_embeder import BaiLianEmbeder

embedder = BaiLianEmbeder(dimension=1024)

# 文档侧：批量向量化，顺序与入参一致
vectors = embedder.embed_documents(["第一段", "第二段"])

# 查询侧：单条向量化，可附加英文指令
query_vec = embedder.embed_query(
    "什么是 RAG？",
    instruct="Given a question, retrieve passages that answer it",
)
```

依赖安装：

```bash
pip install dashscope
```

### 1.7 与其他模块的衔接与注意事项

1. **与切分模块的衔接**：`MarkdownChunker` 的 `target=400 / max_size=560` token 远低于 8192 上限，本模块不再做截断；若更换切分策略，需自行保证单块不超限。
2. **维度一致性**：`dimension` 一旦确定，向量库集合（collection）的维度必须同步，换维度等价于重建索引。
3. **文档/查询对称性**：检索场景建议文档用 `document`、查询用 `query`；聚类、分类等对称任务应显式传 `text_type=None`。
4. **当前边界**：只处理稠密向量（未使用 `output_type="sparse"` 与稀疏向量），未提供异步/并发接口，也未内置 token 级长度检测与截断。

### 1.8 可扩展性

#### 1.8.1 已预留的扩展点

| 扩展点 | 方式 | 影响范围 |
|---|---|---|
| 更换 embedding 模型 | 构造 `BaiLianEmbeder(model=...)` | 需同步 `dimension` 并重建向量库 |
| 调整输出维度 | `dimension` 参数（支持 64~2048） | 集合维度必须一致 |
| 调整批量/重试/限流 | `batch_size`、`max_retries`、`request_interval` | 只影响调用节奏，不影响数据 |
| 接入其他厂商 | 实现 `embed_documents` / `embed_query` 两个方法，传给 `ChromaVector(embedder=...)` | 文档与查询的 `text_type` 语义需自行保证 |

#### 1.8.2 替换向量模型的步骤

1. 实现一个类，提供 `embed_documents(texts) -> list[list[float]]` 与 `embed_query(text) -> list[float]`；
2. 确认输出维度，显式传给 `ChromaVector(dimension=...)`；
3. 若维度变化，先 `store.reset()`，再重新写入；
4. 重跑检索评估，确认 `score` 分布与阈值仍然可用。

#### 1.8.3 可新增能力与成本

| 能力 | 改动 | 成本 |
|---|---|---|
| 异步/并发向量化 | 将串行批改为线程池/信号量 | 需处理限流与重试，收益取决于 QPS |
| embedding 结果缓存 | 以 `hash(text)+model` 为键落盘 | 省重复计费，需失效策略 |
| 稀疏向量 | 本地 Chroma 不支持混合检索 | 需换向量库，见 `vector_store/README.md` |
| 多模态（图片/表格） | 换多模态模型 + 切分策略 | 改动面大，需独立链路 |
| 长度预检 | 调用前用 tokenizer 估算 | 当前依赖切分保证 ≤560，可省略 |

#### 1.8.4 不建议的扩展

- 在 `BaiLianEmbeder` 里做切分或清洗：职责重叠，难以单测；
- 运行时切换 `dimension`：集合维度固定，切换等价于重建索引，应作为离线操作。

## 二、BM25 分词（tokenizer.py）

### 2.1 定位

`tokenizer.py` 不属于向量化流程，而是关键词通道的前置步骤：BM25 依赖精确词项匹配，
建库与查询必须使用同一套切词规则，否则词项无法对齐、分数整体退化为 0。

放在本目录的原因：两条通道共用同一份文本归一约定（NFKC、去零宽字符），
集中维护可以避免“向量用一份清洗规则、BM25 用另一份”的隐性偏差。

### 2.2 处理流程

```
原始文本
  │ normalize：NFKC 全角转半角 + 去零宽字符（BOM/ZWSP/ZWJ）
  ▼
片段切分（正则，丢弃标点/空白/emoji）
  ├── ASCII 标识符与数字 → 小写 + 整体保留 + 子词补充
  └── CJK 连续串（含扩展 A 区） → jieba 搜索模式
  ▼
停用词过滤（中英文功能词）
  ▼
list[str]
```

子词拆分规则：**只有所有子片段都含字母时才拆**，因此

- `err_conn_refused` → `err_conn_refused` / `err` / `conn` / `refused`
- `v1.2.3` → `v1.2.3`（版本号保持整体，避免拆成噪声）

中文使用 `lcut_for_search`，同时产出长词与子词；重复词项保留，供 BM25 统计词频。

### 2.3 规则示例

| 输入 | 输出 |
|---|---|
| `如何配置 Chroma 的 HNSW 参数？` | `配置` / `chroma` / `hnsw` / `参数` |
| `err_conn_refused 与 ERR_TIMEOUT` | `err_conn_refused` / `err` / `conn` / `refused` / `err_timeout` / `err` / `timeout` |
| `v1.2.3 版本说明` | `v1.2.3` / `版本` / `说明` |
| `state-of-the-art 向量检索方案` | `state-of-the-art` / `state` / `art` / `向量` / `检索` / `方案` |
| `Ｃｈｒｏｍａ　全角测试` | `chroma` / `全角` / `测试` |
| `，。！` | `[]` |

### 2.4 对外接口

| 接口 | 作用 |
|---|---|
| `TOKENIZER_VERSION` | 规则版本号，变化后 BM25 缓存必须重建 |
| `STOPWORDS` | 中英文停用词并集，可扩展 |
| `normalize(text)` | NFKC 归一化 + 去零宽字符 |
| `tokenize(text, *, drop_stopwords=True)` | 单条文本切词 |
| `tokenize_batch(texts, ...)` | 批量切词，顺序与入参一致 |
| `warmup()` | 预热 jieba 词典（约 0.3~1s），建议启动时调用 |

### 2.5 版本与缓存

`TOKENIZER_VERSION` 会写入 BM25 缓存的 `meta.json`。以下改动都属于“规则变更”，必须递增版本号：

- 修改 `_SEGMENT` 正则或子词拆分条件；
- 调整 `STOPWORDS`；
- 加载领域词典（`jieba.add_word` / `load_userdict`）；
- 改变归一化策略。

版本不一致时 `BM25Retriever.load()` 会直接抛 `ValueError`，要求重建，而不是静默返回错误结果。

### 2.6 可扩展性

| 需求 | 做法 | 注意 |
|---|---|---|
| 领域停用词 | 扩展 `STOPWORDS` | 需递增版本并重建 BM25 缓存 |
| 专有名词 | `jieba.add_word("HNSW")` | 同上 |
| 同义词归一 | 在 `normalize` 中加替换表 | 会影响词项与原文字符偏移，仅用于 BM25 |
| 新语言 | 扩展 `_SEGMENT` 与切分分支 | 需补对应停用词 |
| 自定义切词 | 调用方包一层函数 | 当前 BM25 直接导入 `tokenize`，替换需改导入或后续改为依赖注入 |

边界：`tokenizer.py` 只做“文本 → 词项”，不承担打分、过滤、检索等逻辑，
保持纯函数便于单测与复用；同理，也不应在其中做 embedding 相关的向量化处理。
