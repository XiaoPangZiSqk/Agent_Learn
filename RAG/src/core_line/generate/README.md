# 上下文组装与生成校验（Generate）

该模块位于 RAG 核心链路末端：

```
加载数据(load_file) -> 切分(chunk) -> 向量化(embedding) -> 索引构建/存储 -> 召回/检索(retrieve) -> 重排(rerank) -> 生成(generate)
```

目录内包含两个彼此独立、职责清晰的子模块：

| 文件 | 大类 | 职责 | 上游输入 | 下游输出 |
|---|---|---|---|---|
| `context_assembler.py` | 上下文组装 | 对精排结果做去重、截断、编号、预算控制 | `CrossEncoderReranker.rerank()` 的 `results` | `AssembledContext` |
| `generator.py` | 生成与自校验 | CoT / Self-consistency / Reflexion 生成，引用核验、NLI 检查、SSE 流式输出 | `AssembledContext` | `GenerationResult` / SSE 事件流 |

两者的协作关系：

```
CrossEncoderReranker.rerank()                     Generator.generate()
        │ results（按 rerank_score 降序）                  ▲
        ▼                                                  │
ContextAssembler.assemble()  ─── AssembledContext ─────────┘
        │                           │
        │ to_prompt_block()         │ citation_map()
        ▼                           ▼
  LLM prompt 拼接             CitationVerifier 核验
```

## 一、设计目标与边界

| 目标 | 做法 | 边界 |
|---|---|---|
| 上下文与生成解耦 | `context_assembler` 不调 LLM，`generator` 不做检索 | 两者通过 `AssembledContext` 数据结构衔接 |
| 消费精排输出 | 字段与 `reranker.rerank()` 的 `results` 逐条对齐 | 不兼容非 rerank 格式，入参显式校验 |
| 严格证据约束 | 系统提示强制"仅依据上下文"，`evidence_sufficient` 标记 | 不能消除 LLM 幻觉，仅做结构化约束与事后校验 |
| 引用可溯源 | 每个 chunk 分配稳定编号 `[1]..[n]`，核验引用有效性 | 引用粒度为 chunk 级，不做句级定位 |
| 首包优先流式 | SSE 首包输出结论，后续包补推理/引用/校验 | 生成完成后才开始推流，非 token 级流式 |
| 配置集中管理 | 全部配置从 `.env` 读取，构造参数可覆盖 | 不管理 `.env` 文件本身 |

当前**不包含**：token 级流式生成、本地 NLI 模型、向量数据库操作、HTTP 服务层封装。

## 二、上下文组装（context_assembler.py）

### 2.1 输入格式

直接消费 `CrossEncoderReranker.rerank()` 返回的 `results` 列表：

```python
[
    {
        "id":             str,           # "{doc_id}::{chunk_id}"
        "text":           str,           # chunk 正文
        "metadata":       dict,          # doc_id / chunk_id / heading_path 等
        "original_rank":  int,           # 精排前排名（融合后输入顺序）
        "rerank_rank":    int,           # 精排后排名
        "original_score": float | None,  # RRF 融合分
        "rerank_score":   float,         # 精排相关性分
        "channels":       list,          # 命中来源（可选）
    },
    ...
]
```

### 2.2 组装流程

```
rerank_results（按 rerank_score 降序）
        │
        ▼
┌───────────────────────────────────┐
│ 1. 校验字段                        │  id / text 必填且非空，类型显式检查
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 2. 按 id 去重                      │  保留 rerank_rank 最靠前者
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 3. 单条超长截断                    │  超过 max_chunk_chars 截短 text
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 4. Token 预算截断                  │  从后往前丢弃，保证 ≥ min_chunks
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 5. 统一编号 [1], [2], ...          │  生成 citation_map 供核验
└──────────────┬────────────────────┘
               ▼
        AssembledContext
```

### 2.3 输出结构

**AssembledContext**

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | `str` | 原始查询文本 |
| `chunks` | `list[ContextChunk]` | 有序 chunk 列表（已编号） |
| `total_chars` | `int` | 全部 chunk 字符数 |
| `total_tokens` | `int` | 估算 token 数 |
| `truncated` | `bool` | 是否发生截断 |
| `dropped_count` | `int` | 被丢弃的 chunk 数 |
| `is_empty` | `bool`（属性） | 上下文是否为空 |

**ContextChunk**

| 字段 | 类型 | 说明 |
|---|---|---|
| `index` | `int` | 引用编号，从 1 开始 |
| `id` | `str` | 唯一键 `"{doc_id}::{chunk_id}"` |
| `text` | `str` | chunk 正文（可能被截短） |
| `metadata` | `dict` | 元数据 |
| `rerank_score` | `float` | 精排相关性分 |
| `original_rank` | `int` | 精排前排名 |
| `char_count` | `int` | 字符数 |

**方法**

| 方法 | 返回 | 说明 |
|---|---|---|
| `to_prompt_block()` | `str` | 渲染为带编号的提示文本块 |
| `citation_map()` | `dict[int, str]` | `{编号: chunk_id}` 映射 |

### 2.4 Token 计数

| 条件 | 策略 | 精度 |
|---|---|---|
| 已安装 `tiktoken` | `cl100k_base` 编码器精确计数 | 精确 |
| 未安装 `tiktoken` | 字符数 / 2 近似 | 中英混合场景够用 |

## 三、生成与自校验（generator.py）

### 3.1 整体流程

```
query + AssembledContext
        │
        ▼
┌───────────────────────────────────┐
│ 阶段 1  Self-consistency 多次采样   │  默认 n=3，temperature>0
│         每次独立调用 DeepSeek API   │
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 阶段 2  多数投票聚合               │  选票数最多的 answer，合并 reasoning
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 阶段 3  引用核验                   │  [n] 引用映射到 chunk 编号
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 阶段 4  NLI 一致性检查             │  LLM-as-Judge 判断蕴含/中立/矛盾
└──────────────┬────────────────────┘
               ▼
┌───────────────────────────────────┐
│ 阶段 5  Reflexion 迭代（必要时）    │  低置信度/引用无效/NLI 不通过时触发
│         反思修正，最多 N 轮         │
└──────────────┬────────────────────┘
               ▼
        GenerationResult
```

### 3.2 提示策略

| 策略 | 提示构建方法 | 作用 |
|---|---|---|
| **CoT** | `build_cot_prompt()` | 链式思考：识别证据 → 逐步推理 → 得出结论 → 标注引用 |
| **Self-consistency** | `build_self_consistency_prompt()` | 多次独立采样 + 多数投票，降低随机性 |
| **Reflexion** | `build_reflexion_prompt()` | 反思校验问题并修正答案 |

所有提示共享统一的系统约束（`SYSTEM_PROMPT`）：

1. 仅依据给定上下文回答，不使用外部知识
2. 证据不足时明确说明"证据不足，无法回答"
3. 事实性陈述必须标注 `[编号]` 引用
4. 结论先行，推理细节后补
5. 严格 JSON 格式输出

### 3.3 结构化输出 schema

LLM 被强制输出以下 JSON 结构：

```json
{
    "answer": "最终答案（结论优先）",
    "reasoning": ["推理步骤1", "推理步骤2"],
    "citations": [{"index": 1, "text": "被引用的原文片段"}],
    "confidence": 0.92,
    "evidence_sufficient": true
}
```

### 3.4 引用核验

`CitationVerifier` 从 `answer` 中提取 `[n]` 格式引用，逐一映射到 `AssembledContext.citation_map()`：

| 核验结果 | 含义 | 后续动作 |
|---|---|---|
| `valid: true` | 引用编号对应真实 chunk | 正常 |
| `valid: false` | 引用编号超出上下文范围 | 标记无效，触发 Reflexion |

### 3.5 NLI 一致性检查

`NLIChecker` 将答案断言视为假设、上下文视为前提，由 LLM-as-Judge 判断三类关系：

| 标签 | 含义 | 后续动作 |
|---|---|---|
| `entailment` | 前提蕴含假设（有据可依） | 通过 |
| `neutral` | 无法判断（可能超出上下文） | 记录问题，可触发 Reflexion |
| `contradiction` | 与上下文矛盾 | 记录问题，触发 Reflexion |

### 3.6 Reflexion 触发条件

| 条件 | 说明 |
|---|---|
| `nli_passed = False` | NLI 检查未通过 |
| 存在无效引用 | `CitationVerifier.has_invalid()` 为 `True` |
| `confidence < confidence_threshold` | 置信度低于阈值（默认 0.5） |

触发后迭代反思修正，最多 `max_reflexion_rounds` 轮；全部通过则提前退出。

### 3.7 SSE 流式输出

事件推送顺序（**首包优先**）：

```
event: conclusion          ← 首包：结论 + 置信度 + 证据充分性
data: {"answer": "...", "confidence": 0.92, "evidence_sufficient": true}

event: reasoning           ← CoT 推理步骤
data: {"steps": ["步骤1", "步骤2"]}

event: citations           ← 引用核验结果
data: {"citations": [{"index": 1, "valid": true, "chunk_id": "..."}]}

event: verification        ← NLI 校验结果
data: {"nli_passed": true, "issues": []}

event: done                ← 完成统计
data: {"total_rounds": 1, "self_consistency_votes": 3}

event: error               ← 错误（异常时）
data: {"error": "...", "stage": "..."}
```

`SSEEvent.to_sse()` 输出标准 `text/event-stream` 格式，可直接用于 HTTP 响应。

## 四、配置（.env）

所有配置从项目根目录 `.env` 读取，构造参数可覆盖。

### 4.1 DeepSeek API

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DS_API_KEY` | （必填） | DeepSeek API Key |
| `DS_BASE_URL` | `https://api.deepseek.com` | 接口地址 |
| `DS_MODEL` | `deepseek-chat` | 模型名 |
| `DS_TEMPERATURE` | `0.3` | 采样温度 |

### 4.2 上下文组装

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `CONTEXT_MAX_TOKENS` | `4096` | 上下文 token 预算 |
| `CONTEXT_MIN_CHUNKS` | `1` | 至少保留 chunk 数 |
| `CONTEXT_MAX_CHUNK_CHARS` | `2000` | 单条 chunk 字符上限 |

### 4.3 生成参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GENERATOR_SELF_CONSISTENCY_N` | `3` | Self-consistency 采样次数 |
| `GENERATOR_MAX_TOKENS` | `2048` | 单次生成最大 token 数 |
| `GENERATOR_MAX_REFLEXION_ROUNDS` | `2` | Reflexion 最大迭代轮数 |
| `GENERATOR_CONFIDENCE_THRESHOLD` | `0.5` | 低于此置信度触发 Reflexion |

## 五、核心类与方法总览

### ContextAssembler

| 方法 | 说明 | 返回 |
|---|---|---|
| `assemble(query, rerank_results)` | 组装精排结果为结构化上下文 | `AssembledContext` |
| `to_prompt_block()` | 渲染提示文本块（`AssembledContext` 方法） | `str` |
| `citation_map()` | 生成引用映射（`AssembledContext` 方法） | `dict[int, str]` |

### Generator

| 方法 | 说明 | 返回 |
|---|---|---|
| `generate(query, context)` | 完整生成流程（同步） | `GenerationResult` |
| `generate_stream(query, context)` | SSE 流式生成 | `Iterator[SSEEvent]` |

### GenerationResult

| 字段 | 类型 | 说明 |
|---|---|---|
| `answer` | `str` | 最终答案 |
| `reasoning_steps` | `list[str]` | CoT 推理步骤 |
| `citations` | `list[dict]` | 引用核验结果 |
| `confidence` | `float` | 置信度 0.0~1.0 |
| `evidence_sufficient` | `bool` | 证据是否充分 |
| `nli_passed` | `bool` | NLI 是否通过 |
| `nli_issues` | `list[str]` | NLI 问题描述 |
| `self_consistency_votes` | `int` | 采样次数 |
| `reflexion_rounds` | `int` | 反思迭代轮数 |

## 六、使用示例

### 6.1 完整链路（重排 → 组装 → 生成）

```python
from core_line.rerank.reranker import CrossEncoderReranker
from core_line.generate.context_assembler import ContextAssembler
from core_line.generate.generator import Generator

# 1. 精排
reranker = CrossEncoderReranker()
rerank_result = reranker.rerank(query, fused_candidates, top_k=8)

# 2. 上下文组装
assembler = ContextAssembler()
ctx = assembler.assemble(query, rerank_result["results"])

# 3. 生成与自校验
generator = Generator()
result = generator.generate(query, ctx)

print(result.answer)
print(f"置信度: {result.confidence}, NLI 通过: {result.nli_passed}")
```

### 6.2 SSE 流式输出

```python
for event in generator.generate_stream(query, ctx):
    print(event.to_sse())
    # event: conclusion
    # data: {"answer": "HNSW 可通过 metadata 配置...", "confidence": 0.92, "evidence_sufficient": true}
    # ...
```

### 6.3 手动构造上下文（不依赖 reranker）

```python
from core_line.generate.context_assembler import ContextAssembler

assembler = ContextAssembler(max_tokens=2048, min_chunks=2)
ctx = assembler.assemble("如何配置 HNSW？", [
    {"id": "a.md::0", "text": "Chroma 使用 HNSW...", "metadata": {"doc_id": "a.md"}},
])
```

## 七、可扩展性

| 扩展点 | 当前状态 | 建议做法 | 注意事项 |
|---|---|---|---|
| NLI 检查 | LLM-as-Judge | 替换为本地 NLI 模型（如 `cross-encoder/nli-deberta`） | 保持 `check(answer, context) -> (bool, list[str])` 接口 |
| 流式输出 | 生成完成后推流 | 改用 `chat_stream` 实现 token 级流式 | Self-consistency 多次采样与 token 级流式有冲突 |
| 引用粒度 | chunk 级 `[n]` | 扩展为句级引用 `[n:m]` | 需要 chunk 内句子切分与对齐 |
| 聚合策略 | 多数投票 | 加权投票（按 confidence 加权） | 或改用 LLM 聚合多候选 |
| SSE 传输 | `to_sse()` 字符串 | 对接 FastAPI `StreamingResponse` / Flask | 保持事件格式不变即可 |
| 多轮对话 | 单轮问答 | `messages` 列表传入历史 | 需扩展 `PromptBuilder` 支持多轮 |

扩展原则：

1. **保持 `AssembledContext` 结构稳定**：`generator` 全部依赖此结构，变更需同步两处；
2. **SSE 事件类型不随意增删**：前端消费依赖事件名；
3. **配置只从 `.env` 读**：新增配置项保持 `模块前缀_参数名` 的命名约定。

## 八、限制与注意事项

1. **Self-consistency 消耗大**：`n=3` 意味着 3 次 API 调用 + 1 次 NLI 检查 + 可能的 Reflexion 轮次，成本与延迟成倍增加。
2. **非 token 级流式**：`generate_stream` 先完成全部生成与校验后再推流，首包虽"优先"但存在等待时间。
3. **NLI 检查本身调用 LLM**：会产生额外 API 开销；可注入 `temperature=0.0` 降低成本但不减少调用次数。
4. **JSON 解析容错有限**：LLM 输出非 JSON 时降级为 `answer=原文, confidence=0.3`，不会重试解析。
5. **证据不足是声明而非保证**：`evidence_sufficient` 由 LLM 自评，不能完全消除幻觉，需结合 NLI 与引用核验共同把关。
6. **上下文预算不含提示词**：`CONTEXT_MAX_TOKENS` 仅限制 chunk 正文，系统提示与问题不计入。
7. **reflexion 不会无限迭代**：达到 `max_reflexion_rounds` 后返回最后一次修正结果，不保证完全通过。

## 九、验证清单

| 检查项 | 期望 |
|---|---|
| `ContextAssembler().max_tokens` | 与 `.env` 中 `CONTEXT_MAX_TOKENS` 一致 |
| `DeepSeekClient().model` | 与 `.env` 中 `DS_MODEL` 一致 |
| `assemble(query, [])` | 返回 `is_empty=True` 的空上下文，不抛错 |
| `assemble(query, 重复id列表)` | 去重后 chunk 数 < 输入条数 |
| `CitationVerifier().verify("[1]", ctx)` | `valid=True` |
| `CitationVerifier().verify("[99]", ctx)` | `valid=False` |
| `SSEStreamer.stream_result(result)` 首事件 | `event == "conclusion"` |
| `Generator._parse_response('{"answer":"x"}')` | `answer == "x"` |
| `.env` 缺少 `DS_API_KEY` 时构造 `Generator()` | 抛出 `ValueError` |
