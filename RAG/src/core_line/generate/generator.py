"""
生成与自校验（generator）：基于 OpenAI SDK 调用 DeepSeek API，结合
CoT / Self-consistency / Reflexion 三种提示策略完成知识问答的生成、
自校验与 SSE 流式输出。

数据流：
    AssembledContext（ContextAssembler.assemble() 输出，源自 reranker 精排结果）
        -> PromptBuilder 构建 CoT 提示（含引用编号与回答约束）
        -> DeepSeekClient 调用 LLM 生成结构化答案（JSON）
        -> Self-consistency 多次采样 + 多数投票聚合
        -> CitationVerifier 引用核验（[n] 引用是否指向真实上下文）
        -> NLIChecker 自然语言推理一致性检查（答案与上下文是否矛盾）
        -> Reflexion 自我反思迭代（发现不一致时修正）
        -> SSEStreamer 流式输出（首包优先：先输出结论，再补充细节）

配置（从 .env 读取，构造参数可覆盖）：
    DS_API_KEY       DeepSeek API Key（必填）
    DS_BASE_URL      接口地址（默认 https://api.deepseek.com）
    DS_MODEL         模型名（默认 deepseek-chat）
    DS_TEMPERATURE   采样温度（默认 0.3）
    GENERATOR_SELF_CONSISTENCY_N    Self-consistency 采样次数上限（默认 3，简单问题自动降为 1）
    GENERATOR_MAX_TOKENS            单次生成最大 token 数（默认 2048）
    GENERATOR_MAX_REFLEXION_ROUNDS  Reflexion 最大迭代轮数（默认 2）
    GENERATOR_CONFIDENCE_THRESHOLD  置信度阈值（默认 0.5）
    GENERATOR_SC_MODEL              SC 采样轻量模型（缺省同主模型；配置见 core_line/config.py）
    GENERATOR_FAST_MODE             快速模式（true 时 SC=1、跳过 NLI/Reflexion/润色，缺省 false）
    RAG_CACHE_*                     语义缓存（生成结果按 (query 归一化, 索引版本) 缓存，core_line/cache.py）

设计要点：
    * 模块化：DeepSeekClient / PromptBuilder / CitationVerifier /
      NLIChecker / SSEStreamer 各司其职，可独立测试与替换
    * 结构化输出：LLM 输出严格 JSON schema，含 answer / reasoning /
      citations / confidence / evidence_sufficient 字段
    * 证据约束：提示中强制"仅依据上下文作答，证据不足时明确说明"，
      生成后二次校验 evidence_sufficient 与 citations 的一致性
    * 引用核验：answer 中出现的 [n] 引用必须映射到 AssembledContext 的
      chunk 编号，无法映射的引用标记为 invalid
    * NLI 一致性：将答案视为假设、上下文视为前提，LLM-as-Judge 做蕴含判断，
      发现矛盾或不蕴含时触发 Reflexion 修正；输出契约收窄为单一标签杜绝截断，
      解析失败先容错修复再重调一次，仍失败降级为 nli_check_skipped 标注
      （nli_checked=False，不伪装 nli_passed=True）
    * Self-consistency：并发采样 + 波次提前终止 + 次数自适应（简单问题 n=1，
      复杂问题用满 n），对最终答案做多数投票
    * Reflexion：对低置信度 / NLI 不通过 / 无效引用的答案自动反思修正
    * SSE 流式：事件类型 conclusion / reasoning / citations /
      verification / done / error，首包立即输出结论（first-packet priority）
    * 错误处理：API 限流/5xx 指数退避重试，不可重试错误立即抛出

SSE 事件流（text/event-stream）：
    event: conclusion
    data: {"answer": "...", "confidence": 0.92, "evidence_sufficient": true}

    event: reasoning
    data: {"steps": ["步骤1...", "步骤2..."]}

    event: citations
    data: {"citations": [{"index": 1, "valid": true, "chunk_id": "..."}]}

    event: verification
    data: {"nli_checked": true, "nli_passed": true, "issues": []}

    event: done
    data: {"total_rounds": 1, "self_consistency_votes": 3}

    event: error
    data: {"error": "...", "stage": "..."}

用法（与 rerank + 上下文组装串联）：
    from core_line.rerank.reranker import CrossEncoderReranker
    from core_line.generate.context_assembler import ContextAssembler
    from core_line.generate.generator import Generator

    rerank_result = CrossEncoderReranker().rerank(query, fused, top_k=8)
    ctx = ContextAssembler().assemble(query, rerank_result["results"])

    generator = Generator()
    result = generator.generate(query, ctx)          # 非流式
    for event in generator.generate_stream(query, ctx):  # SSE 流式
        print(event.to_sse())

注意：
    * DeepSeek API 与 OpenAI API 兼容，使用 openai SDK 的 base_url 指向 DeepSeek；
    * Self-consistency 与 Reflexion 会多次调用 API，消耗更多 token 与时间；
    * NLI 检查使用 LLM-as-Judge（同样调用 DeepSeek），可替换为本地 NLI 模型；
    * 流式模式下先完成全部生成与校验，再按首包优先顺序推送事件。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Mapping, Sequence

from dotenv import load_dotenv

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam
except ImportError:
    OpenAI = None  # type: ignore[assignment]
    ChatCompletionMessageParam = dict  # type: ignore[assignment,misc]

from core_line.cache import SemanticCache, get_default_cache
from core_line.cache import make_key, normalize_query, text_fingerprint
from core_line.config import get_generator_fast_mode, get_generator_sc_model
from core_line.generate.context_assembler import AssembledContext

load_dotenv()
logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ================================================================ 数据结构
@dataclass(slots=True)
class SSEEvent:
    """SSE 流式事件。

    param:
        event: 事件类型（conclusion / reasoning / citations / verification / done / error）
        data:  事件负载（JSON 可序列化 dict）
    """

    event: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@dataclass(slots=True)
class GenerationResult:
    """生成与自校验的最终结果。

    param:
        answer:               最终答案文本
        reasoning_steps:      CoT 推理步骤列表
        citations:            引用列表 [{"index": int, "valid": bool, "chunk_id": str|None}]
        confidence:           置信度 0.0~1.0
        evidence_sufficient:  上下文证据是否充分
        nli_checked:          NLI 检查是否实际执行并得出判定（False=未检查）
        nli_passed:           NLI 一致性检查是否通过（nli_checked=False 时恒为 False，
                              与 nli_checked 联合区分"检查通过"与"未检查"）
        nli_issues:           NLI 检查发现的问题（未检查时含 nli_check_skipped: 原因）
        self_consistency_votes: Self-consistency 采样次数
        reflexion_rounds:     Reflexion 迭代轮数
        raw_responses:        各次采样的原始 LLM 响应（调试用）
        degraded:             是否为降级兜底结果（生成服务失败时的上下文摘要）
        cached:               是否来自语义缓存
        error:                降级原因（degraded=True 时非空）
    """

    answer: str
    reasoning_steps: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    evidence_sufficient: bool = False
    nli_checked: bool = True
    nli_passed: bool = True
    nli_issues: list[str] = field(default_factory=list)
    self_consistency_votes: int = 1
    reflexion_rounds: int = 0
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning_steps": self.reasoning_steps,
            "citations": self.citations,
            "confidence": self.confidence,
            "evidence_sufficient": self.evidence_sufficient,
            "nli_checked": self.nli_checked,
            "nli_passed": self.nli_passed,
            "nli_issues": self.nli_issues,
            "self_consistency_votes": self.self_consistency_votes,
            "reflexion_rounds": self.reflexion_rounds,
            "degraded": self.degraded,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationResult":
        """从缓存字典恢复结果（raw_responses 不持久化，标记 cached=True）。"""
        valid = {f.name for f in fields(cls)} - {"raw_responses", "cached"}
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(cached=True, **kwargs)


# ================================================================ DeepSeek 客户端
class DeepSeekClient:
    """DeepSeek API 客户端（OpenAI SDK 兼容），带重试与错误处理。

    配置从 .env 读取：DS_API_KEY / DS_BASE_URL / DS_MODEL

    param:
        api_key:    API Key，缺省读 DS_API_KEY
        base_url:   接口地址，缺省读 DS_BASE_URL
        model:      模型名，缺省读 DS_MODEL
        max_retries: 最大重试次数，默认 3
        retry_backoff: 退避基数（秒），默认 1.0
        max_retry_delay: 单次退避上限（秒），默认 8.0
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-flash"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        *,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 8.0,
    ):
        if OpenAI is None:
            raise ImportError(
                "缺少 openai SDK：请安装 openai（pip install openai）"
            )

        self.api_key = api_key or os.getenv("DS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "缺少 DeepSeek API Key：请在 .env 中设置 DS_API_KEY 或构造时传入 api_key"
            )

        self.base_url = base_url or os.getenv("DS_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("DS_MODEL", self.DEFAULT_MODEL)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_retry_delay = max_retry_delay

        if max_retries < 1:
            raise ValueError("max_retries 至少为 1")

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info("DeepSeekClient 初始化: model=%s, base_url=%s", self.model, self.base_url)

    def chat(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
        model: str | None = None,
    ) -> dict[str, Any]:
        """同步调用，返回 {"content": str, "usage": dict}。

        temperature 缺省读 .env 的 DS_TEMPERATURE，默认 0.3。
        model 为按次覆盖的模型名（如 SC 采样用轻量模型），缺省用 self.model。
        """
        if temperature is None:
            temperature = float(os.getenv("DS_TEMPERATURE", "0.3"))
        if max_tokens is None:
            max_tokens = int(os.getenv("GENERATOR_MAX_TOKENS", "2048"))

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model or self.model,
                    "messages": list(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                return {"content": content, "usage": usage}

            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS:
                    raise RuntimeError(f"DeepSeek API 调用失败（不可重试）: {exc}") from exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"DeepSeek API 调用失败（已尝试 {self.max_retries} 次）: {exc}"
                    ) from exc
                delay = min(self.retry_backoff * (2 ** (attempt - 1)), self.max_retry_delay)
                logger.warning("DeepSeek API 请求失败（第 %d 次），%.1fs 后重试: %s", attempt, delay, exc)
                time.sleep(delay)

        raise RuntimeError("DeepSeek API 调用失败：不应到达此处")

    def chat_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式调用，逐段产出文本增量。"""
        if temperature is None:
            temperature = float(os.getenv("DS_TEMPERATURE", "0.3"))
        if max_tokens is None:
            max_tokens = int(os.getenv("GENERATOR_MAX_TOKENS", "2048"))

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            raise RuntimeError(f"DeepSeek 流式调用失败: {exc}") from exc


# ================================================================ Prompt 构建
class PromptBuilder:
    """提示构建器：CoT / Self-consistency / Reflexion / NLI 校验提示。

    所有提示均遵循核心约束：仅依据上下文作答，证据不足时明确说明。
    """

    SYSTEM_PROMPT = (
        "你是一个严格的知识问答助手。请遵守以下规则：\n"
        "1. **仅依据给定的上下文回答**，不使用任何外部知识或猜测。\n"
        "2. 如果上下文中的证据不足以回答问题，必须明确说明"
        '"根据提供的上下文，证据不足，无法回答该问题"，不要编造答案。\n'
        "3. 回答中的事实性陈述必须标注引用来源，格式为 [编号]，编号对应上下文中的 chunk 编号。\n"
        "4. 先给出结论（简明直接的答案），再补充推理细节。\n"
        "5. 以 JSON 格式输出，包含以下字段：\n"
        '   - "answer": 最终答案（结论优先，简洁明确）\n'
        '   - "reasoning": 推理步骤数组（每步一个字符串，体现 CoT 链式思考）\n'
        '   - "citations": 引用数组 [{"index": 编号, "text": 被引用的关键原文片段}]\n'
        '   - "confidence": 置信度（0.0~1.0，基于证据充分程度评估）\n'
        '   - "evidence_sufficient": 布尔值，上下文证据是否充分\n'
    )

    @classmethod
    def build_cot_prompt(cls, query: str, context: AssembledContext) -> list[ChatCompletionMessageParam]:
        """构建 CoT（Chain-of-Thought）提示。"""
        context_block = context.to_prompt_block()
        user_msg = (
            f"## 上下文\n{context_block}\n\n"
            f"## 问题\n{query}\n\n"
            "请按以下步骤思考并回答：\n"
            "1. 仔细阅读上下文，识别与问题相关的证据\n"
            "2. 逐步推理，将证据与问题关联\n"
            "3. 得出结论（如果证据不足，明确说明）\n"
            "4. 标注引用来源\n"
            "5. 以 JSON 格式输出结果"
        )
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    @classmethod
    def build_self_consistency_prompt(cls, query: str, context: AssembledContext) -> list[ChatCompletionMessageParam]:
        """构建 Self-consistency 采样提示（增加多样性引导）。"""
        context_block = context.to_prompt_block()
        user_msg = (
            f"## 上下文\n{context_block}\n\n"
            f"## 问题\n{query}\n\n"
            "请独立思考并回答这个问题。注意从不同角度分析上下文中的证据，"
            "确保推理过程完整。以 JSON 格式输出结果。"
        )
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    @classmethod
    def build_polish_prompt(
        cls,
        query: str,
        context: AssembledContext,
        draft_answer: str,
        draft_reasoning: list[str],
    ) -> list[ChatCompletionMessageParam]:
        """构建最终润色提示（模型分层：SC 采样用轻量模型，润色用强模型）。"""
        context_block = context.to_prompt_block()
        steps_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(draft_reasoning)) or "（无）"
        user_msg = (
            f"## 上下文\n{context_block}\n\n"
            f"## 问题\n{query}\n\n"
            f"## 草稿答案（由轻量模型多次投票得出）\n{draft_answer}\n\n"
            f"## 草稿推理步骤\n{steps_text}\n\n"
            "请基于上下文核对草稿答案的事实与引用，消除重复与冗余，润色为最终答案。"
            "草稿正确的部分保持不变，错误或无据的部分必须修正。以 JSON 格式输出结果。"
        )
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    @classmethod
    def build_reflexion_prompt(
        cls,
        query: str,
        context: AssembledContext,
        previous_answer: str,
        issues: list[str],
    ) -> list[ChatCompletionMessageParam]:
        """构建 Reflexion 反思修正提示。"""
        context_block = context.to_prompt_block()
        issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "（无具体问题描述）"
        user_msg = (
            f"## 上下文\n{context_block}\n\n"
            f"## 问题\n{query}\n\n"
            f"## 你之前的回答\n{previous_answer}\n\n"
            f"## 自我校验发现的问题\n{issues_text}\n\n"
            "请反思上述问题，重新审视上下文证据，修正你的回答。"
            "如果之前的回答确实正确，可以保持不变但需要给出理由。"
            "以 JSON 格式输出修正后的结果。"
        )
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    @classmethod
    def build_nli_prompt(
        cls,
        answer: str,
        context: AssembledContext,
        *,
        retry_note: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """构建 NLI 一致性检查提示（前提=上下文，假设=答案断言）。

        输出契约刻意收窄为单一标签（仅 overall 字段），从源头消除长 JSON 截断；
        retry_note 用于解析失败后的重调，附上上次错误信息约束输出格式。
        """
        context_block = context.to_prompt_block()
        user_msg = (
            f"## 前提（上下文）\n{context_block}\n\n"
            f"## 假设（答案中的断言）\n{answer}\n\n"
            "请判断「假设」中的断言是否被「前提」蕴含：\n"
            '- "entailment": 前提蕴含假设（答案有据可依）\n'
            '- "neutral": 前提无法判断假设（答案存在无据断言）\n'
            '- "contradiction": 前提与假设矛盾（答案与上下文冲突）\n\n'
            '只输出一个 JSON 对象，仅含 "overall" 一个字段，取值只能是 '
            '"entailment" / "neutral" / "contradiction" 之一。'
            "不要输出任何其他字段、解释或推理过程。"
        )
        if retry_note:
            user_msg += f"\n\n{retry_note}"
        system_msg = (
            "你是一个自然语言推理（NLI）专家。"
            "你的任务是判断假设是否被前提蕴含、与前提矛盾或中立。"
            "只输出 JSON，不要额外解释。"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


# ================================================================ 引用核验
class CitationVerifier:
    """引用核验器：验证答案中的 [n] 引用是否指向有效上下文 chunk。"""

    _CITE_PATTERN = re.compile(r"\[(\d+)\]")

    def verify(self, answer: str, context: AssembledContext) -> list[dict[str, Any]]:
        """提取答案中的引用并逐个核验。

        return:
            [{"index": int, "valid": bool, "chunk_id": str | None}]
        """
        citation_map = context.citation_map()
        found_indices = self._CITE_PATTERN.findall(answer)

        results: list[dict[str, Any]] = []
        seen: set[int] = set()
        for idx_str in found_indices:
            idx = int(idx_str)
            if idx in seen:
                continue
            seen.add(idx)
            chunk_id = citation_map.get(idx)
            results.append({"index": idx, "valid": chunk_id is not None, "chunk_id": chunk_id})
        return results

    def has_invalid(self, citations: list[dict[str, Any]]) -> bool:
        return any(not c["valid"] for c in citations)


# ================================================================ JSON 容错修复
def _extract_first_json_object(text: str) -> str | None:
    """平衡括号扫描，提取首个完整 JSON 对象（忽略字符串内的括号/转义）。"""
    depth, start = 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def _auto_close_json(text: str) -> str:
    """补全截断 JSON：闭合未终止的字符串与未闭合的括号，并去掉结尾悬空的 , :。"""
    stack: list[str] = []
    in_str, esc = False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    base = text.rstrip()
    while base and base[-1] in ",:" and not in_str:
        base = base[:-1].rstrip()   # 去掉结尾悬空的 , :（字符串内的不动）
    return base + ('"' if in_str else "") + "".join(reversed(stack))


def _repair_json(content: str) -> Any | None:
    """容错解析：直接解析 -> 提取首个完整 JSON 对象 -> 补全截断的括号/引号
    （补全失败则截回上一个结构边界丢弃悬空键值后再闭合，逐级重试）。"""
    text = (content or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    obj_text = _extract_first_json_object(text)
    if obj_text:
        try:
            return json.loads(obj_text)
        except json.JSONDecodeError:
            pass

    seen: set[str] = set()
    for seed in (text, obj_text or ""):
        base = seed
        for _ in range(6):
            candidate = _auto_close_json(base)
            if candidate and candidate not in seen:
                seen.add(candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            # 截回最后一个结构边界（丢掉悬空的 "key": / 半截值）再闭合
            idx = max(base.rfind(","), base.rfind("{"), base.rfind("["))
            if idx < 0:
                break
            base = base[:idx] if base[idx] == "," else base[:idx + 1]
    return None


# ================================================================ NLI 一致性检查
class NLIChecker:
    """NLI 一致性检查器（LLM-as-Judge）：判断答案是否被上下文蕴含。

    解析失败时的三段式处理（不静默）：
        1. 容错修复（_repair_json：提取首个完整 JSON / 补全截断括号引号）；
        2. 修复失败则带错误信息重调一次 NLI；
        3. 仍失败才降级：nli_checked=False，issues 记录
           "nli_check_skipped: parse_error: ..."，让上层可见校验缺口。

    param:
        client: DeepSeekClient 实例
    """

    VALID_LABELS = ("entailment", "neutral", "contradiction")
    #: 单次 NLI 调用的输出上限（输出仅单一标签，预留充足余量杜绝截断）
    MAX_TOKENS = 2048

    def __init__(self, client: DeepSeekClient):
        self.client = client

    def check(self, answer: str, context: AssembledContext) -> tuple[bool, list[str], bool]:
        """执行 NLI 检查。

        return:
            (passed, issues, checked)：passed=True 表示判定一致；issues 为问题描述列表；
            checked=False 表示校验未能完成（解析/接口失败），passed 此时为 False
            ——上层应以 (checked, passed) 区分"未检查"与"检查通过"。
        """
        if not answer.strip():
            return True, [], True    # 无可校验断言，视同通过

        data, err = self._call(answer, context, retry_note=None)
        if data is None:
            note = (f"（上次输出无法解析：{err}。请严格按要求只输出 JSON 对象，"
                    '仅含 "overall" 字段）')
            data, err = self._call(answer, context, retry_note=note)
        if data is None:
            logger.warning("NLI 校验未完成（修复与重调均失败），降级并标注缺口: %s", err)
            return False, [f"nli_check_skipped: {err}"], False

        issues = self._issues_from(data)
        overall = str(data.get("overall", "")).strip().lower()
        passed = overall == "entailment" and not issues
        if not passed:
            logger.info("NLI 未通过: overall=%s, issues=%d 条", overall, len(issues))
        return passed, issues, True

    def _call(
        self, answer: str, context: AssembledContext, *, retry_note: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """单次 NLI 调用 + 容错解析，返回 (data, error)。"""
        messages = PromptBuilder.build_nli_prompt(answer, context, retry_note=retry_note)
        try:
            resp = self.client.chat(
                messages, temperature=0.0, max_tokens=self.MAX_TOKENS, json_mode=True
            )
        except RuntimeError as exc:
            return None, f"api_error: {exc}"
        data = _repair_json(resp.get("content", ""))
        if not isinstance(data, dict) or not str(data.get("overall", "")).strip():
            return None, "parse_error: 响应不是 JSON 对象或缺少 overall 标签"
        return data, None

    @classmethod
    def _issues_from(cls, data: dict[str, Any]) -> list[str]:
        """从判定结果提取问题列表：兼容逐条 results 明细与单一 overall 标签两种输出。"""
        issues: list[str] = []
        results = data.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, Mapping):
                    continue
                label = str(item.get("label", "neutral")).lower()
                claim = item.get("claim", "")
                explanation = item.get("explanation", "")
                if label == "contradiction":
                    issues.append(f"矛盾: {claim} — {explanation}")
                elif label == "neutral":
                    issues.append(f"无据: {claim} — {explanation}")
            return issues
        overall = str(data.get("overall", "")).strip().lower()
        if overall == "contradiction":
            issues.append("NLI 判定为 contradiction：答案与上下文矛盾")
        elif overall != "entailment":
            issues.append(f"NLI 判定为 {overall or 'unknown'}：答案存在无据断言或无法判断")
        return issues


# ================================================================ SSE 流式输出
class SSEStreamer:
    """SSE 流式输出器：首包优先，先输出结论再补充细节。

    事件推送顺序：
        1. conclusion  （首包：结论 + 置信度 + 证据充分性）
        2. reasoning   （CoT 推理步骤）
        3. citations   （引用核验结果）
        4. verification（NLI 校验结果）
        5. done        （完成事件，含统计信息）
    """

    @staticmethod
    def stream_result(result: GenerationResult) -> Iterator[SSEEvent]:
        """按首包优先顺序产出 SSE 事件。"""
        yield SSEEvent(
            event="conclusion",
            data={
                "answer": result.answer,
                "confidence": result.confidence,
                "evidence_sufficient": result.evidence_sufficient,
                "degraded": result.degraded,
                "cached": result.cached,
            },
        )
        if result.reasoning_steps:
            yield SSEEvent(event="reasoning", data={"steps": result.reasoning_steps})
        if result.citations:
            yield SSEEvent(event="citations", data={"citations": result.citations})
        yield SSEEvent(
            event="verification",
            data={"nli_checked": result.nli_checked, "nli_passed": result.nli_passed,
                  "issues": result.nli_issues},
        )
        yield SSEEvent(
            event="done",
            data={
                "total_rounds": 1 + result.reflexion_rounds,
                "self_consistency_votes": result.self_consistency_votes,
                "cached": result.cached,
                "degraded": result.degraded,
                "error": result.error,
            },
        )

    @staticmethod
    def stream_error(error: str, stage: str = "unknown") -> Iterator[SSEEvent]:
        yield SSEEvent(event="error", data={"error": error, "stage": stage})


# ================================================================ 生成器主类
class Generator:
    """生成与自校验主类：编排 CoT / Self-consistency / Reflexion 完整流程。

    配置从 .env 读取，构造参数可覆盖：
        DS_API_KEY / DS_BASE_URL / DS_MODEL / DS_TEMPERATURE
        GENERATOR_SELF_CONSISTENCY_N / GENERATOR_MAX_TOKENS
        GENERATOR_MAX_REFLEXION_ROUNDS / GENERATOR_CONFIDENCE_THRESHOLD

    param:
        api_key:              DeepSeek API Key（缺省读 DS_API_KEY）
        base_url:             接口地址（缺省读 DS_BASE_URL）
        model:                模型名（缺省读 DS_MODEL）
        self_consistency_n:   Self-consistency 采样次数（缺省读 GENERATOR_SELF_CONSISTENCY_N）
        temperature:          采样温度（缺省读 DS_TEMPERATURE）
        max_tokens:           单次生成最大 token 数（缺省读 GENERATOR_MAX_TOKENS）
        max_reflexion_rounds: Reflexion 最大迭代轮数（缺省读 GENERATOR_MAX_REFLEXION_ROUNDS）
        confidence_threshold: 置信度阈值（缺省读 GENERATOR_CONFIDENCE_THRESHOLD）
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        self_consistency_n: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_reflexion_rounds: int | None = None,
        confidence_threshold: float | None = None,
        sc_model: str | None = None,
        fast_mode: bool | None = None,
        cache: SemanticCache | None = None,
    ):
        self.client = DeepSeekClient(api_key=api_key, base_url=base_url, model=model)
        self.nli_checker = NLIChecker(self.client)
        self.citation_verifier = CitationVerifier()

        self.self_consistency_n = self_consistency_n or int(os.getenv("GENERATOR_SELF_CONSISTENCY_N", "3"))
        self.temperature = temperature if temperature is not None else float(os.getenv("DS_TEMPERATURE", "0.3"))
        self.max_tokens = max_tokens or int(os.getenv("GENERATOR_MAX_TOKENS", "2048"))
        self.max_reflexion_rounds = max_reflexion_rounds or int(os.getenv("GENERATOR_MAX_REFLEXION_ROUNDS", "2"))
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else float(
            os.getenv("GENERATOR_CONFIDENCE_THRESHOLD", "0.5")
        )

        # 模型分层：SC 采样可用轻量模型（GENERATOR_SC_MODEL），最终润色用主模型（强模型）
        self.sc_model = sc_model if sc_model is not None else get_generator_sc_model()
        # 快速模式（GENERATOR_FAST_MODE）：SC=1、跳过 NLI/Reflexion/润色，延迟最低档
        self.fast_mode = fast_mode if fast_mode is not None else get_generator_fast_mode()

        # 语义缓存：(query 归一化, 索引版本) -> 生成结果；None 走默认缓存（RAG_CACHE_ENABLED）
        self._cache = get_default_cache() if cache is None else cache

        if self.self_consistency_n < 1:
            raise ValueError("self_consistency_n 至少为 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 必须在 0.0~1.0 之间")

    # ---------------------------------------------------------------- 同步生成
    def generate(
        self,
        query: str,
        context: AssembledContext,
        *,
        index_version: str | None = None,
    ) -> GenerationResult:
        """完整生成流程：语义缓存 -> Self-consistency -> 润色 -> 引用核验 -> NLI -> Reflexion。

        param:
            query:         用户查询
            context:       组装后的上下文
            index_version: 索引版本标识（缓存键组成部分）；缺省用上下文 chunk id 指纹代替，
                           检索结果变化会自然失效旧缓存

        return:
            GenerationResult。生成服务失败时降级返回"检索到的上下文摘要 + 明确未生成"
            （degraded=True），而不是全链路报错。
        """
        self._validate_input(query, context)

        cache_key = make_key(
            "gen", "v2", self.client.model, self.sc_model, self.fast_mode,
            self.self_consistency_n, normalize_query(query),
            index_version or self._context_fingerprint(context),
        )
        if self._cache:
            hit = self._cache.get_json("generation", cache_key)
            if isinstance(hit, dict) and "answer" in hit:
                logger.info("生成缓存命中（key %s…）", cache_key[:12])
                return GenerationResult.from_dict(hit)

        try:
            samples = self._self_consistency(query, context)
            best = self._aggregate(samples)

            # 模型分层：SC 采样用轻量模型时，由主模型（强模型）做最终润色
            if self._need_polish():
                best = self._polish(query, context, best)

            citations = self.citation_verifier.verify(best["answer"], context)
            if self.fast_mode:
                nli_passed, nli_issues, nli_checked = (
                    False, ["nli_check_skipped: fast_mode：快速模式跳过 NLI 检查"], False
                )
            else:
                nli_passed, nli_issues, nli_checked = self.nli_checker.check(
                    best["answer"], context
                )

            reflexion_rounds = 0
            if (
                not self.fast_mode
                and self.max_reflexion_rounds > 0
                and self._need_reflexion(best, nli_passed, citations, nli_checked)
            ):
                (best, citations, nli_passed, nli_issues, nli_checked,
                 reflexion_rounds) = self._reflexion(
                    query, context, best, citations, nli_passed, nli_issues, nli_checked
                )

            result = GenerationResult(
                answer=best["answer"],
                reasoning_steps=best.get("reasoning", []),
                citations=citations,
                confidence=best.get("confidence", 0.0),
                evidence_sufficient=best.get("evidence_sufficient", False),
                nli_checked=nli_checked,
                nli_passed=nli_passed,
                nli_issues=nli_issues,
                self_consistency_votes=len(samples),
                reflexion_rounds=reflexion_rounds,
                raw_responses=samples,
            )
        except Exception as exc:
            logger.error("生成流程失败，降级返回上下文摘要: %s", exc)
            result = self._degraded_result(query, context, exc)

        if self._cache and not result.degraded:
            self._cache.set_json("generation", cache_key, result.to_dict())
        return result

    # ---------------------------------------------------------------- 流式生成
    def generate_stream(
        self,
        query: str,
        context: AssembledContext,
        *,
        index_version: str | None = None,
    ) -> Iterator[SSEEvent]:
        """SSE 流式生成：执行完整流程后按首包优先顺序推送事件。

        首包（conclusion）在最终答案确定后立即推送。
        """
        try:
            result = self.generate(query, context, index_version=index_version)
        except Exception as exc:
            logger.error("生成失败: %s", exc)
            yield from SSEStreamer.stream_error(str(exc), stage="generate")
            return

        yield from SSEStreamer.stream_result(result)

    # ---------------------------------------------------------------- 内部方法
    def _validate_input(self, query: str, context: AssembledContext) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        if not isinstance(context, AssembledContext):
            raise TypeError(f"context 应为 AssembledContext，实际为 {type(context).__name__}")

    # ---------------------------------------------------------------- Self-consistency（并行 + 自适应）
    _COMPLEX_MARKERS = re.compile(
        r"为什么|为何|怎么|如何|怎样|对比|比较|区别|差异|优缺点|分析|评估|总结|归纳|推理|证明|"
        r"权衡|取舍|各有什么|有哪些方法|最佳实践|为什么"
    )

    def _sc_target(self, query: str) -> int:
        """自适应采样次数：简单问题 1 次，复杂问题用满 self_consistency_n；快速模式固定 1。"""
        if self.fast_mode or self.self_consistency_n <= 1:
            return 1
        return self.self_consistency_n if not self._is_simple_query(query) else 1

    @classmethod
    def _is_simple_query(cls, query: str) -> bool:
        """启发式判断简单事实型问题（短、无复杂疑问词、单问）。"""
        text = query.strip()
        if text.count("？") + text.count("?") >= 2:
            return False
        if len(text) > 25:
            return False
        return not cls._COMPLEX_MARKERS.search(text)

    @staticmethod
    def _answer_key(answer: str) -> str:
        """答案归一化键：去引用编号后去空白/标点小写，用于一致性判断。"""
        return normalize_query(re.sub(r"\[\d+\]", "", answer or ""))

    def _answers_agree(self, samples: list[dict[str, Any]], need: int = 2) -> bool:
        """是否存在 need 个样本给出一致答案（归一化后）。"""
        counts: dict[str, int] = {}
        for s in samples:
            key = self._answer_key(s.get("answer", ""))
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= need:
                return True
        return False

    def _sample_once(self, query: str, context: AssembledContext) -> dict[str, Any]:
        """单次独立采样（SC 采样按 sc_model 分层，缺省用主模型）。"""
        messages = PromptBuilder.build_self_consistency_prompt(query, context)
        resp = self.client.chat(
            messages, temperature=self.temperature, max_tokens=self.max_tokens,
            model=self.sc_model, json_mode=True,
        )
        parsed = self._parse_response(resp["content"])
        parsed["_raw"] = resp["content"]
        return parsed

    def _sample_parallel(self, query: str, context: AssembledContext, count: int) -> list[dict[str, Any]]:
        """并发执行 count 次采样，成功结果按提交顺序返回（单次失败仅告警跳过）。"""
        if count <= 0:
            return []
        if count == 1:
            try:
                return [self._sample_once(query, context)]
            except Exception as exc:
                logger.warning("Self-consistency 采样失败: %s", exc)
                return []
        samples: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(self._sample_once, query, context) for _ in range(count)]
            for i, future in enumerate(futures, start=1):
                try:
                    samples.append(future.result())
                    logger.debug("Self-consistency 并行采样 %d/%d 完成", i, count)
                except Exception as exc:
                    logger.warning("Self-consistency 第 %d 次采样失败: %s", i, exc)
        return samples

    def _self_consistency(self, query: str, context: AssembledContext) -> list[dict[str, Any]]:
        """Self-consistency：并发采样 + 波次提前终止 + 次数自适应。

        第一波并发采 min(2, target) 次；若 2/2 归一化答案一致（高置信一致）提前终止，
        否则补发剩余采样；总请求次数不超过 _sc_target()，延迟从串行 n 倍降为约 1~2 倍。
        """
        target = self._sc_target(query)
        fired = min(2, target)
        samples = self._sample_parallel(query, context, fired)

        if target > fired and len(samples) >= 2 and self._answers_agree(samples, need=2):
            logger.info("Self-consistency 提前终止：%d 次采样答案一致（目标 %d）", len(samples), target)
            return samples

        rest = target - fired
        if rest > 0:
            samples.extend(self._sample_parallel(query, context, rest))

        if not samples:
            raise RuntimeError("Self-consistency 全部采样失败，无法生成答案")
        logger.info("Self-consistency 完成：%d 次有效采样（目标 %d）", len(samples), target)
        return samples

    # ---------------------------------------------------------------- 模型分层（润色）
    def _need_polish(self) -> bool:
        """SC 用轻量模型且非快速模式时，由主模型做最终润色。"""
        return bool(self.sc_model) and self.sc_model != self.client.model and not self.fast_mode

    def _polish(self, query: str, context: AssembledContext, best: dict[str, Any]) -> dict[str, Any]:
        """强模型最终润色（失败时保留聚合结果，不影响主流程）。"""
        messages = PromptBuilder.build_polish_prompt(
            query, context, best.get("answer", ""), best.get("reasoning", [])
        )
        try:
            resp = self.client.chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            polished = self._parse_response(resp["content"])
            polished["_raw"] = resp["content"]
            logger.info("最终润色完成（模型 %s）", self.client.model)
            return polished
        except Exception as exc:
            logger.warning("最终润色失败，保留聚合结果: %s", exc)
            return best

    # ---------------------------------------------------------------- 降级兜底
    @staticmethod
    def _context_fingerprint(context: AssembledContext) -> str:
        """上下文指纹（索引版本的缺省代理）：chunk id 列表的稳定哈希。"""
        return text_fingerprint(",".join(c.id for c in context.chunks))

    @staticmethod
    def _degraded_result(
        query: str, context: AssembledContext, exc: Exception
    ) -> GenerationResult:
        """生成失败降级：返回检索到的上下文摘要 + 明确未生成，而非全链路报错。"""
        lines = [
            "【生成服务暂时不可用，本次未生成正式回答】",
            f"原因：{type(exc).__name__}: {exc}",
            "",
            "以下为检索到的上下文摘要（仅供人工参考）：",
        ]
        for c in context.chunks[:5]:
            snippet = c.text.replace("\n", " ")[:120]
            source = c.metadata.get("doc_id", "unknown")
            lines.append(f"[{c.index}] (来源: {source}) {snippet}…")
        if not context.chunks:
            lines.append("（无检索到的上下文）")
        return GenerationResult(
            answer="\n".join(lines),
            reasoning_steps=[f"生成降级（{type(exc).__name__}）：已回退为上下文摘要输出"],
            citations=[],
            confidence=0.0,
            evidence_sufficient=False,
            nli_checked=False,
            nli_passed=False,
            nli_issues=["nli_check_skipped: degraded_mode：降级模式未执行生成与自校验"],
            self_consistency_votes=0,
            reflexion_rounds=0,
            raw_responses=[],
            degraded=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _aggregate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """多数投票聚合：选择出现次数最多的 answer，合并 reasoning。"""
        if len(samples) == 1:
            return samples[0]

        vote_map: dict[str, list[dict[str, Any]]] = {}
        for s in samples:
            key = s.get("answer", "").strip()
            vote_map.setdefault(key, []).append(s)

        best_key = max(vote_map, key=lambda k: len(vote_map[k]))
        best_group = vote_map[best_key]
        best = dict(best_group[0])

        all_steps: list[str] = []
        for s in best_group:
            for step in s.get("reasoning", []):
                if step not in all_steps:
                    all_steps.append(step)
        best["reasoning"] = all_steps

        logger.info("Self-consistency 聚合: %d 个候选，最高票 %d 票", len(vote_map), len(best_group))
        return best

    def _need_reflexion(
        self,
        best: dict[str, Any],
        nli_passed: bool,
        citations: list[dict[str, Any]],
        nli_checked: bool = True,
    ) -> bool:
        if nli_checked and not nli_passed:
            return True     # 明确判定不一致才触发；未检查只标注缺口不触发
        if self.citation_verifier.has_invalid(citations):
            return True
        if best.get("confidence", 1.0) < self.confidence_threshold:
            return True
        return False

    def _reflexion(
        self,
        query: str,
        context: AssembledContext,
        best: dict[str, Any],
        citations: list[dict[str, Any]],
        nli_passed: bool,
        nli_issues: list[str],
        nli_checked: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], bool, list[str], bool, int]:
        """Reflexion 迭代：反思修正直到通过或达到最大轮数。"""
        current = best
        current_citations = citations
        current_nli_passed = nli_passed
        current_nli_issues = nli_issues
        current_nli_checked = nli_checked
        rounds = 0

        for _ in range(self.max_reflexion_rounds):
            rounds += 1
            issues = list(current_nli_issues)
            if self.citation_verifier.has_invalid(current_citations):
                invalid = [c for c in current_citations if not c["valid"]]
                issues.append(f"无效引用: {[c['index'] for c in invalid]}")
            if current.get("confidence", 1.0) < self.confidence_threshold:
                issues.append(f"置信度过低: {current.get('confidence', 0.0):.2f}")

            messages = PromptBuilder.build_reflexion_prompt(
                query, context, current.get("answer", ""), issues
            )
            try:
                resp = self.client.chat(
                    messages, temperature=self.temperature, max_tokens=self.max_tokens
                )
                revised = self._parse_response(resp["content"])
                revised["_raw"] = resp["content"]
            except Exception as exc:
                logger.warning("Reflexion 第 %d 轮修正失败: %s", rounds, exc)
                break

            current_citations = self.citation_verifier.verify(revised["answer"], context)
            (current_nli_passed, current_nli_issues,
             current_nli_checked) = self.nli_checker.check(revised["answer"], context)
            current = revised

            logger.info(
                "Reflexion 第 %d 轮: nli_checked=%s nli_passed=%s, confidence=%.2f",
                rounds, current_nli_checked, current_nli_passed,
                current.get("confidence", 0.0),
            )

            if (
                (current_nli_passed or not current_nli_checked)
                and not self.citation_verifier.has_invalid(current_citations)
                and current.get("confidence", 0.0) >= self.confidence_threshold
            ):
                break

        return (current, current_citations, current_nli_passed,
                current_nli_issues, current_nli_checked, rounds)

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        """解析 LLM JSON 响应：直接解析 -> 容错修复（首个完整对象/补全截断）-> 原文兜底。"""
        content = content.strip()
        data = _repair_json(content)
        if isinstance(data, dict):
            return Generator._normalize_response(data)

        logger.warning("LLM 响应不是有效 JSON，使用原始文本作为 answer")
        return {
            "answer": content,
            "reasoning": [],
            "citations": [],
            "confidence": 0.3,
            "evidence_sufficient": False,
        }

    @staticmethod
    def _normalize_response(data: dict[str, Any]) -> dict[str, Any]:
        """归一化响应字段，确保必需字段存在且类型正确。"""
        answer = data.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        reasoning = data.get("reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        if not isinstance(reasoning, list):
            reasoning = []

        raw_citations = data.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []

        confidence = data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        evidence_sufficient = bool(data.get("evidence_sufficient", False))

        return {
            "answer": answer,
            "reasoning": reasoning,
            "citations": raw_citations,
            "confidence": confidence,
            "evidence_sufficient": evidence_sufficient,
        }
