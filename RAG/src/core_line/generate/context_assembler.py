"""
上下文组装（context_assembler）：将 CrossEncoderReranker.rerank() 精排后的
chunk 结果组装为生成层可直接消费的结构化上下文。

数据流：
    CrossEncoderReranker.rerank() 的 results（按 rerank_score 降序）
        -> 校验并抽取 (id, text, metadata, rerank_score) 等字段
        -> 去重（按 id，保留 rerank_rank 最靠前者）
        -> 按 token 预算截断（优先保留高 rerank_score 候选）
        -> 统一编号 [1], [2], ... 并生成引用映射
        -> 输出 AssembledContext，供 generator 消费

输入格式（与 reranker.rerank() 的 results 逐条对齐）：
    {
        "id":             str,           # "{doc_id}::{chunk_id}"
        "text":           str,           # chunk 正文
        "metadata":       dict,          # doc_id / chunk_id / heading_path 等
        "original_rank":  int,           # 原始排序（融合后输入顺序，从 1 开始）
        "rerank_rank":    int,           # 精排后排序（从 1 开始）
        "original_score": float | None,  # RRF 融合分
        "rerank_score":   float,         # 精排相关性分
        "channels":       list,          # 命中来源（可选）
    }

设计要点：
    * 职责单一：只做"组织与构建"，不调用 LLM、不做语义判断
    * 字段对齐：直接消费 reranker 输出，以 rerank_rank / rerank_score 排序截断
    * Token 预算：支持字符级近似与 tiktoken 精确计数（可选），
      超预算时按 rerank_rank 从后往前丢弃，保证至少保留 min_chunks 条
    * 引用编号：每个 chunk 分配稳定编号 [1]..[n]，生成层按编号引用并核验
    * 配置读取：从 .env 读取 CONTEXT_MAX_TOKENS / CONTEXT_MIN_CHUNKS /
      CONTEXT_MAX_CHUNK_CHARS，构造参数可覆盖
    * 异常兜底：空输入返回空上下文不抛错；非法条目显式报错

输出结构（AssembledContext）：
    AssembledContext.query        str              # 原始查询
    AssembledContext.chunks       list[ContextChunk]  # 有序 chunk（已编号）
    AssembledContext.total_chars  int              # 全部 chunk 字符数
    AssembledContext.total_tokens int              # 估算 token 数
    AssembledContext.truncated    bool             # 是否发生截断
    AssembledContext.dropped_count int             # 被丢弃的 chunk 数

    ContextChunk.index       int        # 引用编号，从 1 开始
    ContextChunk.id          str        # "{doc_id}::{chunk_id}"
    ContextChunk.text        str        # chunk 正文（可能被截短）
    ContextChunk.metadata    dict       # 元数据
    ContextChunk.rerank_score float     # 精排相关性分
    ContextChunk.original_rank int      # 精排前排名
    ContextChunk.char_count  int        # 字符数

用法（与 rerank 层串联）：
    from core_line.rerank.reranker import CrossEncoderReranker
    from core_line.generate.context_assembler import ContextAssembler

    reranker = CrossEncoderReranker()
    rerank_result = reranker.rerank(query, fused_candidates, top_k=8)

    assembler = ContextAssembler()                     # 配置自动读 .env
    ctx = assembler.assemble(query, rerank_result["results"])

    prompt_block = ctx.to_prompt_block()               # 供 generator 使用
    cite_map = ctx.citation_map()                      # {编号: chunk_id}

注意：
    * max_tokens 为"上下文区域"预算，不含系统提示与用户问题本身；
    * 若单条 chunk 超预算且是唯一候选，会截断其 text 并标记 truncated=True；
    * token 计数默认字符数 / 2 近似；安装 tiktoken 后自动切换精确计数；
    * 本模块不依赖任何外部 API，可独立测试。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODER = None


def _estimate_tokens(text: str) -> int:
    """估算 token 数：有 tiktoken 用精确计数，否则按字符数 / 2 近似。"""
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 2)


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """组装后的单条上下文 chunk（已编号，供生成层引用）。

    param:
        index:         引用编号，从 1 开始，生成层用 [index] 引用
        id:            唯一键 "{doc_id}::{chunk_id}"
        text:          chunk 正文（可能被截短）
        metadata:      元数据（doc_id / chunk_id / heading_path 等）
        rerank_score:  精排相关性分（CrossEncoderReranker 打分）
        original_rank: 精排前排名（融合后输入顺序）
        char_count:    text 字符数
    """

    index: int
    id: str
    text: str
    metadata: dict[str, Any]
    rerank_score: float
    original_rank: int
    char_count: int


@dataclass(slots=True)
class AssembledContext:
    """组装完成的上下文，供 generator 直接消费。

    param:
        query:        原始查询文本
        chunks:       有序 chunk 列表（已编号，按 rerank_rank）
        total_chars:  全部 chunk 字符数
        total_tokens: 估算 token 数
        truncated:    是否发生了截断（丢弃或文本截短）
        dropped_count:被丢弃的 chunk 数
    """

    query: str
    chunks: list[ContextChunk]
    total_chars: int = 0
    total_tokens: int = 0
    truncated: bool = False
    dropped_count: int = 0

    @property
    def is_empty(self) -> bool:
        """上下文是否为空（无可用于回答的证据）。"""
        return len(self.chunks) == 0

    def to_prompt_block(self) -> str:
        """渲染为带编号的提示文本块，供拼接进 LLM prompt。"""
        if not self.chunks:
            return "（无检索到的上下文）"
        parts: list[str] = []
        for c in self.chunks:
            source = c.metadata.get("doc_id", "unknown")
            parts.append(f"[{c.index}] (来源: {source})\n{c.text}")
        return "\n\n".join(parts)

    def citation_map(self) -> dict[int, str]:
        """返回 {编号: chunk_id} 映射，供引用核验使用。"""
        return {c.index: c.id for c in self.chunks}


class ContextAssembler:
    """上下文组装器：对 reranker 精排结果做去重、截断、编号、预算控制。

    配置优先级：构造参数 > 环境变量（.env） > 默认值

    param:
        max_tokens:      上下文 token 预算，缺省读 CONTEXT_MAX_TOKENS
        min_chunks:      至少保留 chunk 数，缺省读 CONTEXT_MIN_CHUNKS
        max_chunk_chars: 单条 chunk 字符上限，缺省读 CONTEXT_MAX_CHUNK_CHARS
        reserve_chars:   预留字符数（安全边际），默认 200
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        min_chunks: int | None = None,
        max_chunk_chars: int | None = None,
        reserve_chars: int = 200,
    ):
        self.max_tokens = max_tokens or int(os.getenv("CONTEXT_MAX_TOKENS", "4096"))
        self.min_chunks = min_chunks if min_chunks is not None else int(os.getenv("CONTEXT_MIN_CHUNKS", "1"))
        self.max_chunk_chars = max_chunk_chars or int(os.getenv("CONTEXT_MAX_CHUNK_CHARS", "2000"))
        self.reserve_chars = reserve_chars

        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正数，当前为 {self.max_tokens}")
        if self.min_chunks < 0:
            raise ValueError(f"min_chunks 不能为负数，当前为 {self.min_chunks}")
        if self.max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars 必须为正数，当前为 {self.max_chunk_chars}")

    # ---------------------------------------------------------------- 对外接口
    def assemble(
        self,
        query: str,
        rerank_results: Sequence[Mapping[str, Any]],
    ) -> AssembledContext:
        """将 CrossEncoderReranker.rerank() 的 results 组装为结构化上下文。

        param:
            query:          原始查询文本
            rerank_results: reranker.rerank() 输出的 results 列表，
                            每条含 id / text / metadata / rerank_score / rerank_rank 等

        return:
            AssembledContext（已编号、去重、按预算截断）
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        if isinstance(rerank_results, (str, bytes)):
            raise TypeError("rerank_results 应为列表，单条请包成 [item]")

        rerank_results = list(rerank_results)
        if not rerank_results:
            logger.info("精排结果为空，返回空上下文")
            return AssembledContext(query=query, chunks=[])

        validated = self._validate_rerank_results(rerank_results)
        deduped = self._deduplicate(validated)
        chunks, dropped, truncated = self._truncate_and_number(deduped)

        total_chars = sum(c.char_count for c in chunks)
        total_tokens = _estimate_tokens("".join(c.text for c in chunks))

        return AssembledContext(
            query=query,
            chunks=chunks,
            total_chars=total_chars,
            total_tokens=total_tokens,
            truncated=truncated or dropped > 0,
            dropped_count=dropped,
        )

    # ---------------------------------------------------------------- 内部实现
    @staticmethod
    def _validate_rerank_results(
        items: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """校验 reranker 输出格式并抽取必要字段。"""
        validated: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise TypeError(f"results[{i}] 不是映射类型，实际为 {type(item).__name__}")

            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'results[{i}] 缺少有效的 "text" 字段')

            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f'results[{i}] 缺少有效的 "id" 字段')

            metadata = item.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise TypeError(f'results[{i}].metadata 应为映射，实际为 {type(metadata).__name__}')

            rerank_score = item.get("rerank_score", 0.0)
            if not isinstance(rerank_score, (int, float)):
                raise TypeError(f'results[{i}].rerank_score 应为数值，实际为 {type(rerank_score).__name__}')

            rerank_rank = item.get("rerank_rank", i + 1)
            original_rank = item.get("original_rank", i + 1)

            validated.append(
                {
                    "id": item_id,
                    "text": text,
                    "metadata": dict(metadata) if metadata else {},
                    "rerank_score": float(rerank_score),
                    "rerank_rank": int(rerank_rank),
                    "original_rank": int(original_rank),
                }
            )
        return validated

    @staticmethod
    def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 id 去重，保留 rerank_rank 最靠前的。"""
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        # 已按 rerank_score 降序排列（reranker 保证），先出现的优先
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                result.append(item)
            else:
                logger.debug("重复 id 已跳过: %s", item["id"])
        return result

    def _truncate_and_number(
        self, items: list[dict[str, Any]]
    ) -> tuple[list[ContextChunk], int, bool]:
        """按 token 预算截断（从后往前丢弃）并编号。

        return:
            (chunks, dropped_count, text_truncated)
        """
        chunks: list[ContextChunk] = []
        used_tokens = 0
        dropped = 0
        text_truncated = False

        for item in items:
            text = item["text"]
            if len(text) > self.max_chunk_chars:
                text = text[: self.max_chunk_chars]
                text_truncated = True
                logger.debug("chunk %s 超长，截断至 %d 字符", item["id"], self.max_chunk_chars)

            token_cost = _estimate_tokens(text)

            if chunks and used_tokens + token_cost > self.max_tokens:
                if len(chunks) >= self.min_chunks:
                    dropped += 1
                    continue
                # 低于 min_chunks，强制保留但截断 text
                remaining = max(0, self.max_tokens - used_tokens - self.reserve_chars)
                if remaining <= 0:
                    remaining = 100
                text = text[: remaining * 2]
                token_cost = _estimate_tokens(text)
                text_truncated = True
                logger.debug("chunk %s 为保证 min_chunks 被截短", item["id"])

            used_tokens += token_cost
            chunks.append(
                ContextChunk(
                    index=len(chunks) + 1,
                    id=item["id"],
                    text=text,
                    metadata=item["metadata"],
                    rerank_score=item["rerank_score"],
                    original_rank=item["original_rank"],
                    char_count=len(text),
                )
            )

        return chunks, dropped, text_truncated
