"""
百炼（DashScope）文本向量化

模型：text-embedding-v4（Qwen3-Embedding 系列）
    * 单次请求最多 10 条文本
    * 单条文本最长 8192 token（chunk 需控制在该上限内，超长会被截断/报错）
    * 支持维度：2048/1536/1024/768/512/256/128/64，默认 1024
    * 支持 text_type 区分文档与查询，查询侧还可附加英文指令（instruct）

用法：
    embedder = BaiLianEmbeder(api_key="sk-xxx", dimension=1024)
    vectors = embedder.embed_documents(["第一段", "第二段"])
    query_vec = embedder.embed_query("什么是 RAG？")
"""

from __future__ import annotations

import logging
import os
import time
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# 可重试状态码：限流 + 服务端错误
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

class BaiLianEmbeder:
    """基于百炼 SDK 的文本向量化器（text-embedding-v4）。"""

    MODEL = os.getenv("BAILIAN_EMBEDDING_MODEL", "text-embedding-v4")
    MAX_BATCH_SIZE = 10          # 单次请求文本条数上限
    MAX_INPUT_TOKENS = 8192      # 单条文本 token 上限
    SUPPORTED_DIMENSIONS = (2048, 1536, 1024, 768, 512, 256, 128, 64)
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        dimension: int | None = DEFAULT_DIMENSION,
        batch_size: int = MAX_BATCH_SIZE,
        *,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 8.0,
        request_interval: float = 0.0,
    ):
        """
        param:
            api_key: 百炼 API Key，缺省读取环境变量 DASHSCOPE_API_KEY
            model: 向量模型名，默认 text-embedding-v4
            dimension: 输出维度，None 表示用模型默认（1024）
            batch_size: 单次请求文本条数，不能超过 10
            max_retries: 单批最大尝试次数（含首次）
            retry_backoff: 退避基数，第 n 次重试等待 retry_backoff * 2^(n-1) 秒
            max_retry_delay: 单次退避等待上限（秒）
            request_interval: 两次请求之间的固定间隔（秒），用于主动限流
        """
        self.api_key = api_key or os.getenv("BAILIAN_EMBEDDING_API_KEY")
        if not self.api_key:
            raise ValueError("缺少百炼 API Key：请传入 api_key 或设置环境变量 DASHSCOPE_API_KEY")

        if dimension is not None and dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"dimension 必须是 {self.SUPPORTED_DIMENSIONS} 之一，或 None，当前为 {dimension}"
            )
        if not 1 <= batch_size <= self.MAX_BATCH_SIZE:
            raise ValueError(f"batch_size 必须在 1~{self.MAX_BATCH_SIZE} 之间，当前为 {batch_size}")
        if max_retries < 1:
            raise ValueError("max_retries 至少为 1")

        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_retry_delay = max_retry_delay
        self.request_interval = request_interval

    # ---------------------------------------------------------------- 对外接口
    def embed_documents(self, texts: list[str], *, text_type: str | None = "document") -> list[list[float]]:
        """批量向量化文档片段，返回顺序与入参严格一致。

        param:
            texts: 待向量化文本列表
            text_type: document（默认，偏正文表达）/ query（偏标题式表达）/ None（不做区分）
        """
        if isinstance(texts, str):
            raise TypeError("embed_documents 接收 list[str]，单条文本请用 embed_query")
        texts = list(texts)
        if not texts:
            return []

        self._validate_texts(texts)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch, text_type=text_type))
            logger.debug("已向量化 %d/%d 条文本", len(vectors), len(texts))
        return vectors

    def embed_query(self, text: str, *, instruct: str | None = None) -> list[float]:
        """向量化查询文本。

        param:
            text: 查询文本（固定按 text_type="query" 向量化）
            instruct: 英文任务指令，用于优化检索场景的向量质量
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("查询文本不能为空")

        return self._embed_batch([text], text_type="query", instruct=instruct)[0]

    # ---------------------------------------------------------------- 内部实现
    def _embed_batch(self, texts: list[str], *, text_type: str | None = None,
                     instruct: str | None = None) -> list[list[float]]:
        """下单批请求，带指数退避重试。"""
        payload = {
            "model": self.model,
            "input": texts,
            "api_key": self.api_key,
        }
        if self.dimension is not None:
            payload["dimension"] = self.dimension
        if text_type:
            payload["text_type"] = text_type
        if instruct:
            payload["instruct"] = instruct

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = dashscope.TextEmbedding.call(**payload)
            except Exception as exc:                     # 网络异常等，可重试
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"百炼 embedding 请求异常（已尝试 {self.max_retries} 次）: {exc}"
                    ) from exc
                self._backoff(attempt, f"请求异常: {exc}")
                continue

            if resp.status_code == HTTPStatus.OK:
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
                return self._parse(resp, expected=len(texts))

            detail = (f"status={resp.status_code}, code={resp.code}, "
                      f"message={resp.message}, request_id={resp.request_id}")
            if resp.status_code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                raise RuntimeError(f"百炼 embedding 调用失败: {detail}")
            self._backoff(attempt, detail)

        raise RuntimeError("百炼 embedding 调用失败：重试次数已耗尽")

    @staticmethod
    def _parse(resp, expected: int) -> list[list[float]]:
        """按 text_index 还原向量顺序，保证与输入一一对应。"""
        output = resp.output or {}
        items = output.get("embeddings") or []
        if len(items) != expected:
            raise RuntimeError(f"返回向量条数与请求不一致：期望 {expected}，实际 {len(items)}")

        vectors: list[list[float] | None] = [None] * expected
        for fallback, item in enumerate(items):
            index = item.get("text_index", fallback)
            if not 0 <= index < expected:
                raise RuntimeError(f"返回的 text_index 越界: {index}")
            vectors[index] = item.get("embedding")

        if any(v is None for v in vectors):
            raise RuntimeError("百炼 embedding 返回缺少部分文本的向量")
        return vectors                        # type: ignore[return-value]

    def _backoff(self, attempt: int, reason: str) -> None:
        """指数退避：1s、2s、4s……并封顶。"""
        delay = min(self.retry_backoff * (2 ** (attempt - 1)), self.max_retry_delay)
        logger.warning("百炼 embedding 第 %d 次重试，%.1fs 后继续：%s", attempt, delay, reason)
        time.sleep(delay)

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"第 {i} 条文本不是 str 类型: {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"第 {i} 条文本为空")
