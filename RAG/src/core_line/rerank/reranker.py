"""
交叉编码精排（rerank）：对多路召回 + RRF 融合后的候选做二次精准排序。

数据流：
    HybridRetriever.search() 的融合候选
        -> 校验并抽取 (query, text) 成对输入
        -> 交叉编码器逐对打相关性分（百炼 qwen3-rerank，可插拔替换）
        -> 按精排分降序重排
        -> 输出「原始排序 / 精排排序 / 两种分数」对比结果，供生成层取 Top-N

设计要点：
    * 模块化：CrossEncoderReranker 只负责编排（校验、打分调度、对比输出、排序），
      BaiLianRerankScorer 只负责打分（分批、重试、顺序还原）；打分器可注入替换，
      自定义打分器只需提供 score(query, texts) -> list[float]
    * 成对打分：每条候选的 text 与查询组成 (query, text) 对送入交叉编码器；
      交叉编码器联合编码查询与文本，相关性判断精度高于双塔召回
    * 批量可配：batch_size 控制单次请求文档数（qwen3-rerank 单次上限 500 条），
      模型与批大小均可通过构造参数或环境变量（BAILIAN_RERANK_MODEL）切换
    * 异常兜底：空候选返回空结果不打分；非法入参显式抛错；API 限流/5xx 指数退避
      重试后仍失败抛 RuntimeError；打分条数与候选数不一致直接报错，不静默对齐

输出结构（原始 vs 精排对比）：
    {
        "query":   str,      # 查询文本
        "model":   str,      # 打分模型名（自定义打分器为其 model 属性或类名）
        "count":   int,      # 参与精排的候选总数（top_k 截断前）
        "results": [         # 按 rerank_score 降序（同分按 original_rank 升序）
            {
                "id":             str,           # 候选唯一键（"{doc_id}::{chunk_id}"）
                "text":           str,           # chunk 正文
                "metadata":       dict,          # chunk 元数据（doc_id/chunk_id/heading_path 等）
                "original_rank":  int,           # 原始排序（融合后输入顺序，从 1 开始）
                "rerank_rank":    int,           # 精排后排序（从 1 开始）
                "original_score": float | None,  # 原始分（RRF 融合分，输入无 score 时为 None）
                "rerank_score":   float,         # 精排分（交叉编码相关性分）
                "channels":       list,          # 输入携带时原样透传（融合层命中来源）
            },
        ]
    }

用法：
    from core_line.retrieve.retriever import HybridRetriever, RecallChannel
    from core_line.rerank.reranker import CrossEncoderReranker

    hybrid = HybridRetriever([RecallChannel("vector", store), RecallChannel("fts5", fts5)])
    reranker = CrossEncoderReranker(model="qwen3-rerank", batch_size=50)

    fused = hybrid.search("如何配置 HNSW 参数", top_k=20)      # 召回 + RRF 融合候选
    result = reranker.rerank("如何配置 HNSW 参数", fused, top_k=5)
    for item in result["results"]:
        print(item["rerank_rank"], item["original_rank"], item["rerank_score"], item["id"])

注意：
    * 默认打分器走百炼 TextReRank，需要 API Key（构造传 api_key，或按统一读取链
      BAILIAN_RERANK_API_KEY -> BAILIAN_EMBEDDING_API_KEY -> DASHSCOPE_API_KEY 解析，
      见 core_line/config.py）；
    * qwen3-rerank 限制：单条文档 4,000 token、单次请求 500 条 / 120,000 token；
      超长文本服务端直接返回 400（不会截断），请保证 chunk 在上限内；
      gte-rerank 已于 2026-05-30 下线，模型名请用 qwen3-rerank / qwen3.7-text-rerank /
      gte-rerank-v2；
    * rerank_score（相关性分）与 original_score（RRF 融合分）量纲不同，禁止互相比较
      或相加；对比只看 original_rank 与 rerank_rank 的位次变化；
    * 本层不去重：同一 id 多次出现会各自成条（跨通道去重是融合层职责）；
    * top_k 截断发生在打分与排序之后，全体候选都会被打分。
"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import Any, Mapping, Protocol, Sequence

try:
    import dashscope
except ImportError:  # 允许未安装 SDK 时仅注入自定义打分器
    dashscope = None

try:  # 以 RAG/src 为导入根（与 RagApp.py 保持一致）
    from core_line.config import bailian_key_hint, find_bailian_api_key
    from core_line.config import get_bailian_base_url, get_rerank_model
    from core_line.cache import SemanticCache, get_default_cache
    from core_line.cache import make_key, normalize_query, text_fingerprint
except ImportError:  # 作为包的一部分导入时的兜底
    from ..config import bailian_key_hint, find_bailian_api_key
    from ..config import get_bailian_base_url, get_rerank_model
    from ..cache import SemanticCache, get_default_cache
    from ..cache import make_key, normalize_query, text_fingerprint

logger = logging.getLogger(__name__)

# 可重试状态码：限流 + 服务端错误
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RerankScorer(Protocol):
    """打分器协议：对 (query, texts) 成对打相关性分，返回与 texts 等长的浮点列表。"""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class BaiLianRerankScorer:
    """百炼（DashScope）交叉编码重排打分器（默认 qwen3-rerank）。"""

    MODEL = get_rerank_model()   # BAILIAN_RERANK_MODEL，缺省 qwen3-rerank（集中读取见 core_line/config.py）
    DEFAULT_BATCH_SIZE = 50   # 默认单次请求文档条数
    MAX_BATCH_SIZE = 500      # 单次请求文档条数上限（qwen3 系列）
    MAX_DOC_TOKENS = 4000     # 单条文档 token 上限，超限服务端返回 400（不截断）

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 8.0,
        request_interval: float = 0.0,
        base_url: str | None = None,
        cache: SemanticCache | None = None,
    ):
        """
        param:
            api_key: 百炼 API Key，缺省按统一读取链解析（见 core_line/config.py）：
                BAILIAN_RERANK_API_KEY -> BAILIAN_EMBEDDING_API_KEY -> DASHSCOPE_API_KEY
            model: 重排模型名，缺省读 BAILIAN_RERANK_MODEL（默认 qwen3-rerank，gte-rerank 已下线勿用）
            batch_size: 单次请求文档条数，1~500
            max_retries: 单批最大尝试次数（含首次）
            retry_backoff: 退避基数，第 n 次重试等待 retry_backoff * 2^(n-1) 秒
            max_retry_delay: 单次退避等待上限（秒）
            request_interval: 两次请求之间的固定间隔（秒），用于主动限流
            base_url: 百炼 API 基址覆盖，缺省读 BAILIAN_BASE_URL（兼容模式地址
                自动转换为原生 /api/v1 基址），未配置则走 SDK 公网默认端点
            cache: 语义缓存实例（(query, 文本) 打分结果缓存）；None 走默认缓存，
                禁用可传 SemanticCache(enabled=False)
        """
        if dashscope is None:
            raise ImportError("缺少 dashscope SDK：请安装 dashscope，或为 CrossEncoderReranker 注入自定义 scorer")
        self.api_key = api_key or find_bailian_api_key()
        if not self.api_key:
            raise ValueError(bailian_key_hint())
        self.base_url = base_url or get_bailian_base_url()
        self._cache = get_default_cache() if cache is None else cache
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model 不能为空")
        if not 1 <= batch_size <= self.MAX_BATCH_SIZE:
            raise ValueError(f"batch_size 必须在 1~{self.MAX_BATCH_SIZE} 之间，当前为 {batch_size}")
        if max_retries < 1:
            raise ValueError("max_retries 至少为 1")

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_retry_delay = max_retry_delay
        self.request_interval = request_interval

    # ---------------------------------------------------------------- 对外接口
    def score(self, query: str, texts: list[str]) -> list[float]:
        """对 (query, texts) 成对打相关性分，返回与 texts 等长、顺序一致的浮点列表。

        param:
            query: 查询文本
            texts: 候选文本列表（单条文本请包成 [text]）
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询文本不能为空")
        if isinstance(texts, str):
            raise TypeError("score 接收 list[str]，单条文本请包成 [text]")
        texts = list(texts)
        if not texts:
            return []

        self._validate_texts(texts)

        q_key = normalize_query(query)
        keys = [make_key("rerank", self.model, q_key, text_fingerprint(t)) for t in texts]
        cached: dict[str, float] = {}
        if self._cache:
            cached = self._cache.get_many("rerank", keys)

        scores: list[float | None] = [cached.get(k) for k in keys]
        missing_pos = [i for i, k in enumerate(keys) if k not in cached]

        if missing_pos:
            missing_texts = [texts[i] for i in missing_pos]
            fresh: list[float] = []
            for start in range(0, len(missing_texts), self.batch_size):
                batch = missing_texts[start:start + self.batch_size]
                fresh.extend(self._score_batch(query, batch))
                logger.debug("已精排打分 %d/%d 条候选", len(fresh), len(missing_texts))
            if self._cache:
                self._cache.set_many(
                    "rerank", {keys[i]: s for i, s in zip(missing_pos, fresh)}
                )
            for i, s in zip(missing_pos, fresh):
                scores[i] = s

        logger.debug("精排打分完成：%d 条（缓存命中 %d，API %d）",
                     len(texts), len(texts) - len(missing_pos), len(missing_pos))
        return scores                        # type: ignore[return-value]

    # ---------------------------------------------------------------- 内部实现
    def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        """单批请求，带指数退避重试。"""
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "api_key": self.api_key,
            "return_documents": False,   # 文本由调用方持有，减少返回负载
        }
        if self.base_url:
            payload["base_address"] = self.base_url   # dashscope 按次覆盖 API 基址（原生协议）

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = dashscope.TextReRank.call(**payload)
            except Exception as exc:                     # 网络异常等，可重试
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"百炼 rerank 请求异常（已尝试 {self.max_retries} 次）: {exc}"
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
                raise RuntimeError(f"百炼 rerank 调用失败: {detail}")
            self._backoff(attempt, detail)

        raise RuntimeError("百炼 rerank 调用失败：重试次数已耗尽")

    @staticmethod
    def _parse(resp: Any, expected: int) -> list[float]:
        """按 index 还原打分顺序，保证与输入文档一一对应。"""
        output = resp.output or {}
        items = _field(output, "results") or []
        if len(items) != expected:
            raise RuntimeError(f"返回打分条数与请求不一致：期望 {expected}，实际 {len(items)}")

        scores: list[float | None] = [None] * expected
        for fallback, item in enumerate(items):
            index = _field(item, "index")
            index = fallback if index is None else index
            if not 0 <= index < expected:
                raise RuntimeError(f"返回的 index 越界: {index}")
            value = _field(item, "relevance_score")
            if value is None:
                raise RuntimeError(f"第 {index} 条缺少 relevance_score")
            scores[index] = float(value)

        if any(score is None for score in scores):
            raise RuntimeError("百炼 rerank 返回缺少部分文档的打分")
        return scores                        # type: ignore[return-value]

    def _backoff(self, attempt: int, reason: str) -> None:
        """指数退避：1s、2s、4s……并封顶。"""
        delay = min(self.retry_backoff * (2 ** (attempt - 1)), self.max_retry_delay)
        logger.warning("百炼 rerank 第 %d 次重试，%.1fs 后继续：%s", attempt, delay, reason)
        time.sleep(delay)

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"第 {i} 条文本不是 str 类型: {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"第 {i} 条文本为空")


class CrossEncoderReranker:
    """交叉编码精排器：编排候选校验、成对打分、对比输出；打分器可插拔。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = BaiLianRerankScorer.MODEL,
        batch_size: int = BaiLianRerankScorer.DEFAULT_BATCH_SIZE,
        scorer: RerankScorer | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 8.0,
        request_interval: float = 0.0,
    ):
        """
        param:
            api_key: 百炼 API Key（仅默认打分器使用）
            model: 重排模型名，默认 qwen3-rerank（仅默认打分器使用）
            batch_size: 单次请求文档条数（仅默认打分器使用）
            scorer: 自定义打分器，需提供 score(query, texts) -> list[float]；
                提供后 api_key/model/batch_size 等参数被忽略，打分与分批由其自行负责
            max_retries / retry_backoff / max_retry_delay / request_interval:
                默认打分器的重试与限流参数
        """
        if scorer is not None:
            if not callable(getattr(scorer, "score", None)):
                raise TypeError("scorer 需提供 score(query, texts) 方法")
            self._scorer = scorer
            self.model = str(getattr(scorer, "model", type(scorer).__name__))
        else:
            self._scorer = BaiLianRerankScorer(
                api_key=api_key,
                model=model,
                batch_size=batch_size,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                max_retry_delay=max_retry_delay,
                request_interval=request_interval,
            )
            self.model = self._scorer.model

    # ---------------------------------------------------------------- 精排
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        *,
        top_k: int | None = None,
        fallback_on_error: bool = True,
    ) -> dict[str, Any]:
        """对融合候选做交叉编码精排，返回原始/精排对比结果。

        param:
            query: 查询文本（与每条候选的 text 组成成对输入）
            candidates: 多路召回 + RRF 融合后的候选列表，每条至少含 text，
                建议含 id / metadata / score（即 HybridRetriever.search 的输出）
            top_k: 精排后保留条数上限，None 表示全部返回；截断发生在打分与排序之后
            fallback_on_error: 打分器失败/超时时降级为 RRF 排序直接进入生成
                （保持原顺序、rerank_score 回填 original_score），而非全链路报错

        return:
            {"query", "model", "count", "results", "degraded", "error"}；results 按
            rerank_score 降序（同分按 original_rank 升序），每条含 original_rank /
            rerank_rank / original_score / rerank_score 对比字段，结构见模块文档

        说明：
            空候选返回 count=0 的空结果且不调用打分器；本层不去重（融合层职责）。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询文本不能为空")
        if isinstance(candidates, (str, bytes, Mapping)):
            raise TypeError("candidates 需为候选 dict 列表，单条候选请包成 [candidate]")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k 至少为 1，当前为 {top_k}")

        records = self._normalize(list(candidates))
        if not records:
            logger.debug("精排候选为空，返回空结果")
            return {"query": query, "model": self.model, "count": 0,
                    "results": [], "degraded": False, "error": None}

        try:
            raw_scores = list(self._scorer.score(query, [record["text"] for record in records]))
            if len(raw_scores) != len(records):
                raise RuntimeError(
                    f"打分器返回 {len(raw_scores)} 条分数，与候选数 {len(records)} 不一致"
                )
            for index, value in enumerate(raw_scores):
                try:
                    records[index]["rerank_score"] = float(value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"打分器第 {index} 条分数不是数值: {value!r}") from exc
        except Exception as exc:
            if not fallback_on_error:
                raise
            return self._degraded_result(query, records, exc, top_k)

        records.sort(key=lambda record: (-record["rerank_score"], record["original_rank"]))
        for new_rank, record in enumerate(records, start=1):
            record["rerank_rank"] = new_rank

        total = len(records)
        if top_k is not None:
            records = records[:top_k]
        logger.debug("精排完成：%d 条候选，返回 %d 条", total, len(records))
        return {"query": query, "model": self.model, "count": total, "results": records,
                "degraded": False, "error": None}

    @staticmethod
    def _degraded_result(
        query: str,
        records: list[dict[str, Any]],
        exc: Exception,
        top_k: int | None,
    ) -> dict[str, Any]:
        """降级兜底：打分失败时保持 RRF 排序直接进入生成。"""
        logger.warning("rerank 打分失败，降级为 RRF 排序直接进入生成: %s", exc)
        for record in records:
            original = record.get("original_score")
            record["rerank_score"] = float(original) if isinstance(original, (int, float)) else 0.0
            record["rerank_rank"] = record["original_rank"]
        records.sort(key=lambda record: record["rerank_rank"])
        total = len(records)
        if top_k is not None:
            records = records[:top_k]
        return {
            "query": query,
            "model": "degraded:rrf-order",
            "count": total,
            "results": records,
            "degraded": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # ---------------------------------------------------------------- 内部实现
    @staticmethod
    def _normalize(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """校验并归一化候选，保留输入顺序与原始分，不做去重。"""
        records: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise TypeError(f"第 {index} 条候选不是 dict: {type(candidate).__name__}")
            text = candidate.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {index} 条候选的 text 为空")

            record_id = candidate.get("id")
            record: dict[str, Any] = {
                "id": str(record_id) if record_id is not None else str(index),
                "text": text,
                "metadata": dict(candidate.get("metadata") or {}),
                "original_rank": index + 1,
                "rerank_rank": 0,
                "original_score": candidate.get("score"),
                "rerank_score": 0.0,
            }
            if "channels" in candidate:
                record["channels"] = candidate["channels"]
            records.append(record)
        return records


# ---------------------------------------------------------------- 工具函数
def _field(obj: Any, key: str) -> Any:
    """兼容 dict 与对象两种返回形态读取字段。"""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)
