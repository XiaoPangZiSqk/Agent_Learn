"""
查询流水线（pipeline）：召回 -> RRF 融合 -> 重排 -> 上下文组装 -> 生成 的编排与并发执行。

职责：
1. run()        单查询全链路编排，返回各阶段产物与耗时（timings_ms）；
2. run_stream() 单查询 SSE 流式贯通：每完成一个阶段即推送 progress 事件，
   生成阶段复用 Generator.generate_stream 的首包优先事件流；
3. run_batch()  批量查询并发执行：各查询整链跑在线程池上，天然形成阶段重叠
   ——查询 i 的生成/重排 与 查询 i+1 的召回 并行，吞吐显著提升；
   结果按输入顺序返回。

依赖组件（构造注入，均为项目内既有组件）：
    retriever:  HybridRetriever（提供 search(query, top_k=) -> 融合候选）
    reranker:   CrossEncoderReranker（提供 rerank(query, candidates, top_k=)，
                自带 RRF 降级兜底）
    assembler:  ContextAssembler（提供 assemble(query, rerank_results) -> AssembledContext）
    generator:  Generator（提供 generate / generate_stream，自带缓存与降级兜底）

用法：
    pipeline = RagPipeline(hybrid, reranker, assembler, generator)
    result = pipeline.run("K-means 的原理是什么？")
    for event in pipeline.run_stream("冲泡红茶的水温？"):
        print(event.to_sse())
    results = pipeline.run_batch(["问题1", "问题2", "问题3"], max_workers=3)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from core_line.generate.generator import GenerationResult, SSEEvent, SSEStreamer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """单查询流水线结果：各阶段产物 + 分阶段耗时。"""

    query: str
    fused: list[dict[str, Any]] = field(default_factory=list)
    rerank: dict[str, Any] = field(default_factory=dict)
    context: Any = None
    result: GenerationResult | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """任一环节进入降级模式即为 True（rerank 或生成）。"""
        if self.rerank.get("degraded"):
            return True
        return bool(self.result and self.result.degraded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "fused_count": len(self.fused),
            "rerank_degraded": bool(self.rerank.get("degraded")),
            "context_chunks": len(self.context.chunks) if self.context else 0,
            "result": self.result.to_dict() if self.result else None,
            "timings_ms": self.timings_ms,
        }


class RagPipeline:
    """查询流水线编排器。"""

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        assembler: Any,
        generator: Any,
        *,
        fused_top_k: int = 6,
        rerank_top_k: int = 5,
    ):
        """
        param:
            retriever:   多路召回 + RRF 融合器（HybridRetriever）
            reranker:    交叉编码精排器（CrossEncoderReranker，失败自动降级 RRF 顺序）
            assembler:   上下文组装器（ContextAssembler）
            generator:   生成器（Generator，失败自动降级为上下文摘要）
            fused_top_k: 融合层返回候选数
            rerank_top_k: 精排后保留候选数（进入生成的上下文条数上限）
        """
        self.retriever = retriever
        self.reranker = reranker
        self.assembler = assembler
        self.generator = generator
        self.fused_top_k = fused_top_k
        self.rerank_top_k = rerank_top_k

    # ---------------------------------------------------------------- 单查询（同步）
    def run(self, query: str, *, index_version: str | None = None) -> PipelineResult:
        """单查询全链路：召回+融合 -> 重排 -> 组装 -> 生成。

        return:
            PipelineResult（含 timings_ms 分阶段耗时；rerank/生成的降级标记见 degraded 属性）
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        fused = self.retriever.search(query, top_k=self.fused_top_k)
        timings["recall_fusion_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rerank = self.reranker.rerank(query, fused, top_k=self.rerank_top_k)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        if rerank.get("degraded"):
            logger.warning("流水线降级：rerank 未生效，沿用 RRF 排序（%s）", rerank.get("error"))

        t0 = time.perf_counter()
        context = self.assembler.assemble(query, rerank["results"])
        timings["assemble_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result = self.generator.generate(query, context, index_version=index_version)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        logger.info("流水线完成：%s（总耗时 %.0fms，degraded=%s）",
                    query[:20], sum(timings.values()), result.degraded)
        return PipelineResult(
            query=query, fused=fused, rerank=rerank,
            context=context, result=result, timings_ms=timings,
        )

    # ---------------------------------------------------------------- 单查询（SSE 流式贯通）
    def run_stream(
        self, query: str, *, index_version: str | None = None
    ) -> Iterator[SSEEvent]:
        """SSE 流式贯通：每完成一个阶段推送 progress 事件，最后推送生成事件流。

        事件序列：
            progress(recall_fusion) -> progress(rerank) -> progress(assemble)
            -> Generator.generate_stream 的 conclusion/reasoning/citations/verification/done
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")

        try:
            t0 = time.perf_counter()
            fused = self.retriever.search(query, top_k=self.fused_top_k)
            yield SSEEvent(event="progress", data={
                "stage": "recall_fusion", "count": len(fused),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            })

            t0 = time.perf_counter()
            rerank = self.reranker.rerank(query, fused, top_k=self.rerank_top_k)
            yield SSEEvent(event="progress", data={
                "stage": "rerank", "count": rerank.get("count", 0),
                "degraded": bool(rerank.get("degraded")),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            })

            t0 = time.perf_counter()
            context = self.assembler.assemble(query, rerank["results"])
            yield SSEEvent(event="progress", data={
                "stage": "assemble", "chunks": len(context.chunks),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
        except Exception as exc:
            logger.error("流水线检索阶段失败: %s", exc)
            yield from SSEStreamer.stream_error(str(exc), stage="retrieve")
            return

        yield from self.generator.generate_stream(query, context, index_version=index_version)

    # ---------------------------------------------------------------- 批量（阶段重叠）
    def run_batch(
        self,
        queries: Sequence[str],
        *,
        max_workers: int = 3,
        index_version: str | None = None,
    ) -> list[PipelineResult]:
        """批量查询并发执行，结果按输入顺序返回。

        各查询整链跑在线程池上：查询 i 的生成/重排 与 查询 i+1 的召回 并行，
        形成"重排与下一查询的召回并行"的阶段流水线重叠，整体耗时接近
        最慢单链而非各链之和。
        """
        queries = list(queries)
        if not queries:
            return []
        workers = max(1, min(int(max_workers), len(queries)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self.run, q, index_version=index_version) for q in queries
            ]
            return [f.result() for f in futures]
