"""
在线端查询：双路召回（向量 + FTS5 关键词）→ RRF 融合 → 交叉重排 → 上下文组装 → 生成。

依赖离线端（offline_build.py）产出的两路索引：
    向量  : data/chroma_data/       （ChromaVector）
    关键词: data/fts5_index/fts5.db（FTS5Retriever）

用法（在 RAG/src 目录下执行）：
    python online_query.py "K-means 的原理是什么？"   # 单次查询
    python online_query.py                            # 交互式循环，q 退出
"""

from __future__ import annotations

import argparse
import sys

from core_line.generate.context_assembler import ContextAssembler
from core_line.generate.generator import Generator
from core_line.pipeline import PipelineResult, RagPipeline
from core_line.rerank.reranker import CrossEncoderReranker
from core_line.retrieve.fts5_retriever import FTS5Retriever
from core_line.retrieve.retriever import HybridRetriever, RecallChannel
from core_line.utils.config import ConfigError, preflight
from core_line.vector_store.chromadb_store import ChromaVector


def create_pipeline() -> RagPipeline:
    """装载两路索引并组装查询流水线。

    raise:
        RuntimeError: FTS5 关键词索引不存在（需先执行离线构建）
    """
    config = preflight()   # 向量/重排（百炼）+ 生成（DeepSeek）配置齐检
    print(f"[配置] 预检通过 | {config.summary()}")

    fts5 = FTS5Retriever()
    if not fts5.load():
        raise RuntimeError(
            "FTS5 关键词索引不存在，请先执行离线构建: python offline_build.py"
        )

    store = ChromaVector()   # 只读操作不需 Key，查询时懒加载 embedding
    if store.count() == 0:
        print("[警告] 向量集合为空，仅关键词一路参与召回（请先执行离线构建）")

    hybrid = HybridRetriever([
        RecallChannel("vector", store),
        RecallChannel("keyword", fts5),
    ])
    return RagPipeline(
        retriever=hybrid,
        reranker=CrossEncoderReranker(),
        assembler=ContextAssembler(),
        generator=Generator(),
    )


def run_query(pipeline: RagPipeline, query: str) -> PipelineResult:
    """执行单次查询并打印结果摘要，返回完整 PipelineResult。"""
    result = pipeline.run(query)

    print(f"\n[问题] {query}")
    if result.result is not None:
        print(f"[答案] {result.result.answer}")
        verdict = "是" if result.result.evidence_sufficient else "否"
        print(f"[置信度] {result.result.confidence:.2f} | 证据充分: {verdict}")
        if result.result.degraded:
            print(f"[降级] {result.result.error}")
        if result.result.nli_issues:
            print(f"[NLI] {'; '.join(result.result.nli_issues)}")

    print(f"[召回] {len(result.fused)} 条融合候选")
    for item in result.context.chunks if result.context else []:
        print(f"  [{item.index}] {item.id} ({item.char_count} chars)")

    elapsed = " | ".join(f"{k}={v:.0f}ms" for k, v in result.timings_ms.items())
    print(f"[耗时] {elapsed}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG 在线端查询：双路召回 → RRF 融合 → 交叉重排 → 生成",
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="查询文本；缺省进入交互式循环",
    )
    args = parser.parse_args(argv)

    try:
        pipeline = create_pipeline()
    except (ConfigError, RuntimeError) as exc:
        print(f"[启动失败] {exc}")
        return 1

    if args.query is not None:
        run_query(pipeline, args.query)
        return 0

    # 交互式循环
    print("[交互] 输入问题后回车查询，输入 q 退出")
    while True:
        try:
            query = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue
        try:
            run_query(pipeline, query)
        except Exception as exc:   # 单次查询失败不退出循环
            print(f"[查询失败] {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
