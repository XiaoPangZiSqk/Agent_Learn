"""
离线端构建：加载文档 → 结构化切分 → 记录标准化 → 双通道落库。

通道：
    向量  : ChromaVector.add_records   -> data/chroma_data/      （百炼 embedding）
    关键词: FTS5Retriever.build        -> data/fts5_index/fts5.db（SQLite FTS5）

两路通道消费同一批 to_records() 记录，id 规则统一为 {doc_id}::{chunk_id}，
与在线端（online_query.py）的双路召回按 id 对齐。

用法（在 RAG/src 目录下执行）：
    python offline_build.py                     # 构建 data/load_file/ 下全部文档
    python offline_build.py --source 目录/文件   # 指定源
    python offline_build.py --rebuild           # 清空两路索引后全量重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from core_line.chunk.markdown_chunker import MarkdownChunker, to_records
from core_line.load_file.load_data import LoadData
from core_line.retrieve.fts5_retriever import FTS5Retriever
from core_line.utils.config import ConfigError, preflight
from core_line.vector_store.chromadb_store import ChromaVector

# 默认源文档目录：RAG/data/load_file/
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "data" / "load_file"


def build_index(source: str | Path | None = None, *, rebuild: bool = False) -> dict[str, Any]:
    """执行一次离线构建，返回统计信息。

    param:
        source:  文档目录或单个文件，缺省 RAG/data/load_file/
        rebuild: True 表示清空两路索引后全量重建（向量集合 reset + FTS5 清库重建）

    return:
        {"source", "docs", "records", "vector_ids", "vector_total", "fts5"}

    说明：
        增量构建（rebuild=False）按 doc_id 先删后写，文档内容/切分参数变化后
        不残留幽灵块；FTS5 侧还会按内容指纹跳过未变更文档的分词。
    """
    source_path = Path(source) if source is not None else DEFAULT_SOURCE
    if not source_path.exists():
        raise FileNotFoundError(f"源路径不存在: {source_path}")

    # 构建只需百炼 embedding，不需要 DeepSeek 生成配置
    config = preflight(need_bailian=True, need_generation=False)
    print(f"[配置] 预检通过 | {config.summary()}")

    loader = LoadData()
    chunker = MarkdownChunker()
    store = ChromaVector()
    fts5 = FTS5Retriever()

    records: list[dict[str, Any]] = []
    doc_ids: list[str] = []
    for document in loader.load_file(source_path):
        doc_id = document.metadata["file_name"]
        doc_records = to_records(chunker.chunk(document.page_content), doc_id)
        records.extend(doc_records)
        doc_ids.append(doc_id)
        print(f"[切分] {doc_id}: {len(doc_records)} 块")

    if not records:
        raise RuntimeError(f"未从 {source_path} 加载到任何可索引文档")

    # 向量通道：先按 doc_id 清掉旧块再写入，chunk 数变少时不留幽灵块
    if rebuild:
        store.reset()
    else:
        for doc_id in sorted(set(doc_ids)):
            store.delete(where={"doc_id": doc_id})
    vector_ids = store.add_records(records)

    # 关键词通道：SQLite FTS5 单文件倒排索引
    try:
        fts5_stats = fts5.build(records, rebuild=rebuild)
    finally:
        fts5.close()

    stats: dict[str, Any] = {
        "source": str(source_path),
        "docs": len(set(doc_ids)),
        "records": len(records),
        "vector_ids": len(vector_ids),
        "vector_total": store.count(),
        "fts5": fts5_stats,
    }
    print(
        f"[构建完成] 文档 {stats['docs']} 篇 | 记录 {stats['records']} 条 | "
        f"向量库共 {stats['vector_total']} 条 | FTS5 {fts5_stats}"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG 离线端构建：加载 → 切分 → 双通道落库（向量 Chroma + 关键词 FTS5）",
    )
    parser.add_argument(
        "--source", default=str(DEFAULT_SOURCE),
        help="源文档目录或单个文件（默认: %(default)s）",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="清空两路索引后全量重建（默认增量：按 doc_id 先删后写）",
    )
    args = parser.parse_args(argv)

    try:
        build_index(args.source, rebuild=args.rebuild)
    except (ConfigError, FileNotFoundError, RuntimeError) as exc:
        print(f"[构建失败] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
