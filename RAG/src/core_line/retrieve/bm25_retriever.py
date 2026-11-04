"""
BM25 关键词召回（方案 B：本地分词缓存）

数据流：
    构建期：to_records() 的 records -> tokenize -> docs.pkl / manifest.json / meta.json
    查询期：load() 只读本地缓存 -> 重建 BM25 统计量 -> search()

设计要点：
    * 不从 Chroma 拉语料：BM25 与向量库各自消费同一批 records，互不依赖
    * 缓存只保存分词结果与文档身份信息，BM25 统计量在加载时重建
      （省掉最慢的 jieba 分词，又不依赖 rank_bm25 的内部字段）
    * 同一文档重新构建时先按 doc_id 删除旧 chunk，避免 chunk_id 重排留下幽灵记录
    * 落盘顺序 docs -> manifest -> meta，meta.json 作为提交标记，避免加载半份索引
    * 分词规则版本（TOKENIZER_VERSION）不一致时拒绝加载，强制重建

用法：
    retriever = BM25Retriever("data/bm25_index")
    retriever.build(records)                 # 构建期：与 store.add_records 并列调用
    retriever.load()                         # 查询期：只读缓存
    hits = retriever.search("配置 HNSW 参数", top_k=10)

注意：
    docs.pkl 使用 pickle 序列化，只能加载本项目自己生成的缓存文件。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # 以 RAG/src 为导入根（与 RagApp.py 保持一致）
    from core_line.embedding.tokenizer import TOKENIZER_VERSION, tokenize
except ImportError:  # 作为包的一部分导入时的兜底
    from ..embedding.tokenizer import TOKENIZER_VERSION, tokenize

logger = logging.getLogger(__name__)

# RAG 项目根目录：.../RAG/src/core_line/retrieve/bm25_retriever.py -> .../RAG
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class BM25Retriever:
    """基于本地分词缓存的 BM25 关键词召回器。"""

    SCHEMA_VERSION = 1
    DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "bm25_index"
    DEFAULT_TOP_K = 20
    DOCS_FILE = "docs.pkl"
    MANIFEST_FILE = "manifest.json"
    META_FILE = "meta.json"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        param:
            cache_dir: 缓存目录，缺省为 RAG/data/bm25_index
            k1: 词频饱和参数，越大越强调词频（常用 1.2~2.0）
            b: 文档长度归一化强度，0 表示不归一化，1 表示完全归一化
        """
        if k1 <= 0:
            raise ValueError(f"k1 必须为正数，当前为 {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b 必须在 0~1 之间，当前为 {b}")

        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR).expanduser().resolve()
        self.k1 = k1
        self.b = b

        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._metas: list[dict[str, Any]] = []
        self._texts: list[str] = []
        self._manifest: dict[str, dict[str, Any]] = {}
        self._index: _BM25Index | None = None
        self._loaded = False

    # ---------------------------------------------------------------- 对外属性
    @property
    def docs_path(self) -> Path:
        return self.cache_dir / self.DOCS_FILE

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / self.MANIFEST_FILE

    @property
    def meta_path(self) -> Path:
        return self.cache_dir / self.META_FILE

    @property
    def is_ready(self) -> bool:
        """索引是否已构建或已加载。"""
        return self._index is not None

    @property
    def doc_ids(self) -> list[str]:
        """已建索引的文档 id 列表（来自 manifest）。"""
        return sorted(self._manifest)

    def __len__(self) -> int:
        return len(self._ids)

    # ---------------------------------------------------------------- 构建
    def build(self, records: Sequence[Mapping[str, Any]], *, rebuild: bool = False) -> dict[str, int]:
        """用 to_records() 的产物构建/更新关键词索引，并落盘。

        param:
            records: chunk 记录列表，需含 doc_id、chunk_id、text
            rebuild: True 表示清空现有缓存后全量重建

        return:
            {"records", "chunks", "docs", "tokens", "removed"} 统计信息

        说明：
            增量构建（rebuild=False）会先加载已有缓存，再按 doc_id 覆盖同文档的旧 chunk；
            同一批 records 内出现重复 id 直接报错，避免静默丢数据。
        """
        if isinstance(records, Mapping):
            raise TypeError("build 接收记录列表，单条记录请包成 [record]")
        records = list(records)

        if rebuild:
            self._reset_state()
        elif not self._loaded and self.meta_path.exists():
            self.load()

        grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        seen: set[str] = set()
        for i, record in enumerate(records):
            record_id, doc_id, text, metadata = self._parse_record(record, index=i)
            if record_id in seen:
                raise ValueError(f"第 {i} 条记录 id 重复: {record_id}")
            seen.add(record_id)
            grouped.setdefault(doc_id, []).append((record_id, text, metadata))

        removed = self._drop_docs(set(grouped))

        for doc_id, items in grouped.items():
            for record_id, text, metadata in items:
                self._ids.append(record_id)
                self._tokens.append(tokenize(text))
                self._metas.append(metadata)
                self._texts.append(text)
            self._manifest[doc_id] = {
                "chunks": len(items),
                "hash": _doc_hash(items),
                "updated_at": _now(),
            }

        self._rebuild_index()
        self._save()
        self._loaded = True

        stats = {
            "records": len(records),
            "chunks": len(self._ids),
            "docs": len(self._manifest),
            "tokens": sum(len(t) for t in self._tokens),
            "removed": removed,
        }
        logger.debug("BM25 构建完成：%s", stats)
        return stats

    # ---------------------------------------------------------------- 加载
    def load(self) -> bool:
        """从本地缓存加载索引。

        return:
            True 表示加载成功；False 表示缓存不存在（调用方可选择重建）

        raise:
            ValueError: 版本不匹配
            RuntimeError: 缓存文件损坏
        """
        if not self.meta_path.exists() or not self.docs_path.exists():
            return False

        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"BM25 缓存元数据损坏: {self.meta_path}") from exc

        if meta.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"BM25 缓存结构版本不匹配（期望 {self.SCHEMA_VERSION}，"
                f"实际 {meta.get('schema_version')}），请重建缓存"
            )
        if meta.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ValueError(
                f"分词器版本不匹配（期望 {TOKENIZER_VERSION}，"
                f"实际 {meta.get('tokenizer_version')}），请重建缓存"
            )

        try:
            with self.docs_path.open("rb") as fp:
                payload = pickle.load(fp)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise RuntimeError(f"BM25 缓存损坏，请重建: {self.docs_path}") from exc

        ids = list(payload.get("ids") or [])
        tokens = list(payload.get("tokens") or [])
        metas = list(payload.get("metas") or [])
        texts = list(payload.get("texts") or [])
        if not (len(ids) == len(tokens) == len(metas) == len(texts)):
            raise RuntimeError("BM25 缓存字段长度不一致，请重建")

        self._ids = ids
        self._tokens = tokens
        self._metas = metas
        self._texts = texts
        self._manifest = self._read_manifest()
        self._rebuild_index()
        self._loaded = True
        logger.debug("BM25 缓存已加载：%d 条 chunk，%d 篇文档", len(self._ids), len(self._manifest))
        return True

    # ---------------------------------------------------------------- 查询
    def search(self, query: str, *, top_k: int = DEFAULT_TOP_K,
               where: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """按关键词检索 chunk，返回按 BM25 分数降序的结果。

        param:
            query: 查询文本（内部使用与建库相同的分词规则）
            top_k: 返回条数上限
            where: 元数据过滤条件，支持 $eq/$ne/$in/$nin/$gt/$gte/$lt/$lte/$and/$or

        return:
            [{"id", "text", "metadata", "score"}]

        说明：
            BM25 分数没有固定上界，不要与向量 cosine 分数直接比较；
            混合召回请在外部使用 RRF 按排名融合。
        """
        if self._index is None:
            raise RuntimeError("BM25 索引未就绪：请先 build() 或 load()")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询文本不能为空")
        if top_k < 1:
            raise ValueError(f"top_k 至少为 1，当前为 {top_k}")
        if where is not None:
            _validate_where(where)   # 提前校验，避免无命中时静默放过非法条件

        query_tokens = _dedupe(tokenize(query))
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        candidates: list[tuple[float, str, int]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            if where and not _match_metadata(self._metas[idx], where):
                continue
            candidates.append((score, self._ids[idx], idx))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [{
            "id": self._ids[idx],
            "text": self._texts[idx],
            "metadata": self._metas[idx],
            "score": score,
        } for score, _, idx in candidates[:top_k]]

    # ---------------------------------------------------------------- 维护
    def clear(self) -> None:
        """删除缓存文件并清空内存状态（全量重建前可显式调用）。"""
        for path in (self.docs_path, self.manifest_path, self.meta_path):
            path.unlink(missing_ok=True)
        self._reset_state()
        logger.debug("BM25 缓存已清空：%s", self.cache_dir)

    # ---------------------------------------------------------------- 内部实现
    def _reset_state(self) -> None:
        self._ids = []
        self._tokens = []
        self._metas = []
        self._texts = []
        self._manifest = {}
        self._index = None
        self._loaded = False

    def _rebuild_index(self) -> None:
        """用已分词语料重建 BM25 统计量（tf / df / idf / avgdl）。"""
        self._index = _BM25Index(self._tokens, k1=self.k1, b=self.b) if self._tokens else None

    def _drop_docs(self, doc_ids: set[str]) -> int:
        """删除指定文档的旧 chunk，返回删除条数。"""
        if not doc_ids:
            return 0

        keep = [i for i, meta in enumerate(self._metas) if meta.get("doc_id") not in doc_ids]
        removed = len(self._ids) - len(keep)
        if removed:
            self._ids = [self._ids[i] for i in keep]
            self._tokens = [self._tokens[i] for i in keep]
            self._metas = [self._metas[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
        for doc_id in doc_ids:
            self._manifest.pop(doc_id, None)
        return removed

    def _save(self) -> None:
        """原子落盘：docs.pkl -> manifest.json -> meta.json（最后写作为提交标记）。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self._ids,
            "tokens": self._tokens,
            "metas": self._metas,
            "texts": self._texts,
        }

        tmp = self.docs_path.with_suffix(self.docs_path.suffix + ".tmp")
        with tmp.open("wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.docs_path)

        _write_json(self.manifest_path, self._manifest)

        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": self.k1,
            "b": self.b,
            "chunk_count": len(self._ids),
            "doc_count": len(self._manifest),
            "token_count": sum(len(t) for t in self._tokens),
            "built_at": _now(),
        }
        _write_json(self.meta_path, meta)

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("BM25 manifest 损坏，已忽略: %s", self.manifest_path)
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _parse_record(record: Mapping[str, Any], *, index: int) -> tuple[str, str, str, dict[str, Any]]:
        """校验并归一化一条 record -> (id, doc_id, text, metadata)。"""
        if not isinstance(record, Mapping):
            raise TypeError(f"第 {index} 条记录不是 dict: {type(record).__name__}")

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {index} 条记录的 text 为空")

        doc_id = record.get("doc_id")
        if doc_id is None:
            raise ValueError(f"第 {index} 条记录缺少 doc_id")

        record_id = record.get("id")
        if record_id:
            record_id = str(record_id)
        elif record.get("chunk_id") is not None:
            record_id = f"{doc_id}::{record['chunk_id']}"
        else:
            raise ValueError(f"第 {index} 条记录缺少 chunk_id，无法生成 id")

        metadata = {k: v for k, v in record.items() if k not in ("id", "text")}
        return record_id, str(doc_id), text, metadata


class _BM25Index:
    """纯内存 BM25 索引：倒排表 + IDF + 文档长度。

    打分公式（与 Lucene 的 BM25 一致）：
        idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
        score(D, Q) = Σ idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))
    """

    __slots__ = ("k1", "b", "_postings", "_idf", "_doc_len", "_avgdl", "_size")

    def __init__(self, corpus: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._size = len(corpus)
        self._doc_len = [len(doc) for doc in corpus]
        total = sum(self._doc_len)
        self._avgdl = (total / self._size) if self._size else 0.0

        postings: dict[str, list[tuple[int, int]]] = {}
        doc_freq: Counter[str] = Counter()
        for idx, doc in enumerate(corpus):
            if not doc:
                continue
            for term, freq in Counter(doc).items():
                postings.setdefault(term, []).append((idx, freq))
                doc_freq[term] += 1

        n = self._size
        self._postings = postings
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        """返回长度等于文档数的分数数组。"""
        scores = [0.0] * self._size
        if not self._size or self._avgdl <= 0:
            return scores

        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, freq in self._postings[term]:
                normalizer = self.k1 * (1 - self.b + self.b * self._doc_len[idx] / self._avgdl)
                scores[idx] += idf * freq * (self.k1 + 1) / (freq + normalizer)
        return scores


# ---------------------------------------------------------------- 工具函数
def _dedupe(tokens: Sequence[str]) -> list[str]:
    """query 词项去重（保持顺序），避免子词扩展造成重复加权。"""
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _doc_hash(items: Sequence[tuple[str, str, dict[str, Any]]]) -> str:
    """按 id 排序后对文本做指纹，用于发现"chunk 数不变但内容变了"。"""
    digest = hashlib.blake2b(digest_size=8)
    for record_id, text, _ in sorted(items, key=lambda item: item[0]):
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """JSON 原子写入，避免进程中断留下半份文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 元数据过滤
_ALLOWED_OPS = frozenset({"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"})


def _validate_where(where: Mapping[str, Any]) -> None:
    """检索前校验 where 结构，避免无命中时非法条件被静默跳过。"""
    if not isinstance(where, Mapping):
        raise TypeError(f"where 需要 dict，当前为 {type(where).__name__}")

    for key, condition in where.items():
        if key in ("$and", "$or"):
            if not isinstance(condition, Sequence) or isinstance(condition, (str, bytes)):
                raise ValueError(f"{key} 需要条件列表")
            for item in condition:
                _validate_where(item)
        elif isinstance(condition, Mapping):
            if not condition:
                raise ValueError(f"字段 {key} 的过滤条件不能为空")
            for operator, expected in condition.items():
                if operator not in _ALLOWED_OPS:
                    raise ValueError(f"不支持的过滤操作符: {operator}")
                if operator in ("$in", "$nin") and (
                        not isinstance(expected, Sequence) or isinstance(expected, (str, bytes))):
                    raise ValueError(f"{operator} 需要列表参数")


def _match_metadata(metadata: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    """按 Chroma 风格的 where 条件过滤单条元数据。"""
    for key, condition in where.items():
        if key == "$and":
            if not isinstance(condition, Sequence) or isinstance(condition, (str, bytes)):
                raise ValueError("$and 需要条件列表")
            if not all(_match_metadata(metadata, cond) for cond in condition):
                return False
        elif key == "$or":
            if not isinstance(condition, Sequence) or isinstance(condition, (str, bytes)):
                raise ValueError("$or 需要条件列表")
            if not any(_match_metadata(metadata, cond) for cond in condition):
                return False
        elif not _match_field(metadata.get(key), condition):
            return False
    return True


def _match_field(actual: Any, condition: Any) -> bool:
    """单字段匹配：支持直接比较与操作符字典。"""
    if not isinstance(condition, Mapping):
        return actual == condition

    for operator, expected in condition.items():
        if operator == "$eq":
            matched = actual == expected
        elif operator == "$ne":
            matched = actual != expected
        elif operator in ("$in", "$nin"):
            if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
                raise ValueError(f"{operator} 需要列表参数")
            matched = actual in expected
            if operator == "$nin":
                matched = not matched
        elif operator in ("$gt", "$gte", "$lt", "$lte"):
            matched = _compare(actual, expected, operator)
        else:
            raise ValueError(f"不支持的过滤操作符: {operator}")
        if not matched:
            return False
    return True


def _compare(actual: Any, expected: Any, operator: str) -> bool:
    """数值/字符串比较；类型不可比时视为不匹配。"""
    try:
        if operator == "$gt":
            return actual > expected
        if operator == "$gte":
            return actual >= expected
        if operator == "$lt":
            return actual < expected
        return actual <= expected
    except TypeError:
        return False
