"""
Chroma 向量库（本地持久化模式）

* 使用 chromadb.PersistentClient 把索引落盘到本地目录，重启后可直接复用
* 集合固定使用 cosine 距离，检索结果给出 distance 与 score(= 1 - distance)
* 写入接受 chunker.to_records() 产出的记录，id 规则 "{doc_id}::{chunk_id}"，
  重复写入走 upsert，因此重建索引不会产生副本

用法：
    from core_line.vector_store.chromadb_store import ChromaVector

    store = ChromaVector()                        # 默认 RAG/chroma_data + rag_chunks
    store.add_records(records)                    # records 来自 to_records()
    hits = store.query("什么是 RAG？", top_k=5)    # [{"id","text","metadata","distance","score"}]

注意：
    dimension 与 embedding 模型强绑定（text-embedding-v4 默认 1024 维）；
    换模型或换维度等价于重建索引，需要先 reset() 再重新写入。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

try:  # 以 RAG/src 为导入根（与 RagApp.py 保持一致）
    from core_line.embedding.bai_lian_embeder import BaiLianEmbeder
except ImportError:  # 作为包的一部分导入时的兜底
    from ..embedding.bai_lian_embeder import BaiLianEmbeder

logger = logging.getLogger(__name__)

# RAG 项目根目录：.../RAG/src/core_line/vector_store/chromadb_store.py -> .../RAG
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_persist_dir() -> Path:
    """解析 CHROMA_PERSIST_DIR：缺省 RAG/data/chroma_data；相对路径锚定 RAG 项目根。

    注意不能按 cwd 解析——从 RAG/src 运行时会落到 RAG/src/data 下。
    """
    raw = (os.getenv("CHROMA_PERSIST_DIR") or "").strip()
    if not raw:
        return _PROJECT_ROOT / "data" / "chroma_data"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


class ChromaVector:
    """Chroma 本地持久化向量库：负责 chunk 记录的写入、读取与相似度检索。"""

    DEFAULT_PERSIST_DIR = _resolve_persist_dir()
    DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "rag_chunks")
    DEFAULT_TOP_K = 5
    WRITE_BATCH_SIZE = 64        # 单次 upsert 条数（embedding 侧还会按模型上限二次分批）
    DISTANCE_SPACE = "cosine"    # 距离度量，score = 1 - distance
    METADATA_TYPES = (str, int, float, bool)

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embedder: BaiLianEmbeder | None = None,
        *,
        dimension: int | None = None,
    ):
        """
        param:
            persist_dir: 本地持久化目录，缺省读 CHROMA_PERSIST_DIR，再缺省为 RAG/chroma_data
            collection_name: 集合名（3~512 位，仅字母数字 ._-，且首尾不能是符号）
            embedder: 向量化器，缺省懒加载 BaiLianEmbeder（需要 DASHSCOPE_API_KEY）
            dimension: 向量维度，缺省取 embedder.dimension，再缺省 1024
        """
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name 不能为空")

        self.persist_dir = Path(persist_dir or self.DEFAULT_PERSIST_DIR).expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self._embedder = embedder
        self.dimension = dimension if dimension is not None else \
            getattr(embedder, "dimension", None) or BaiLianEmbeder.DEFAULT_DIMENSION
        if self.dimension <= 0:
            raise ValueError(f"dimension 必须为正整数，当前为 {self.dimension}")

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._open_collection()
        logger.debug("Chroma 集合就绪：%s（目录 %s）", self.collection_name, self.persist_dir)

    # ---------------------------------------------------------------- 对外属性
    @property
    def collection(self) -> Collection:
        """底层 chromadb Collection，便于调用本类未封装的能力。"""
        return self._collection

    def __len__(self) -> int:
        return self.count()

    # ---------------------------------------------------------------- 写入
    def add_records(self, records: Sequence[Mapping[str, Any]], *,
                    batch_size: int = WRITE_BATCH_SIZE) -> list[str]:
        """写入 由markdown_chunker.py中 to_records() 产出的 chunk 记录，返回写入的 id 列表。

        param:
            records: 记录列表，每条至少包含 text，建议含 doc_id/chunk_id
            batch_size: 单批写入条数

        说明：
            同 id 写入会覆盖旧记录（upsert），重复构建索引不会产生副本。
            to_records()返回的dict列表中没有 id 键，所以会执行 _make_id():
                由 doc_id 与 chunk_id 生成稳定 id，保证重复写入可覆盖。

        """
        if isinstance(records, Mapping):
            raise TypeError("add_records 接收记录列表，单条记录请包成 [record]")

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for i, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"第 {i} 条记录不是 dict: {type(record).__name__}")

            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {i} 条记录的 text 为空")

            record_id = str(record.get("id") or self._make_id(record, index=i))
            metadata = self._clean_metadata(
                {k: v for k, v in record.items() if k not in ("id", "text")}
            )
            ids.append(record_id)
            texts.append(text)
            metadatas.append(metadata or {"id": record_id})   # Chroma 不接受空 metadata

        if not texts:
            return []
        return self._upsert(ids, texts, metadatas, batch_size=batch_size)

    def add_texts(self, texts: Sequence[str], *,
                  metadatas: Sequence[Mapping[str, Any]] | None = None,
                  ids: Sequence[str] | None = None,
                  batch_size: int = WRITE_BATCH_SIZE) -> list[str]:
        """直接写入文本（不经过 to_records 的通用入口），返回写入的 id 列表。

        param:
            texts: 待写入文本列表
            metadatas: 与 texts 等长的元数据列表；整体为 None 表示不存元数据
            ids: 与 texts 等长的 id 列表，缺省用 uuid 生成
            batch_size: 单批写入条数
        """
        if isinstance(texts, str):
            raise TypeError("add_texts 接收 list[str]，单条文本请包成 [text]")
        texts = list(texts)
        if not texts:
            return []

        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {i} 条文本为空")

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(f"metadatas 长度 {len(metadatas)} 与 texts 长度 {len(texts)} 不一致")
        if ids is not None and len(ids) != len(texts):
            raise ValueError(f"ids 长度 {len(ids)} 与 texts 长度 {len(texts)} 不一致")

        resolved_ids = [str(i) for i in ids] if ids is not None else [str(uuid.uuid4()) for _ in texts]
        resolved_metas = None
        if metadatas is not None:
            resolved_metas = [self._clean_metadata(m) for m in metadatas]
            if any(not m for m in resolved_metas):
                raise ValueError("metadata 清洗后存在空 dict，Chroma 要求 metadata 非空；"
                                 "不需要元数据时请把 metadatas 整体传 None")
        return self._upsert(resolved_ids, texts, resolved_metas, batch_size=batch_size)

    # ---------------------------------------------------------------- 查询
    def query(self, text: str, *, top_k: int = DEFAULT_TOP_K,
              where: Mapping[str, Any] | None = None,
              where_document: Mapping[str, Any] | None = None,
              score_threshold: float | None = None) -> list[dict[str, Any]]:
        """按文本检索最相似的记录。

        param:
            text: 查询文本（按 query 类型向量化）
            top_k: 返回条数上限
            where: 元数据过滤条件，如 {"doc_id": "intro.md"}
            where_document: 原文过滤条件，如 {"$contains": "RAG"}
            score_threshold: 相似度下限，低于该值的命中会被丢弃

        return:
            [{"id": str, "text": str, "metadata": dict, "distance": float, "score": float}]
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("查询文本不能为空")

        vector = self._get_embedder().embed_query(text)
        return self.query_by_vector(
            vector, top_k=top_k, where=where, where_document=where_document,
            score_threshold=score_threshold,
        )

    def query_by_vector(self, vector: Sequence[float], *, top_k: int = DEFAULT_TOP_K,
                        where: Mapping[str, Any] | None = None,
                        where_document: Mapping[str, Any] | None = None,
                        score_threshold: float | None = None) -> list[dict[str, Any]]:
        """按已有向量检索，返回结构与 query() 一致。"""
        vector = list(vector)
        if not vector:
            raise ValueError("查询向量不能为空")
        if len(vector) != self.dimension:
            raise ValueError(
                f"查询向量维度为 {len(vector)}，与集合维度 {self.dimension} 不一致"
            )
        if top_k < 1:
            raise ValueError(f"top_k 至少为 1，当前为 {top_k}")

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=dict(where) if where is not None else None,
            where_document=dict(where_document) if where_document is not None else None,
            include=["documents", "metadatas", "distances"],
        )

        ids = self._first(self._field(result, "ids"))
        documents = self._first(self._field(result, "documents"))
        metadatas = self._first(self._field(result, "metadatas"))
        distances = self._first(self._field(result, "distances"))

        hits: list[dict[str, Any]] = []
        for i, record_id in enumerate(ids):
            distance = float(distances[i]) if i < len(distances) else float("nan")
            score = 1.0 - distance
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append({
                "id": record_id,
                "text": documents[i] if i < len(documents) else None,
                "metadata": dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
                "distance": distance,
                "score": score,
            })
        return hits

    def get(self, ids: str | Sequence[str] | None = None, *,
            where: Mapping[str, Any] | None = None,
            limit: int | None = None,
            offset: int | None = None,
            with_embeddings: bool = False) -> list[dict[str, Any]]:
        """按 id 或元数据条件读取记录（不做向量检索）。

        param:
            ids: 单个 id 或 id 列表
            where: 元数据过滤条件
            limit: 返回条数上限
            offset: 分页偏移
            with_embeddings: 是否同时返回向量
        """
        include = ["documents", "metadatas"]
        if with_embeddings:
            include.append("embeddings")

        result = self._collection.get(
            ids=[ids] if isinstance(ids, str) else (list(ids) if ids is not None else None),
            where=dict(where) if where is not None else None,
            limit=limit,
            offset=offset,
            include=include,
        )

        got_ids = self._to_list(self._field(result, "ids"))
        documents = self._to_list(self._field(result, "documents"))
        metadatas = self._to_list(self._field(result, "metadatas"))
        embeddings = self._to_list(self._field(result, "embeddings"))

        records: list[dict[str, Any]] = []
        for i, record_id in enumerate(got_ids):
            item = {
                "id": record_id,
                "text": documents[i] if i < len(documents) else None,
                "metadata": dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
            }
            if with_embeddings:
                item["embedding"] = self._to_list(embeddings[i]) if i < len(embeddings) else []
            records.append(item)
        return records

    def count(self) -> int:
        """集合内记录总数。"""
        return self._collection.count()

    def delete(self, ids: str | Sequence[str] | None = None, *,
               where: Mapping[str, Any] | None = None) -> None:
        """按 id 或元数据条件删除记录；ids 与 where 至少提供一个。

        清空整个集合请用 reset()。
        """
        if ids is None and where is None:
            raise ValueError("delete 需要 ids 或 where，清空集合请用 reset()")

        self._collection.delete(
            ids=[ids] if isinstance(ids, str) else (list(ids) if ids else None),
            where=dict(where) if where is not None else None,
        )

    def reset(self) -> None:
        """删除并重建当前集合（换 embedding 维度或全量重建索引时使用）。"""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": self.DISTANCE_SPACE}},
        )
        logger.info("已重建 Chroma 集合：%s", self.collection_name)

    # ---------------------------------------------------------------- 内部实现
    def _open_collection(self) -> Collection:
        """打开已有集合，不存在则按 cosine 新建。"""
        try:
            collection = self._client.get_collection(name=self.collection_name)
        except NotFoundError:
            collection = self._client.create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": self.DISTANCE_SPACE}},
            )
            logger.debug("新建 Chroma 集合：%s", self.collection_name)
            return collection

        config = collection.configuration or {}
        hnsw = config.get("hnsw") if isinstance(config, Mapping) else None
        space = hnsw.get("space") if isinstance(hnsw, Mapping) else None
        if space and space != self.DISTANCE_SPACE:
            logger.warning(
                "集合 %s 已存在且距离度量为 %s，若score 不按 1 - distance 计算，可能失真",
                self.collection_name, space,
            )
        return collection

    def _get_embedder(self) -> BaiLianEmbeder:
        """懒加载向量化器，只在真正需要 embedding 时要求 API Key。"""
        if self._embedder is None:
            dim = self.dimension if self.dimension in BaiLianEmbeder.SUPPORTED_DIMENSIONS else None
            self._embedder = BaiLianEmbeder(dimension=dim)
        return self._embedder

    def _upsert(self, ids: list[str], texts: list[str], metadatas: list[dict] | None,
                *, batch_size: int) -> list[str]:
        """分批向量化并写入（upsert 保证同 id 覆盖）。"""
        if batch_size < 1:
            raise ValueError(f"batch_size 至少为 1，当前为 {batch_size}")
        if len(set(ids)) != len(ids):
            raise ValueError("本次写入存在重复 id，请检查 doc_id/chunk_id 是否重复")

        embedder = self._get_embedder()
        total = len(texts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            vectors = embedder.embed_documents(batch_texts)
            self._check_vectors(vectors, expected=len(batch_texts))

            self._collection.upsert(
                ids=ids[start:end],
                embeddings=vectors,
                documents=batch_texts,
                metadatas=None if metadatas is None else metadatas[start:end],
            )
            logger.debug("已写入 %d/%d 条记录", end, total)

        return ids

    def _check_vectors(self, vectors: Sequence[Sequence[float]], *, expected: int) -> None:
        """校验向量条数与维度，避免与集合维度不一致导致写入失败。"""
        if len(vectors) != expected:
            raise RuntimeError(f"向量条数与文本不一致：期望 {expected}，实际 {len(vectors)}")
        for i, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise ValueError(
                    f"第 {i} 条向量维度为 {len(vector)}，与集合维度 {self.dimension} 不一致；"
                    "更换 embedding 模型/维度后需先 reset() 重建索引"
                )

    @staticmethod
    def _make_id(record: Mapping[str, Any], *, index: int) -> str:
        """由 doc_id 与 chunk_id 生成稳定 id，保证重复写入可覆盖。"""
        doc_id = record.get("doc_id")
        chunk_id = record.get("chunk_id")
        if doc_id is None or chunk_id is None:
            raise ValueError(f"第 {index} 条记录缺少 doc_id/chunk_id，无法生成 id")
        return f"{doc_id}::{chunk_id}"

    @classmethod
    def _clean_metadata(cls, metadata: Mapping[str, Any] | None) -> dict[str, str | int | float | bool]:
        """过滤 None 并把非标量值转字符串（Chroma 只接受 str/int/float/bool）。"""
        cleaned: dict[str, str | int | float | bool] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            cleaned[str(key)] = value if isinstance(value, cls.METADATA_TYPES) else str(value)
        return cleaned

    @staticmethod
    def _field(result: Any, key: str) -> Any:
        """兼容 dict 与对象两种返回形态。"""
        if isinstance(result, Mapping):
            return result.get(key)
        return getattr(result, key, None)

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        """把 None / numpy 数组 / 普通序列统一成 list。"""
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)

    @classmethod
    def _first(cls, value: Any) -> list[Any]:
        """取单条查询结果的第一层（Chroma 返回的是 [[...]]）。"""
        outer = cls._to_list(value)
        return cls._to_list(outer[0]) if outer else []
