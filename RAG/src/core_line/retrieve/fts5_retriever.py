"""
SQLite FTS5 关键词召回（方案 C：单文件倒排索引）

数据流：
    构建期：to_records() 的 records -> tokenize -> chunks / chunks_fts / docs / meta（同一个 .db 文件）
    查询期：load() 打开数据库 -> FTS5 MATCH（匹配、打分、过滤、截断都在 SQL 内完成）

设计要点：
    * 复用 core_line.embedding.tokenizer：建库与查询共用同一套规则，词项天然对齐
    * 中文切词不交给 FTS5：文本先用 jieba 分好词，用空格拼成 body_tokens / heading_tokens
      两列再入库，FTS5 侧只用 unicode61 处理空白，因此不需要自定义 tokenizer
    * 标题单独成列：bm25(chunks_fts, 1.0, 0.4) 用第二个权重压低头路径命中，
      避免长文档里“标题撞词”压过正文命中
    * 外部内容表（content='chunks'）+ 触发器：分词结果只存一份，正表与倒排索引由 SQLite 保证同步
    * 元数据过滤下推到 SQL（json_extract）：先过滤再截断，返回条数不会因过滤而缩水
    * 单文件 + 单事务：DDL 与写入一起提交或一起回滚，不存在“半份索引”，无需 BM25 的提交标记文件
    * 增量更新按 doc_id 先删后写：文档内容指纹不变时直接跳过，省掉最慢的分词

用法：
    retriever = FTS5Retriever("data/fts5_index/fts5.db")
    retriever.build(records)                 # 构建期：与 store.add_records 并列调用
    retriever.load()                         # 查询期：打开数据库并校验版本
    hits = retriever.search("配置 HNSW 参数", top_k=10, where={"doc_id": "retrieve.md"})

注意：
    * search 返回结构与 BM25Retriever 一致（id/text/metadata/score），可直接交给 RRF 融合；
      但 FTS5 的 bm25 分数与手写 BM25、cosine 三者量纲不同，不能跨通道比较或相加。
    * 表结构与分词器版本写入 meta 表；任一不一致都会拒绝加载，必须 rebuild=True 重建。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # 以 RAG/src 为导入根（与 RagApp.py 保持一致）
    from core_line.embedding.tokenizer import TOKENIZER_VERSION, tokenize
except ImportError:  # 作为包的一部分导入时的兜底
    from ..embedding.tokenizer import TOKENIZER_VERSION, tokenize

logger = logging.getLogger(__name__)

# RAG 项目根目录：.../RAG/src/core_line/retrieve/fts5_retriever.py -> .../RAG
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 单条一语句定义，避免 executescript 隐式提交破坏 build 的事务原子性
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS chunks (
        rowid          INTEGER PRIMARY KEY,
        id             TEXT NOT NULL UNIQUE,
        doc_id         TEXT NOT NULL,
        chunk_id       INTEGER,
        heading_path   TEXT NOT NULL DEFAULT '',
        text           TEXT NOT NULL,
        body_tokens    TEXT NOT NULL,
        heading_tokens TEXT NOT NULL DEFAULT '',
        meta           TEXT NOT NULL DEFAULT '{}',
        token_count    INTEGER NOT NULL DEFAULT 0,
        updated_at     TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        body_tokens,
        heading_tokens,
        content='chunks',
        content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, body_tokens, heading_tokens)
        VALUES (new.rowid, new.body_tokens, new.heading_tokens);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, body_tokens, heading_tokens)
        VALUES ('delete', old.rowid, old.body_tokens, old.heading_tokens);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, body_tokens, heading_tokens)
        VALUES ('delete', old.rowid, old.body_tokens, old.heading_tokens);
        INSERT INTO chunks_fts(rowid, body_tokens, heading_tokens)
        VALUES (new.rowid, new.body_tokens, new.heading_tokens);
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS docs (
        doc_id     TEXT PRIMARY KEY,
        chunks     INTEGER NOT NULL,
        hash       TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

_INSERT_CHUNK = """
INSERT INTO chunks (id, doc_id, chunk_id, heading_path, text,
                    body_tokens, heading_tokens, meta, token_count, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SEARCH_SQL = """
SELECT c.id AS id, c.text AS text, c.meta AS meta,
       -bm25(chunks_fts, ?, ?) AS score
FROM chunks_fts
JOIN chunks AS c ON c.rowid = chunks_fts.rowid
WHERE chunks_fts MATCH ? {where}
ORDER BY score DESC, c.id ASC
LIMIT ?
"""


@dataclass(slots=True)
class _Record:
    """一条待入库 record，metadata 为去掉 id/text 后的全部字段。"""

    id: str
    doc_id: str
    chunk_id: int | None
    heading_path: str
    text: str
    metadata: dict[str, Any]


class FTS5Retriever:
    """基于 SQLite FTS5 的关键词召回器：分词结果落单个 .db 文件，检索在库内完成。"""

    SCHEMA_VERSION = 1
    DB_FILE = "fts5.db"
    DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "fts5_index" / "fts5.db"
    DEFAULT_TOP_K = 20
    META_KEY = "index"
    DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        body_weight: float = 1.0,
        heading_weight: float = 0.4,
    ):
        """
        param:
            db_path: 数据库文件路径；传目录时自动拼接 fts5.db，缺省为 RAG/data/fts5_index/fts5.db
            body_weight: 正文列（body_tokens）在 bm25 打分中的权重
            heading_weight: 标题列（heading_tokens）在 bm25 打分中的权重，压低可减少标题撞词
        """
        if body_weight <= 0:
            raise ValueError(f"body_weight 必须为正数，当前为 {body_weight}")
        if heading_weight < 0:
            raise ValueError(f"heading_weight 不能为负数，当前为 {heading_weight}")

        self.db_path = _resolve_db_path(db_path or self.DEFAULT_DB_PATH)
        self.body_weight = float(body_weight)
        self.heading_weight = float(heading_weight)

        self._conn: sqlite3.Connection | None = None
        self._ready = False

    # ---------------------------------------------------------------- 对外属性
    @property
    def is_ready(self) -> bool:
        """索引是否已构建或已加载（连接可用）。"""
        return self._ready

    @property
    def doc_ids(self) -> list[str]:
        """已建索引的文档 id 列表（来自 docs 表）。"""
        if not self._ready:
            return []
        rows = self._connection().execute("SELECT doc_id FROM docs ORDER BY doc_id").fetchall()
        return [row["doc_id"] for row in rows]

    def __len__(self) -> int:
        if not self._ready:
            return 0
        return int(self._connection().execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # ---------------------------------------------------------------- 构建
    def build(self, records: Sequence[Mapping[str, Any]], *, rebuild: bool = False) -> dict[str, int]:
        """用 to_records() 的产物构建/更新关键词索引。

        param:
            records: chunk 记录列表，需含 doc_id、chunk_id、text
            rebuild: True 表示清空现有索引后全量重建

        return:
            {"records", "chunks", "docs", "tokens", "removed", "skipped"} 统计信息

        说明：
            增量构建（rebuild=False）按 doc_id 先删后写；内容指纹与上次一致的文档直接跳过，
            因此重复构建同一批 records 不会产生重复行，也不会重跑分词。
            同一批 records 内出现重复 id 直接报错，避免静默丢数据。
        """
        grouped = self._group_records(records)
        conn = self._connection()
        existing = {} if rebuild else self._load_doc_hashes(conn)
        if not rebuild:
            self._check_compatible(self._read_meta(conn))   # 版本不符要尽早失败，别先分词再报错

        stamp = _now()
        payload, skipped = self._tokenize_docs(grouped, existing, rebuild=rebuild, stamp=stamp)
        logger.debug("FTS5 分词完成：待写入 %d 篇，跳过 %d 篇", len(payload), len(skipped))

        conn.execute("BEGIN IMMEDIATE")
        try:
            self._ensure_schema(conn)
            removed = 0
            if rebuild:
                removed = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                self._wipe(conn)

            for doc_id, rows in payload.items():
                cursor = conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                removed += max(cursor.rowcount, 0)
                conn.executemany(_INSERT_CHUNK, rows)
                conn.execute(
                    "INSERT INTO docs (doc_id, chunks, hash, updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(doc_id) DO UPDATE SET "
                    "chunks = excluded.chunks, hash = excluded.hash, updated_at = excluded.updated_at",
                    (doc_id, len(rows), _doc_fingerprint(grouped[doc_id]), stamp),
                )

            if rebuild or payload:
                conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")

            stats = self._collect_stats(
                conn,
                records=sum(len(items) for items in grouped.values()),
                removed=removed,
                skipped=skipped,
            )
            self._write_meta(conn, stats, stamp=stamp)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

        self._ready = True
        logger.debug("FTS5 构建完成：%s", stats)
        return stats

    # ---------------------------------------------------------------- 加载
    def load(self) -> bool:
        """打开数据库并校验版本。

        return:
            True 表示加载成功；False 表示数据库文件不存在（调用方可选择重建）

        raise:
            RuntimeError: 文件存在但缺少元数据（构建中断）
            ValueError: 结构版本或分词器版本不匹配
        """
        if not self.db_path.exists():
            self._ready = False
            return False

        conn = self._connection()
        meta = self._read_meta(conn)
        if not meta:
            raise RuntimeError(
                f"FTS5 索引缺少元数据，可能是构建中断，请删除后重建: {self.db_path}"
            )
        self._check_compatible(meta)

        self._ready = True
        logger.debug("FTS5 索引已加载：%s", self.db_path)
        return True

    # ---------------------------------------------------------------- 查询
    def search(self, query: str, *, top_k: int = DEFAULT_TOP_K,
               where: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """按关键词检索 chunk，返回按 bm25 分数降序的结果。

        param:
            query: 查询文本（内部使用与建库相同的分词规则）
            top_k: 返回条数上限
            where: 元数据过滤条件，支持 $eq/$ne/$in/$nin/$gt/$gte/$lt/$lte/$and/$or

        return:
            [{"id", "text", "metadata", "score"}]

        说明：
            过滤在 SQL 内执行（先过滤再截断），因此 top_k 条一定都是满足 where 的记录；
            bm25 分数没有固定上界，不要与向量 cosine 分数直接比较，混合召回请在外部用 RRF。
        """
        if not self._ready:
            raise RuntimeError("FTS5 索引未就绪：请先 build() 或 load()")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询文本不能为空")
        if top_k < 1:
            raise ValueError(f"top_k 至少为 1，当前为 {top_k}")

        where_sql, where_params = _compile_where(where)   # 先校验，无命中也不放过非法条件
        tokens = _dedupe(tokenize(query))
        if not tokens:
            return []

        sql = _SEARCH_SQL.format(where=where_sql)
        params: list[Any] = [
            self.body_weight, self.heading_weight, _build_match(tokens),
            *where_params, top_k,
        ]
        rows = self._connection().execute(sql, params).fetchall()

        return [{
            "id": row["id"],
            "text": row["text"],
            "metadata": json.loads(row["meta"]) if row["meta"] else {},
            "score": float(row["score"]),
        } for row in rows]

    # ---------------------------------------------------------------- 维护
    def clear(self) -> None:
        """关闭连接并删除数据库文件（含 WAL 附属文件），全量重建前可显式调用。"""
        self.close()
        for path in (self.db_path, *_wal_files(self.db_path)):
            path.unlink(missing_ok=True)
        logger.debug("FTS5 索引已清空：%s", self.db_path)

    def close(self) -> None:
        """关闭数据库连接；调用后需重新 load() 才能检索。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._ready = False

    # ---------------------------------------------------------------- 内部实现
    def _connection(self) -> sqlite3.Connection:
        """懒加载连接，首次使用时建目录并打开 WAL 模式。"""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, isolation_level=None)   # 事务由 build 显式控制
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._conn = conn
        return self._conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                raise RuntimeError("当前 SQLite 未启用 FTS5 扩展，无法构建关键词索引") from exc
            raise

    def _wipe(self, conn: sqlite3.Connection) -> None:
        """清空正表与索引；DELETE 会经触发器同步 chunks_fts，再 rebuild 兜底修复损坏的索引。"""
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM docs")
        conn.execute("DELETE FROM meta")
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

    def _load_doc_hashes(self, conn: sqlite3.Connection) -> dict[str, str]:
        try:
            rows = conn.execute("SELECT doc_id, hash FROM docs").fetchall()
        except sqlite3.OperationalError:   # 首次构建，docs 表还不存在
            return {}
        return {row["doc_id"]: row["hash"] for row in rows}

    def _read_meta(self, conn: sqlite3.Connection) -> dict[str, Any]:
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (self.META_KEY,)).fetchone()
        except sqlite3.OperationalError:   # 空库/构建中断，没有 meta 表
            return {}
        if row is None:
            return {}
        try:
            data = json.loads(row["value"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"FTS5 索引元数据损坏，请重建: {self.db_path}") from exc
        return data if isinstance(data, dict) else {}

    def _check_compatible(self, meta: Mapping[str, Any]) -> None:
        if not meta:
            return
        if meta.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"FTS5 索引结构版本不匹配（期望 {self.SCHEMA_VERSION}，"
                f"实际 {meta.get('schema_version')}），请 rebuild=True 重建"
            )
        if meta.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ValueError(
                f"分词器版本不匹配（期望 {TOKENIZER_VERSION}，"
                f"实际 {meta.get('tokenizer_version')}），请 rebuild=True 重建"
            )

    def _write_meta(self, conn: sqlite3.Connection, stats: Mapping[str, int], *, stamp: str) -> None:
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "body_weight": self.body_weight,
            "heading_weight": self.heading_weight,
            "chunk_count": stats["chunks"],
            "doc_count": stats["docs"],
            "token_count": stats["tokens"],
            "built_at": stamp,
        }
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (self.META_KEY, json.dumps(meta, ensure_ascii=False, indent=2)),
        )

    def _collect_stats(self, conn: sqlite3.Connection, *, records: int,
                       removed: int, skipped: int) -> dict[str, int]:
        row = conn.execute(
            "SELECT COUNT(*) AS chunks, COALESCE(SUM(token_count), 0) AS tokens FROM chunks"
        ).fetchone()
        docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        return {
            "records": records,
            "chunks": int(row["chunks"]),
            "docs": int(docs),
            "tokens": int(row["tokens"]),
            "removed": removed,
            "skipped": skipped,
        }

    def _group_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, list[_Record]]:
        """校验记录并按 doc_id 分组；返回的 dict 保持首次出现顺序。"""
        if isinstance(records, Mapping):
            raise TypeError("build 接收记录列表，单条记录请包成 [record]")

        records = list(records)   # 允许传入生成器
        grouped: dict[str, list[_Record]] = {}
        seen: set[str] = set()
        for i, record in enumerate(records):
            parsed = _parse_record(record, index=i)
            if parsed.id in seen:
                raise ValueError(f"第 {i} 条记录 id 重复: {parsed.id}")
            seen.add(parsed.id)
            grouped.setdefault(parsed.doc_id, []).append(parsed)
        return grouped

    def _tokenize_docs(
        self,
        grouped: Mapping[str, list[_Record]],
        existing: Mapping[str, str],
        *,
        rebuild: bool,
        stamp: str,
    ) -> tuple[dict[str, list[tuple]], list[str]]:
        """只对需要重建的文档分词，返回 (待写入行, 被跳过的 doc_id 列表)。"""
        payload: dict[str, list[tuple]] = {}
        skipped: list[str] = []

        for doc_id, items in grouped.items():
            if not rebuild and existing.get(doc_id) == _doc_fingerprint(items):
                skipped.append(doc_id)
                continue

            rows: list[tuple] = []
            for record in items:
                body = tokenize(record.text)
                heading = tokenize(record.heading_path) if record.heading_path else []
                rows.append((
                    record.id,
                    doc_id,
                    record.chunk_id,
                    record.heading_path,
                    record.text,
                    " ".join(body),
                    " ".join(heading),
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    len(body),
                    stamp,
                ))
            payload[doc_id] = rows

        return payload, skipped


# ---------------------------------------------------------------- 工具函数
def _resolve_db_path(path: str | Path) -> Path:
    """目录形态的入参补上 DB_FILE，文件形态原样使用。"""
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        return (resolved / FTS5Retriever.DB_FILE).resolve()
    if resolved.suffix.lower() in FTS5Retriever.DB_SUFFIXES:
        return resolved.resolve()
    return (resolved / FTS5Retriever.DB_FILE).resolve()


def _wal_files(db_path: Path) -> tuple[Path, Path]:
    """WAL 模式下的附属文件，删除数据库时要一起清理。"""
    return Path(f"{db_path}-wal"), Path(f"{db_path}-shm")


def _parse_record(record: Mapping[str, Any], *, index: int) -> _Record:
    """校验并归一化一条 record。"""
    if not isinstance(record, Mapping):
        raise TypeError(f"第 {index} 条记录不是 dict: {type(record).__name__}")

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"第 {index} 条记录的 text 为空")

    raw_doc_id = record.get("doc_id")
    if raw_doc_id is None:
        raise ValueError(f"第 {index} 条记录缺少 doc_id")
    doc_id = str(raw_doc_id)

    record_id = record.get("id")
    if record_id:
        record_id = str(record_id)
    elif record.get("chunk_id") is not None:
        record_id = f"{doc_id}::{record['chunk_id']}"
    else:
        raise ValueError(f"第 {index} 条记录缺少 chunk_id，无法生成 id")

    heading_path = record.get("heading_path")
    return _Record(
        id=record_id,
        doc_id=doc_id,
        chunk_id=_as_int(record.get("chunk_id")),
        heading_path=str(heading_path) if heading_path is not None else "",
        text=text,
        metadata={k: v for k, v in record.items() if k not in ("id", "text")},
    )


def _as_int(value: Any) -> int | None:
    """chunk_id 统一成整数列，非数字值（如自定义字符串 id）存 NULL。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _dedupe(tokens: Sequence[str]) -> list[str]:
    """query 词项去重（保持顺序），避免子词扩展造成重复加权。"""
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _build_match(tokens: Sequence[str]) -> str:
    """把词项拼成 FTS5 查询表达式：每个词加双引号，词间 OR。

    说明：
        加引号是为了让 `err_conn_refused`、`v1.2.3` 这类词项被当作短语整体匹配，
        避免其中的 `-`、`.`、`_` 被 FTS5 当成语法符号；
        用 OR 而不是默认的隐式 AND，与 BM25 通道"多词命中累加分"的行为对齐。
    """
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def _doc_fingerprint(items: Sequence[_Record]) -> str:
    """按 id 排序后对文本做指纹，用于发现"chunk 数不变但内容变了"。"""
    digest = hashlib.blake2b(digest_size=8)
    for record in sorted(items, key=lambda item: item.id):
        digest.update(record.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(record.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 元数据过滤
_ALLOWED_OPS = frozenset({"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"})
_COMPARE_OPS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}

# 独立成列的字段走列过滤，其余字段落 meta JSON，用 json_extract 下推
_COLUMN_EXPRS = {
    "doc_id": "c.doc_id",
    "chunk_id": "c.chunk_id",
    "heading_path": "c.heading_path",
}

# json_extract 的路径不能走参数转义，键名先行格式约束，杜绝注入
_FIELD_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _compile_where(where: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    """把 Chroma 风格的 where 编译成 SQL 片段（带前导 AND）与参数列表。"""
    if where is None:
        return "", []
    if not isinstance(where, Mapping):
        raise TypeError(f"where 需要 dict，当前为 {type(where).__name__}")
    if not where:
        return "", []

    clauses, params = _compile_conditions(where)
    return " AND " + " AND ".join(f"({clause})" for clause in clauses), params


def _compile_conditions(conditions: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    if not isinstance(conditions, Mapping):
        raise TypeError(f"过滤条件需要 dict，当前为 {type(conditions).__name__}")
    if not conditions:
        raise ValueError("过滤条件不能为空")

    clauses: list[str] = []
    params: list[Any] = []
    for key, condition in conditions.items():
        if key in ("$and", "$or"):
            clause, sub_params = _compile_group(key, condition)
        else:
            clause, sub_params = _compile_field(key, condition)
        clauses.append(clause)
        params.extend(sub_params)
    return clauses, params


def _compile_group(key: str, condition: Any) -> tuple[str, list[Any]]:
    if not isinstance(condition, Sequence) or isinstance(condition, (str, bytes)):
        raise ValueError(f"{key} 需要条件列表")
    if not condition:
        raise ValueError(f"{key} 的条件列表不能为空")

    joiner = " AND " if key == "$and" else " OR "
    parts: list[str] = []
    params: list[Any] = []
    for item in condition:
        sub_clauses, sub_params = _compile_conditions(item)
        parts.append("(" + " AND ".join(f"({clause})" for clause in sub_clauses) + ")")
        params.extend(sub_params)
    return "(" + joiner.join(parts) + ")", params


def _compile_field(key: str, condition: Any) -> tuple[str, list[Any]]:
    if not isinstance(key, str) or not _FIELD_KEY.match(key):
        raise ValueError(f"不支持的过滤字段名: {key!r}")

    expr, expr_params = _column_expr(key)
    if not isinstance(condition, Mapping):
        # 直接比较等价于 $eq；用 IS 兼容 None
        return f"{expr} IS ?", [*expr_params, condition]
    if not condition:
        raise ValueError(f"字段 {key} 的过滤条件不能为空")

    clauses: list[str] = []
    params: list[Any] = []
    for operator, expected in condition.items():
        if operator not in _ALLOWED_OPS:
            raise ValueError(f"不支持的过滤操作符: {operator}")
        clause, clause_params = _compile_operator(operator, expected, expr, expr_params)
        clauses.append(clause)
        params.extend(clause_params)
    return " AND ".join(f"({clause})" for clause in clauses), params


def _compile_operator(operator: str, expected: Any, expr: str,
                      expr_params: Sequence[Any]) -> tuple[str, list[Any]]:
    if operator == "$eq":
        return f"{expr} IS ?", [*expr_params, expected]
    if operator == "$ne":
        # IS NOT 是 NULL 安全的：字段缺失时也算"不等于"
        return f"{expr} IS NOT ?", [*expr_params, expected]
    if operator in ("$in", "$nin"):
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            raise ValueError(f"{operator} 需要列表参数")
        values = list(expected)
        if not values:
            return ("0" if operator == "$in" else "1"), []
        marks = ", ".join("?" for _ in values)
        if operator == "$in":
            return f"{expr} IN ({marks})", [*expr_params, *values]
        # 字段缺失视为"不在列表中"，与 Python 侧过滤语义一致
        return (f"({expr} IS NULL OR {expr} NOT IN ({marks}))",
                [*expr_params, *expr_params, *values])
    return f"{expr} {_COMPARE_OPS[operator]} ?", [*expr_params, expected]


def _column_expr(key: str) -> tuple[str, list[Any]]:
    """字段名 -> (SQL 表达式, 该表达式占位符的参数)。"""
    column = _COLUMN_EXPRS.get(key)
    if column is not None:
        return column, []
    return "json_extract(c.meta, ?)", [f"$.{key}"]
