"""
语义缓存（semantic cache）：对嵌入向量、rerank 打分、生成结果做磁盘级缓存。

设计：
- 单文件 sqlite 存储（默认 data/cache/rag_cache.db），线程安全（写锁 + check_same_thread=False）；
- 命名空间隔离：embedding / rerank / generation，互不干扰，可分别清理；
- Key 由 make_key() 对"规范化输入 + 配置指纹"做 sha256，保证：
    * 文档 chunk 不变 -> 向量不变，嵌入缓存可长期复用；
    * (query 归一化, 索引版本) 不变 -> 生成结果可直接命中；
    * (query 归一化, 文本指纹, 模型) 不变 -> rerank 分数可复用；
- TTL 过期即视为未命中并惰性删除；RAG_CACHE_TTL_DAYS <= 0 表示永不过期；
- 总开关 RAG_CACHE_ENABLED=false 时 get_default_cache() 返回 None，调用方零成本跳过。

用法：
    cache = get_default_cache()          # None 表示缓存已关闭
    if cache:
        hit = cache.get_json("generation", key)
        ...
        cache.set_json("generation", key, value)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # 以 RAG/src 为导入根（与 RagApp.py 保持一致）
    from core_line.config import get_cache_enabled, get_cache_path, get_cache_ttl_days
except ImportError:  # 作为包的一部分导入时的兜底
    from .config import get_cache_enabled, get_cache_path, get_cache_ttl_days

logger = logging.getLogger(__name__)

_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DROP_RE = re.compile(r"[\s\W_]+", re.UNICODE)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
)
"""


# ================================================================ Key 工具

def normalize_query(text: str) -> str:
    """查询归一化：NFKC + 小写 + 去空白/标点，只保留文字（CJK 与词字符）。

    归一化后 "K-means 是什么？" 与 "kmeans是什么" 得到不同键但
    "K-means 是什么？" 与 "K-Means 是什么" 得到相同键。
    """
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return _DROP_RE.sub("", normalized)


def make_key(*parts: Any) -> str:
    """把若干部分拼成稳定缓存键（sha256）。None/空串也参与，保证键位稳定。"""
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def text_fingerprint(text: str) -> str:
    """单段文本指纹（sha256 前 32 位，足以避免碰撞且节省键长）。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


# ================================================================ 缓存实现

class SemanticCache:
    """sqlite 语义缓存。enabled=False 时所有读写均为 no-op。"""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        ttl_days: int | None = None,
        enabled: bool = True,
    ):
        """
        param:
            path: 缓存库文件路径，缺省读 RAG_CACHE_PATH（再缺省 data/cache/rag_cache.db）
            ttl_days: 有效期（天），缺省读 RAG_CACHE_TTL_DAYS（再缺省 30）；<=0 永不过期
            enabled: 总开关，False 时本实例所有操作为空操作
        """
        self.enabled = bool(enabled)
        self.path = Path(path).expanduser() if path else get_cache_path()
        self.ttl_days = get_cache_ttl_days() if ttl_days is None else int(ttl_days)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 连接
    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute(SCHEMA)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_ns_created ON cache (namespace, created_at)"
            )
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---------------------------------------------------------------- 读写
    def _expired(self, created_at: float) -> bool:
        return self.ttl_days > 0 and (time.time() - created_at) > self.ttl_days * 86400

    def get_json(self, namespace: str, key: str) -> Any | None:
        """读取一条缓存；未命中/已过期/已关闭返回 None。"""
        if not self.enabled:
            return None
        with self._lock:
            try:
                row = self._connection().execute(
                    "SELECT value, created_at FROM cache WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
                if row is None:
                    return None
                value_raw, created_at = row
                if self._expired(created_at):
                    self._connection().execute(
                        "DELETE FROM cache WHERE namespace=? AND key=?", (namespace, key)
                    )
                    self._connection().commit()
                    return None
                return json.loads(value_raw)
            except sqlite3.Error as exc:
                logger.warning("缓存读取失败（按未命中处理）: %s", exc)
                return None

    def set_json(self, namespace: str, key: str, value: Any) -> None:
        """写入一条缓存（upsert）。"""
        if not self.enabled:
            return
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning("缓存值不可序列化，跳过写入: %s", exc)
            return
        with self._lock:
            try:
                self._connection().execute(
                    "INSERT OR REPLACE INTO cache (namespace, key, value, created_at) VALUES (?,?,?,?)",
                    (namespace, key, payload, time.time()),
                )
                self._connection().commit()
            except sqlite3.Error as exc:
                logger.warning("缓存写入失败（忽略）: %s", exc)

    def get_many(self, namespace: str, keys: Iterable[str]) -> dict[str, Any]:
        """批量读取，返回 {key: value}（仅包含命中项）。"""
        result: dict[str, Any] = {}
        for key in keys:
            value = self.get_json(namespace, key)
            if value is not None:
                result[key] = value
        return result

    def set_many(self, namespace: str, mapping: Mapping[str, Any]) -> None:
        """批量写入 {key: value}。"""
        for key, value in mapping.items():
            self.set_json(namespace, key, value)

    # ---------------------------------------------------------------- 维护
    def clear(self, namespace: str | None = None) -> int:
        """清空指定命名空间（或全部），返回删除条数。"""
        if not self.enabled:
            return 0
        with self._lock:
            try:
                conn = self._connection()
                if namespace is None:
                    cursor = conn.execute("DELETE FROM cache")
                else:
                    cursor = conn.execute("DELETE FROM cache WHERE namespace=?", (namespace,))
                conn.commit()
                return cursor.rowcount or 0
            except sqlite3.Error as exc:
                logger.warning("缓存清理失败: %s", exc)
                return 0

    def purge_expired(self) -> int:
        """主动清理过期条目，返回删除条数（TTL<=0 时为空操作）。"""
        if not self.enabled or self.ttl_days <= 0:
            return 0
        with self._lock:
            try:
                conn = self._connection()
                cursor = conn.execute(
                    "DELETE FROM cache WHERE created_at < ?", (time.time() - self.ttl_days * 86400,)
                )
                conn.commit()
                return cursor.rowcount or 0
            except sqlite3.Error as exc:
                logger.warning("过期缓存清理失败: %s", exc)
                return 0

    def stats(self) -> dict[str, int]:
        """返回 {namespace: 条数} 统计。"""
        if not self.enabled:
            return {}
        with self._lock:
            try:
                rows = self._connection().execute(
                    "SELECT namespace, COUNT(*) FROM cache GROUP BY namespace"
                ).fetchall()
                return {ns: count for ns, count in rows}
            except sqlite3.Error as exc:
                logger.warning("缓存统计失败: %s", exc)
                return {}


# ================================================================ 默认实例

_default_cache: SemanticCache | None = None
_default_lock = threading.Lock()


def get_default_cache() -> SemanticCache | None:
    """获取进程级默认缓存实例（单例）。

    RAG_CACHE_ENABLED=false 或非法命名空间等情形返回 None，调用方直接跳过缓存。
    """
    global _default_cache
    if not get_cache_enabled():
        return None
    with _default_lock:
        if _default_cache is None:
            _default_cache = SemanticCache()
        return _default_cache
