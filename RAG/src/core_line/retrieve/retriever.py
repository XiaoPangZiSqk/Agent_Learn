"""
召回主逻辑：将多种召回方式组合，然后进行RRF融合，返回融合后的结果。
目前已有 向量召回（ChromaVector）与关键词召回（BM25Retriever / FTS5Retriever，可互换或并列）。

召回对象与范围
--------------
* 对象：chunk 级记录（MarkdownChunker + to_records() 的切分产物），不是整篇文档。
  唯一键为 id = "{doc_id}::{chunk_id}"，与各通道写入时的 id 规则一致，
  同一 chunk 在不同通道命中时按 id 精确合并去重。
* 范围：各通道自身索引内全部 chunk 的并集。前提是各通道用同一批 to_records()
  records 构建（同源），否则 id 无法对齐，跨通道去重会失效。
* 过滤：where 为 Chroma 风格元数据过滤条件，统一下推给所有通道（各通道均先过滤
  再截断）；向量通道特有的 where_document / score_threshold 不在本层暴露。
* 深度：每路通道统一召回 channel_top_k 条（默认 20），融合后截取 top_k 条（默认 10）。
  各通道深度必须一致，否则排名偏置会系统性偏向召回更深的通道。

融合规则（RRF，按排名融合）
---------------------------
    score(d) = Σ_c  weight_c / (rrf_k + rank_c(d))

* rank_c(d)：d 在通道 c 结果列表中的排名，从 1 开始（各通道已按自身分数降序返回）；
* rrf_k：平滑常数，默认 60，防止第 1 名过度支配；候选池小（如各路 top_k=10）可调小
  到 10~20 拉开头部差距；
* 只看排名、不看原始分数：cosine 与 bm25 量纲不同，原始 score 禁止跨通道比较或相加；
* 只命中部分通道的记录，未命中通道不贡献分数（不惩罚）；同一通道内重复 id 只取
  最优（最先出现的）排名；
* 最终按融合分降序、id 升序（稳定 tie-break）排序后截取 top_k。

返回结构（供交叉验证器重排）
---------------------------
    [{
        "id":       str,     # "{doc_id}::{chunk_id}"，跨通道唯一键
        "text":     str,     # chunk 正文，重排的直接输入
        "metadata": dict,    # chunk 元数据（doc_id/chunk_id/heading_path 等）
        "score":    float,   # RRF 融合分，越大越相关，重排排序依据
        "channels": [        # 命中来源，按通道注册顺序
            {"channel": str, "rank": int, "score": float | None},
        ],
    }, ...]                   # 按 score 降序、id 升序

* id/text/metadata 与各通道命中结构对齐，交叉验证器可直接取 text 重排，无需回查索引；
* channels 给出每个通道的命中排名与原始分数，交叉验证器可据此判断通道一致性
  （多通道共同命中且排名靠前的记录更可靠）；
* channels[].score 是通道原始分（cosine 相似度或 bm25 分），量纲不同，仅供诊断
  与通道内比较，不要跨通道比较或参与排序。

用法：
    from core_line.vector_store.chromadb_store import ChromaVector
    from core_line.retrieve.fts5_retriever import FTS5Retriever
    from core_line.retrieve.retriever import HybridRetriever, RecallChannel

    store = ChromaVector()                  # 向量通道：需已 add_records 建好索引
    fts5 = FTS5Retriever()                  # 关键词通道：可换 BM25Retriever，接口一致
    if not fts5.load():
        raise RuntimeError("FTS5 索引缺失，请先构建")

    hybrid = HybridRetriever([
        RecallChannel("vector", store),
        RecallChannel("fts5", fts5),
    ])
    hits = hybrid.search("如何配置 HNSW 参数", top_k=10, where={"doc_id": "retrieve.md"})
    # hits -> [{"id", "text", "metadata", "score", "channels"}] -> 交给交叉验证器重排

注意：
    * 各通道必须先 build/load 就绪，未就绪会抛出其原生异常（如 RuntimeError），本层不降级；
    * 通道召回器需提供 search(query, top_k=, where=) 或 query(text, top_k=, where=) 之一
      （BM25Retriever / FTS5Retriever 为 search，ChromaVector 为 query），本层自动适配；
    * 通道 weight 是 RRF 加权系数（weight_c / (rrf_k + rank)），需离线评估，不确定时保持 1.0；
    * 融合只做召回合并，不做精排；输出的 top_k 应大于交叉验证器最终需要的条数。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecallChannel:
    """一路召回通道：通道名 + 召回器 + RRF 权重。

    param:
        name: 通道名，写入返回结果的 channels[].channel，同一配置内需唯一
        retriever: 召回器实例，需提供 search(query, top_k=, where=) 或
            query(text, top_k=, where=)；BM25Retriever / FTS5Retriever / ChromaVector 可直接传入
        weight: RRF 权重系数，必须为正数，默认 1.0；不确定时保持默认
    """

    name: str
    retriever: Any
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("通道 name 不能为空")
        if not isinstance(self.weight, (int, float)):
            raise TypeError(f"通道 {self.name} 的 weight 需为数字，当前为 {type(self.weight).__name__}")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"通道 {self.name} 的 weight 必须为正的有限数，当前为 {self.weight!r}")
        if not (callable(getattr(self.retriever, "search", None))
                or callable(getattr(self.retriever, "query", None))):
            raise TypeError(f"通道 {self.name} 的召回器需提供 search() 或 query()")


class HybridRetriever:
    """多路召回 + RRF 融合：同一查询并行取回各通道命中，按排名融合后交重排层。"""

    DEFAULT_TOP_K = 10           # 融合后返回条数
    DEFAULT_CHANNEL_TOP_K = 20   # 每路通道的召回深度（各通道保持一致）
    DEFAULT_RRF_K = 60           # RRF 平滑常数
    FRAGMENT_PENALTY = 0.85      # 碎块降权：fragment=true 的候选融合分打折（切分器标注）

    def __init__(
        self,
        channels: Sequence[RecallChannel],
        *,
        rrf_k: float = DEFAULT_RRF_K,
        channel_top_k: int = DEFAULT_CHANNEL_TOP_K,
    ) -> None:
        """
        param:
            channels: 召回通道列表（至少一路），如
                [RecallChannel("vector", store), RecallChannel("fts5", fts5)]
            rrf_k: RRF 平滑常数（非负），默认 60；越大头部差距越小
            channel_top_k: 每路通道的默认召回深度，search() 可按次覆盖

        说明：
            通道 name 必须唯一；各通道召回器需已 build/load 就绪。
        """
        if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
            raise TypeError("channels 需为 RecallChannel 序列")
        channels = list(channels)
        if not channels:
            raise ValueError("channels 至少需要一路召回通道")
        if not all(isinstance(channel, RecallChannel) for channel in channels):
            raise TypeError("channels 的每一项都必须是 RecallChannel")
        names = [channel.name for channel in channels]
        if len(set(names)) != len(names):
            raise ValueError(f"通道 name 重复：{names}")
        if not isinstance(rrf_k, (int, float)):
            raise TypeError(f"rrf_k 需为数字，当前为 {type(rrf_k).__name__}")
        if not math.isfinite(rrf_k) or rrf_k < 0:
            raise ValueError(f"rrf_k 必须为非负有限数，当前为 {rrf_k!r}")
        if not isinstance(channel_top_k, int):
            raise TypeError(f"channel_top_k 需为整数，当前为 {type(channel_top_k).__name__}")
        if channel_top_k < 1:
            raise ValueError(f"channel_top_k 至少为 1，当前为 {channel_top_k}")

        self.channels: tuple[RecallChannel, ...] = tuple(channels)
        self.rrf_k = float(rrf_k)
        self.channel_top_k = channel_top_k

    # ---------------------------------------------------------------- 查询
    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        where: Mapping[str, Any] | None = None,
        channel_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """对 query 做多路召回并 RRF 融合，返回融合后的 chunk 列表。

        param:
            query: 查询文本（指定要召回的内容）
            top_k: 融合后返回条数上限
            where: 元数据过滤条件（Chroma 风格），统一下推给所有通道
            channel_top_k: 每路通道的召回深度，缺省用实例的 channel_top_k

        return:
            [{"id", "text", "metadata", "score", "channels"}]，按 score 降序、id 升序；
            score 为 RRF 融合分，channels 为各通道命中详情，供交叉验证器重排。

        说明：
            各通道必须已就绪；任一通道抛出的异常原样向上传播，本层不做降级。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询文本不能为空")
        if top_k < 1:
            raise ValueError(f"top_k 至少为 1，当前为 {top_k}")
        depth = self.channel_top_k if channel_top_k is None else channel_top_k
        if depth < 1:
            raise ValueError(f"channel_top_k 至少为 1，当前为 {depth}")

        fused: dict[str, dict[str, Any]] = {}
        for channel in self.channels:
            hits = self._recall(channel, query, depth, where)
            logger.debug("通道 %s 召回 %d 条", channel.name, len(hits))
            seen: set[str] = set()
            for rank, hit in enumerate(hits, start=1):
                record_id = hit.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(f"通道 {channel.name} 第 {rank} 条命中缺少有效 id，无法融合")
                if record_id in seen:
                    continue   # 同通道重复 id：只取最优（最先出现的）排名
                seen.add(record_id)

                entry = fused.get(record_id)
                if entry is None:
                    entry = {
                        "id": record_id,
                        "text": hit.get("text"),
                        "metadata": dict(hit.get("metadata") or {}),
                        "score": 0.0,
                        "channels": [],
                    }
                    fused[record_id] = entry
                entry["score"] += channel.weight / (self.rrf_k + rank)
                entry["channels"].append({
                    "channel": channel.name,
                    "rank": rank,
                    "score": hit.get("score"),
                })

        # 碎块降权：切分器标注的 fragment 候选稍作压制，避免碎块挤占头部
        for entry in fused.values():
            if entry["metadata"].get("fragment"):
                entry["score"] *= self.FRAGMENT_PENALTY

        results = sorted(fused.values(), key=lambda item: (-item["score"], item["id"]))
        logger.debug("融合完成：%d 路通道，候选 %d 条，返回 %d 条",
                     len(self.channels), len(fused), min(len(results), top_k))
        return results[:top_k]

    # ---------------------------------------------------------------- 内部实现
    @staticmethod
    def _recall(channel: RecallChannel, query: str, top_k: int,
                where: Mapping[str, Any] | None) -> list[Any]:
        """调用单路通道召回；自动适配 search()（BM25/FTS5）与 query()（ChromaVector）。"""
        search = getattr(channel.retriever, "search", None)
        if callable(search):
            hits = search(query, top_k=top_k, where=where)
        else:
            hits = channel.retriever.query(query, top_k=top_k, where=where)
        return list(hits) if hits is not None else []
