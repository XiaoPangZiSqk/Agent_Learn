"""
BM25 关键词索引的统一分词器

职责：
    * 把文本切成词项（term），供 BM25 建库与查询共用
    * 中文走 jieba 搜索模式；英文/代码标识符整体保留，并补充子词
    * 归一化（NFKC、小写、去零宽字符）与停用词过滤

约束：
    建库与查询必须使用同一版本的分词规则。TOKENIZER_VERSION 变化时，
    已落盘的 BM25 缓存（data/bm25_index）必须重建，否则两侧词项无法对齐，
    BM25 分数会整体退化为 0。

用法：
    from core_line.embedding.tokenizer import tokenize

    tokens = tokenize("如何配置 Chroma 的 HNSW 参数？")
    # ['配置', 'chroma', 'hnsw', '参数']
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

import jieba

# 分词规则版本号：规则一变就 +1，并让 BM25 缓存失效重建
TOKENIZER_VERSION = 1

# 零宽字符：BOM、零宽空格、零宽连接符。从网页/PDF 复制的内容常夹带这些字符，
# 会让 "chroma" 与 "chroma\u200b" 变成两个不同词项，必须在切分前清掉。
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")

# 片段切分：只保留两类有意义的片段，标点/空白/emoji 直接丢弃
#   1. ASCII 标识符与数字：err_conn_refused、HNSW、v1.2.3
#   2. CJK 连续串：交给 jieba 再切（含扩展 A 区，覆盖生僻字）
_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*|[\u3400-\u4dbf\u4e00-\u9fff]+")

# 标识符内部的分隔符：下划线、点号、连字符
_IDENT_SEP = re.compile(r"[_.\-]+")

# 中文高频虚词：只收功能词，不收 "使用/配置/检索" 这类有实义的词，
# 避免把查询意图一起过滤掉。
ZH_STOPWORDS = frozenset("""
的 了 是 在 和 与 及 或 也 就 都 而 被 把 对 从 到 为 以 于 这 那 其 之 并 但 又 很 更 最
会 能 要 有 无 不 没 我 你 他 她 它 们 个 些 着 过 吗 呢 吧 啊 呀 哦 嗯 请 若 则 即 等 各 每 该 此 另 且
一个 一些 我们 你们 他们 可以 可能 应该 因为 所以 但是 如果 虽然 然后 以及 并且 或者 而是 不是 没有
已经 还是 只是 就是 这个 那个 这些 那些 什么 怎么 如何 为什么 哪些 哪个 是否
""".split())

# 英文高频虚词：只收语法词，不收 use/using/query 等实义词
EN_STOPWORDS = frozenset("""
a an the and or of to in on at for with by from as is are was were be been being
it its this that these those there here
i you he she we they me him her us them my your his their our
do does did done have has had not no so if then than but also just only very
can could should would will shall may might must
""".split())

STOPWORDS = ZH_STOPWORDS | EN_STOPWORDS


def normalize(text: str) -> str:
    """文本归一化：NFKC + 去零宽字符。

    param:
        text: 原始文本

    return:
        归一化后的文本（未做小写，大小写由切分阶段处理）

    说明：
        NFKC 会把全角字母/数字（Ｃｈｒｏｍａ、１２３）转成半角，
        避免同一术语因排版差异被切成不同词项。
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize 需要 str，当前为 {type(text).__name__}")

    return _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """把一段文本切成 BM25 可用的小写词项列表。

    param:
        text: 待切分文本
        drop_stopwords: 是否过滤中英文高频虚词，默认 True

    return:
        词项列表；空文本或全部为标点时返回 []

    说明：
        * 中文使用 jieba 搜索模式（lcut_for_search），同时产出长词与子词，
          例如 "向量检索" 会给出 "向量"、"检索"、"向量检索"，提升长词召回；
        * 英文/代码标识符先整体保留，再按 _ . - 拆出子词，
          例如 "err_conn_refused" 既保留整体，也产出 "err"、"conn"、"refused"；
        * 纯数字串（v1.2.3 除外）不做子词拆分，避免版本号被拆散后噪声过大。
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenize 需要 str，当前为 {type(text).__name__}")

    tokens: list[str] = []
    for segment in _SEGMENT.findall(normalize(text)):
        if _is_cjk(segment[0]):
            tokens.extend(_split_cjk(segment))
        else:
            tokens.extend(_split_identifier(segment))

    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def tokenize_batch(texts: Sequence[str], *, drop_stopwords: bool = True) -> list[list[str]]:
    """批量分词，返回顺序与入参严格一致。

    param:
        texts: 文本列表
        drop_stopwords: 是否过滤停用词

    说明：
        BM25 建库时可一次处理整篇文档的所有 chunk，减少调用开销。
    """
    if isinstance(texts, str):
        raise TypeError("tokenize_batch 接收 list[str]，单条文本请用 tokenize")

    return [tokenize(t, drop_stopwords=drop_stopwords) for t in texts]


def warmup() -> None:
    """预热 jieba 词典。

    jieba 首次调用会加载词典（约 0.5~1 秒），建议在服务启动阶段调用，
    避免第一个查询承担初始化开销。
    """
    jieba.initialize()


# ---------------------------------------------------------------- 内部实现
def _is_cjk(char: str) -> bool:
    """判断字符是否属于 CJK 统一表意文字（含扩展 A 区）。"""
    return "\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff"


def _split_cjk(segment: str) -> list[str]:
    """用 jieba 搜索模式切分中文片段，过滤空白结果。"""
    return [word for word in jieba.lcut_for_search(segment) if word.strip()]


def _split_identifier(segment: str) -> list[str]:
    """切分 ASCII 标识符：整体保留 + 子词补充。

    param:
        segment: 形如 HNSW、err_conn_refused、v1.2.3 的片段

    return:
        小写词项列表

    说明：
        仅当"所有子片段都含字母"时才拆子词。这样：
        * err_conn_refused -> err_conn_refused / err / conn / refused
        * v1.2.3           -> v1.2.3（版本号不拆）
        既保证代码符号可整体命中，也能被部分关键词召回。
    """
    token = segment.lower().strip("._-")
    if not token:
        return []

    tokens = [token]
    if _IDENT_SEP.search(token):
        parts = [p for p in _IDENT_SEP.split(token) if p]
        # 只有全部子片段都含字母（排除纯数字段）才补子词
        if len(parts) > 1 and all(any(ch.isalpha() for ch in p) for p in parts):
            tokens.extend(parts)
    return tokens
