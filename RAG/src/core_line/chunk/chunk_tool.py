"""
将加载的文档进行节分：

"""
from dataclasses import dataclass
import tiktoken

MAX_CHUNK_SIZE = 560
MIN_CHUNK_SIZE = 100       # 小块合并阈值（target 的 1/4；过紧会产生 9~30 token 碎块）
CHUNK_SIZE = 400
OVERLAP = 60

# ---------- 碎块兜底 ----------
FRAGMENT_TOKENS = 60       # 合并后仍小于此值的块打 fragment=true 标记，检索侧降权

# ---------- 文档形态自适应（代码/围栏密集文档） ----------
CODE_DENSE_THRESHOLD = 0.25    # 围栏/表格字符占比达到该值即视为代码密集文档
CODE_DENSE_TARGET = 600        # 代码密集文档的目标块大小（让相邻代码块装进同块）
CODE_DENSE_MAX = 840           # 对应硬上限（保持 target:max = 400:560 的比例）
CODE_DENSE_MIN = 80            # 对应小块合并阈值

#============================================================ token预测 ============================================================
def predic_token(text: str) -> int:
    """预测text所占token的大小
    cl100k_base:
        词表大小: ~100,256
        中文常用词: 常被拆成 2-3 个字/token
        中文效率: 一般
    o200k_base:
        词表大小: ~199,680(近 2 倍)
        中文常用词: 更多完整词直接映射到单 token
        中文效率: 明显更好(约省 25%-40% token)
    """
    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text)
    return len(ids)

#============================================================ 固定长度切分 ============================================================
def fixed_len_split(text: str = None, chunk_size = CHUNK_SIZE, overlap = OVERLAP) -> list[str]:
    """固定长度切分（按 token 定长硬切，逐级降级后的最后兜底）

    param:
        text: 待切分文本
        chunk_size: 每块的 token 数（注意是 token，不是字符数）
        overlap: 相邻块重叠的 token 数，必须 < chunk_size

    return:
        list[str]，每块 token 数 <= chunk_size；步长 = chunk_size - overlap

    说明：
        按 tiktoken o200k_base 的 token 边界硬切，不保证语义完整；
        极端情况下可能把多字节字符/单词切在两块交界处（定长硬切的固有取舍）。
    """
    if not text:
        return []
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text)
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        piece = enc.decode(ids[start:start + chunk_size])
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(ids):
            break
        start += step
    return chunks

#============================================================ 递归字符切分 ============================================================
def recursive_character_split(text: str = None, chunk_size = CHUNK_SIZE, overlap = OVERLAP,
                              separators: list | None = None, _depth: int = 0) -> list[str]:
    """递归字符切分：优先自然边界，逐级降级，最后定长硬切兜底。

    param:
        text: 待切分文本
        chunk_size: 每块 token 上限（按 predic_token 计量）
        overlap: 相邻块重叠 token 数（取前块尾部 token 作后块开头）
        separators: 边界分隔符列表，逐级降级使用；
            缺省 ["\\n\\n", "\\n", "。", "；", "，", " ", ""]（空串表示定长硬切）
        _depth: 当前降级深度（内部递归使用，外部调用无需传）

    return:
        list[str]，每块 token 数 <= chunk_size

    说明：
        size 一律按 token 计量；分隔符按层级降级，单片段仍超长时用下一层
        分隔符递归切分，全部分隔符用尽后走 fixed_len_split 定长硬切兜底。
    """
    if not text:
        return []
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
    seps = separators if separators is not None else ["\n\n", "\n", "。", "；", "，", " ", ""]

    def tail_tokens(buf: str, n: int) -> str:
        """取 buf 末尾 n 个 token 作为下一块的重叠前缀（token 级，不做字符下标）。"""
        if n <= 0 or not buf:
            return ""
        enc = tiktoken.get_encoding("o200k_base")
        ids = enc.encode(buf)
        return enc.decode(ids[-n:])

    def rec(t: str, depth: int) -> list[str]:
        if not t:
            return []
        if predic_token(t) <= chunk_size:
            return [t]
        if depth >= len(seps) or seps[depth] == "":
            return fixed_len_split(t, chunk_size=chunk_size, overlap=overlap)

        sep = seps[depth]
        chunks: list[str] = []
        buf = ""
        for p in t.split(sep):
            candidate = f"{buf}{sep}{p}" if buf else p
            if predic_token(candidate) <= chunk_size:     # 未满一个 chunk
                buf = candidate
                continue
            if buf:                                        # 当前块已满 -> 落盘
                chunks.append(buf)
                buf = tail_tokens(buf, overlap)            # 前块尾部 token 作重叠种子
                candidate = f"{buf}{sep}{p}" if buf else p
                if predic_token(candidate) <= chunk_size:
                    buf = candidate
                    continue
            if predic_token(p) > chunk_size:               # 单片段仍超长 → 降级递归
                chunks.extend(rec(p, depth + 1))
                buf = ""
            else:
                buf = p                                    # 种子放不下则丢种子，保片段完整
        if buf:
            chunks.append(buf)
        return chunks

    return rec(text, _depth)


#============================================================ 结构化/语法感知切分 ============================================================
import re

# ============================================================
# 1. Markdown 结构解析：行扫描 -> Block
# 这些正则共同构成 Markdown 结构解析的"感官层"：
# 每一条负责识别一类边界类型。解析主循环的判定顺序本身就是边界优先级——
# 围栏 → 标题 → 表格 → 分割线 → 列表 → 引用 → 段落（兜底），
# 先吃原子块，再吃行块，最后才归入段落。
# ============================================================
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")								# ATX 标题（#~`######`）
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*(\S*)\s*$")								# 代码围栏开始行     ```python 、~~~json、裸  ```  都能识别
_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")									# 纯围栏行（无语言名）
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")									# 列表项行
_QUOTE = re.compile(r"^\s*>\s?")													# 引用行 > 
_HR = re.compile(r"^\s*(?:(?:-{3,})|(?:\*{3,})|(?:_{3,}))\s*$")						# 水平分割线 ---
_SETEXT = re.compile(r"^\s*(={3,}|-{3,})\s*$")										# Setext 下划线 ===
_TBL_DELIM = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")			# GFM 表格分隔行
_SENT_SPLIT = re.compile(r"(?<=[。！？；;!?])|(?<=[.!?])(?=\s+[A-Z0-9\"'(\[])")		# 句子边界（零宽）

_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")       # 用于判断文本里是否含中日韩统一表意文字（汉字）的小工具，字符类：匹配"一个"落在该 Unicode 区间的字符


@dataclass
class Block:
    type: str                       # heading/paragraph/list/quote/code/table/hr/frontmatter
    content: str
    path: tuple[str, ...]           # 标题面包屑，含自身
    start: int
    end: int
    level: int = 0
    atomic: bool = False            # True 表示不允许从中间切开


def parse_markdown(md: str) -> list[Block]:
    """
    切分markdown，将整篇文档切分成 标题/段落/列表/引用/代码/表格/分割线/前置内容 Block块，段落是指
    """
    lines = md.splitlines()
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)		# 记录每一行起始偏移位置
        pos += len(ln) + 1

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []	# 记录面包屑路径（level, 标题）

    def path_now() -> tuple[str, ...]:
        return tuple(t for _, t in stack)

    def emit(kind: str, s: int, e: int, level: int = 0, atomic: bool = False) -> None:
        """
        param:
            kind:类型
            s: 开始行
            e: 结束行(包含)
        """
        blocks.append(Block(
            type=kind,
            content="\n".join(lines[s:e + 1]),
            path=path_now(),
            start=offsets[s],
            end=offsets[e] + len(lines[e]),
            level=level,
            atomic=atomic,
        ))

    def push_heading(level: int, title: str) -> None:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    def table_at(k: int) -> bool:	# 该行是否是表格的起始
        return ("|" in lines[k] and k + 1 < n and "|" in lines[k + 1]
                and _TBL_DELIM.match(lines[k + 1]))

    i, n = 0, len(lines)

    # YAML frontmatter（可选支持）
    if n and lines[0].strip() == "---":
        j = 1
        while j < n and lines[j].strip() not in ("---", "..."):
            j += 1
        if j < n:
            emit("frontmatter", 0, j, atomic=True)
            i = j + 1

    while i < n:
        line = lines[i]		# 获取每一行
        if not line.strip():
            i += 1
            continue

        m = _FENCE.match(line)	# 是否是代码块
        if m:                                    # 围栏代码块：整块原子
            marker, flen = m.group(2)[0], len(m.group(2))
            close = re.compile(r"^\s*" + re.escape(marker) + r"{" + str(flen) + r",}\s*$")
            j = i + 1
            while j < n and not close.match(lines[j]):
                j += 1
            e = min(j, n - 1)
            emit("code", i, e, atomic=True)
            i = e + 1
            continue

        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            push_heading(level, m.group(2).strip())
            emit("heading", i, i, level=level)
            i += 1
            continue

        if table_at(i):                            # GFM 表格：整块原子
            j = i + 1
            while j < n and lines[j].strip() and "|" in lines[j]:
                j += 1
            emit("table", i, j - 1, atomic=True)
            i = j
            continue

        if _HR.match(line):
            emit("hr", i, i, atomic=True)
            i += 1
            continue

        if _LIST.match(line):
            j = i
            while j < n and lines[j].strip() and not (
                    _HEADING.match(lines[j]) or _FENCE.match(lines[j])):
                j += 1
            emit("list", i, j - 1)
            i = j
            continue

        if _QUOTE.match(line):
            j = i
            while j < n and lines[j].strip() and _QUOTE.match(lines[j]):
                j += 1
            emit("quote", i, j - 1)
            i = j
            continue

        # 段落（含 setext 标题：文字行后紧跟 === 或 ---）
        # 夹在标题、代码块、引用、列表、表格之间的普通行组成一个Block，类型为段落
        j = i
        while j < n and lines[j].strip():
            if j > i and (_HEADING.match(lines[j]) or _FENCE.match(lines[j])
                          or _QUOTE.match(lines[j]) or _LIST.match(lines[j])
                          or table_at(j)):
                break
            j += 1
        e = j - 1
        if e > i and _SETEXT.match(lines[e]):
            level = 1 if lines[e].strip()[0] == "=" else 2
            push_heading(level, lines[i].strip())
            emit("heading", i, e, level=level)
        else:
            emit("paragraph", i, e)
        i = j

    return blocks


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]
