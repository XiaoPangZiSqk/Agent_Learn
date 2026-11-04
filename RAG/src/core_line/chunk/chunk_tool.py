"""
将加载的文档进行节分：

"""
from dataclasses import dataclass
import tiktoken

MAX_CHUNK_SIZE = 560
MIN_CHUNK_SIZE = 120
CHUNK_SIZE = 400
OVERLAP = 60

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
def fixed_len_split(text:str = None, chunk_size = CHUNK_SIZE, overlap = OVERLAP) -> list[str]:
    """固定长度切分
    param:
        chunk_size: 块大小
        overlap: 重叠部分
    """
    if not text :
        return []
    
    sum_token = predic_token(text=text)
    chunks, start = [], 0
    while start < sum_token:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= sum_token:
            break
        start = end - overlap   # 关键：步长 = chunk_size - overlap
    return chunks

#============================================================ 递归字符切分 ============================================================
def recursive_character_split(text:str = None, chunk_size = CHUNK_SIZE, overlap = OVERLAP, separators:list|None=None, level = 0) -> list[str]:
    """递归字符切分：优先自然边界，逐级降级，最后定长硬切兜底"""
    seps = separators or ["\n\n", "\n", "。", "；", "，", " ", ""]
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    def rec(t:str,sepc:list, level = level):
        if not text :
            return []
        
        if predic_token(text=t) <= chunk_size :
            return [text]    # 目前还不能确定元数据，在真正对一个文件切分时在导入。

        sep = seps[level]
        if sep == "":
            return fixed_len_split(chunk_size=chunk_size, overlap=overlap);

        chunks, buf = [], ""
        for p in t.split(sep=sep) :
            candidate = f"{buf}{sep}{p}" if buf else p  # 文本候选池
            if  predic_token(candidate) <= chunk_size : # 未满一个chunk
                buf = candidate
                continue
            # candidate 大于 chunk，将缓存如chunk
            if buf :
                chunks.append(buf)
                buf = buf[-overlap] if overlap else ""
                candidate =f"{buf}{sep}{p}" if buf else p
                if predic_token(candidate) <= chunk_size :  # 只有这里退出，才会进入下面的if
                    buf = candidate
                    continue
            if predic_token(p) > chunk_size :       # 单片段仍超长 → 降级递归
                chunks.extend(rec(p, sepc=sepc, level=level+1))
                buf = ""
            else:
                buf = p                   # 种子放不下则丢种子，保片段完整

            if buf:
                chunks.append(buf)
            return chunks
        
        return rec(text, list(seps))


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
