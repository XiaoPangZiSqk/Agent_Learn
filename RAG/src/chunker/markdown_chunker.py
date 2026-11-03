"""
Markdown 切分器
"""
from dataclasses import dataclass
from core_line.chunk import ChunkTool


@dataclass
class Chunk:
    core: str                                   # 本体文本
    heading_path: tuple[str, ...]
    start: int                                  # 在源文档中的字符偏移
    end: int
    block_types: tuple[str, ...]
    prefix: str = ""                            # 重叠区（来自上一块句尾）
    overlap_tokens: int = 0
    index: int = -1

    @property
    def text(self) -> str:
        return (self.prefix + "\n\n" + self.core) if self.prefix else self.core

    @property
    def tokens(self) -> int:
        return ChunkTool.predic_token(self.text)


# ============================================================
# 切分器
# ============================================================
def _fence_wrap(text: str) -> tuple[str, str]:
    """识别已包裹围栏的代码块，返回 (头, 尾)，用于超长时按行重包。"""
    first, sep, rest = text.partition("\n")
    m = ChunkTool._FENCE.match(first)
    if not m or not sep:
        return None
    marker = m.group(2)[0] * 3
    last = rest.split("\n")[-1].strip()
    tail = last if ChunkTool._FENCE_LINE.match(last) else marker
    return first, tail


class MarkdownChunker:
    """边界感知的 Markdown 切分器。

    target   : 目标块大小（软预算）
    max_size : 硬上限，超过必须在句子/行级切开
    min_size : 小于该值的块尝试并入同根相邻块
    overlap  : 相邻块重叠预算（自动吸附到句子边界）
    """

    def __init__(self, target: int = ChunkTool.CHUNK_SIZE, max_size: int = ChunkTool.MAX_CHUNK_SIZE,
                 min_size: int = ChunkTool.MIN_CHUNK_SIZE, overlap: int = ChunkTool.OVERLAP):
        assert 0 < target <= max_size, "target 必须 <= max_size"
        self.target = target
        self.max_size = max_size
        self.min_size = min_size
        self.overlap = overlap

    # ---------- 主入口 ----------
    def chunk(self, md: str) -> list[Chunk]:
        blocks = ChunkTool.parse_markdown(md)
        out: list[Chunk] = []
        for path, section in self._split_sections(blocks):
            units = self._pack_blocks(section, path) 	#section:list[Block] , path:( , , ...)
            out.extend(self._apply_overlap(units))
        out = self._merge_small(out)
        for i, c in enumerate(out):
            c.index = i
        return out

    # ---------- 标题分节：每个标题开启新节 ----------
    @staticmethod
    def _split_sections(blocks: list[ChunkTool.Block]):
        """
        找出某一标题路径下的所有block，封装成(path, Blocks)
        """
        sections, cur, cur_path = [], [], ()
        for b in blocks:
            if b.type == "heading":
                if cur:
                    sections.append((cur_path, cur))
                cur_path, cur = b.path, [b]
            else:
                if not cur:
                    cur_path = b.path
                cur.append(b)
        if cur:
            sections.append((cur_path, cur))
        return sections

    # ---------- 节内装箱 ----------
    def _pack_blocks(self, blocks: list[ChunkTool.Block], path) -> list[Chunk]:
        """
        将多个
        """
        chunks: list[Chunk] = []
        cur: list[ChunkTool.Block] = []	# 组合成chunk所需的block块
        cur_tokens = 0

        def flush() -> None:
            """将多个block合成一个chunk"""
            nonlocal cur, cur_tokens
            if not cur:
                return
            items = [(b.content, b.start, b.end) for b in cur]
            chunks.append(self._make_unit(items, path,
                                          tuple(b.type for b in cur)))
            cur, cur_tokens = [], 0

        for b in blocks:
            tk = ChunkTool.predic_token(b.content)
            if b.atomic:
                # 原子块：允许把块撑到 target~max，也不切中；超 max 才降级切
                if tk > self.max_size:
                    flush()
                    chunks.extend(self._split_atomic(b, path))
                elif not cur:
                    cur, cur_tokens = [b], tk
                elif cur_tokens + tk > self.max_size:
                    flush()
                    cur, cur_tokens = [b], tk
                else:
                    cur.append(b)
                    cur_tokens += tk
                continue

            if cur_tokens + tk <= self.target:
                cur.append(b)
                cur_tokens += tk
            elif tk > self.max_size:          # 超长段落 -> 句子级切分
                flush()
                chunks.extend(self._split_long(b, path))
            else:
                flush()
                cur, cur_tokens = [b], tk
        flush()
        return chunks

    @staticmethod
    def _make_unit(items, path, types, join: str = "\n\n") -> Chunk:
        return Chunk(core=join.join(t for t, _, _ in items),
                     heading_path=path,
                     start=items[0][1], end=items[-1][2],
                     block_types=types)

    # ---------- 超长原子块降级 ----------
    def _split_atomic(self, b: ChunkTool.Block, path) -> list[Chunk]:
        if b.type == "code":
            return self._split_code(b, path)
        if b.type == "table":
            return self._split_table(b, path)
        return self._split_lines(b, path)

    def _split_code(self, b: ChunkTool.Block, path) -> list[Chunk]:
        lines = b.content.split("\n")
        if len(lines) < 3:
            return self._split_lines(b, path)
        head, tail = lines[0], lines[-1]
        if not ChunkTool._FENCE_LINE.match(tail):
            tail = ("`" * 3) if "`" in head else ("~" * 3)
            inner = lines[1:]
        else:
            inner = lines[1:-1]

        groups, cur = [], []
        for ln in inner:                       # 空行分组，组内不切
            if not ln.strip() and cur:
                groups.append(cur)
                cur = []
            elif ln.strip() or cur:
                cur.append(ln)
        if cur:
            groups.append(cur)
        if not groups:
            return self._split_lines(b, path)

        items, cursor = [], 0                  # 记录每组在原文中的偏移
        for g in groups:
            body = "\n".join(g)
            idx = b.content.find(body, cursor)
            if idx < 0:
                idx = cursor
            wrapped = "\n".join([head] + g + [tail])
            items.append((wrapped, b.start + idx, b.start + idx + len(body)))
            cursor = idx + len(body)
        return self._pack_pieces(items, path, "code")

    def _split_table(self, b: ChunkTool.Block, path) -> list[Chunk]:
        lines = b.content.split("\n")
        if len(lines) < 3:
            return self._split_lines(b, path)
        head, delim, rows = lines[0], lines[1], lines[2:]
        if ChunkTool.predic_token(head + delim) > self.max_size:
            return self._split_lines(b, path)

        head_tk = ChunkTool.predic_token(head + "\n" + delim)
        groups, cur, cur_t = [], [], head_tk
        for r in rows:                          # 行级装箱
            tk = ChunkTool.predic_token(r)
            if cur and cur_t + tk > self.max_size:
                groups.append(cur)
                cur, cur_t = [], head_tk
            cur.append(r)
            cur_t += tk
        if cur:
            groups.append(cur)

        items, cursor = [], 0
        for g in groups:
            body = "\n".join(g)
            idx = b.content.find(body, cursor)
            if idx < 0:
                idx = cursor
            # 表头恢复：每片都重挂表头，单独召回时列名仍在
            items.append(("\n".join([head, delim] + g),
                          b.start + idx, b.start + idx + len(body)))
            cursor = idx + len(body)
        return self._pack_pieces(items, path, "table")

    def _split_lines(self, b: ChunkTool.Block, path) -> list[Chunk]:
        items, cursor = [], 0
        for ln in b.content.split("\n"):
            idx = b.content.find(ln, cursor) if ln else cursor
            if idx < 0:
                idx = cursor
            items.append((ln, b.start + idx, b.start + idx + len(ln)))
            cursor = idx + len(ln) + 1
        return self._pack_lines(items, path, b.type)

    def _split_long(self, b: ChunkTool.Block, path) -> list[Chunk]:
        sents = ChunkTool.split_sentences(b.content)
        if not sents:
            return self._split_lines(b, path)
        joiner = "" if ChunkTool._HAS_CJK.search(b.content) else " "
        items, cursor = [], 0
        for s in sents:
            idx = b.content.find(s, cursor)
            if idx < 0:
                idx = cursor
            items.append((s, b.start + idx, b.start + idx + len(s)))
            cursor = idx + len(s)
        return self._pack_pieces(items, path, b.type, join=joiner)

    # ---------- 片段级装箱 ----------
    def _pack_pieces(self, items, path, kind: str, join: str = "\n\n") -> list[Chunk]:
        chunks: list[Chunk] = []
        cur: list[tuple[str, int, int]] = []
        cur_t = 0

        def flush() -> None:
            nonlocal cur, cur_t
            if not cur:
                return
            chunks.append(self._make_unit(cur, path, (kind,), join))
            cur, cur_t = [], 0

        for text, s, e in items:
            tk = ChunkTool.predic_token(text)
            if tk > self.max_size:
                flush()
                chunks.extend(self._pack_lines([(text, s, e)], path, kind,
                                               wrap=_fence_wrap(text) if kind == "code" else None))
                continue
            if cur and cur_t + tk > self.target:
                flush()
            cur.append((text, s, e))
            cur_t += tk
        flush()
        return chunks

    def _pack_lines(self, items, path, kind: str,
                    wrap: tuple[str,str]|None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        cur: list[str] = []
        cur_t = 0
        span = [0, 0]

        def flush() -> None:
            nonlocal cur, cur_t
            if not cur:
                return
            body = "\n".join(cur)
            if wrap:
                body = "\n".join([wrap[0], body, wrap[1]])
            chunks.append(Chunk(core=body, heading_path=path,
                                start=span[0], end=span[1],
                                block_types=(kind,)))
            cur, cur_t = [], 0

        for text, s, e in items:
            tk = ChunkTool.predic_token(text)
            if cur and cur_t + tk > self.target:
                flush()
            if not cur:
                span[0] = s
            cur.append(text)
            span[1] = e
            cur_t += tk
        flush()
        return chunks

    # ---------- 重叠：吸附到句子边界 ----------
    def _apply_overlap(self, units: list[Chunk]) -> list[Chunk]:
        prev: Chunk|None = None
        for u in units:
            if prev is not None and self.overlap > 0 and \
                    any(t in ("paragraph", "list", "quote") for t in prev.block_types):
                tail = self._tail(prev.core, self.overlap)
                if tail:
                    u.prefix = tail
                    u.overlap_tokens = ChunkTool.predic_token(tail)
            prev = u
        return units

    def _tail(self, text: str, budget: int) -> str:
        sents = ChunkTool.split_sentences(text)
        picked: list[str] = []
        total = 0
        for s in reversed(sents):
            tk = ChunkTool.predic_token(s)
            if picked and total + tk > budget:
                break
            picked.append(s)
            total += tk
        picked.reverse()
        joiner = "" if ChunkTool._HAS_CJK.search(text) else " "
        return joiner.join(picked)

    # ---------- 小块合并：避免碎片 ----------
    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        merged: list[Chunk] = []
        for c in chunks:
            if (merged
                    and ChunkTool.predic_token(c.core) < self.min_size
                    and merged[-1].heading_path[:1] == c.heading_path[:1]
                    and ChunkTool.predic_token(merged[-1].core + c.core) <= self.target):
                p = merged[-1]
                p.core = p.core + "\n\n" + c.core
                p.end = max(p.end, c.end)
                p.block_types = p.block_types + c.block_types
                p.overlap_tokens = 0
                p.prefix = ""
            else:
                merged.append(c)
        return merged


# ============================================================
# 4. 交给向量库的记录
# ============================================================
def to_records(chunks: list[Chunk], doc_id: str) -> list[dict]:
    return [{
        "doc_id": doc_id,
        "chunk_id": c.index,
        "heading_path": " > ".join(c.heading_path),
        "text": c.text,
        "char_start": c.start,
        "char_end": c.end,
        "tokens": c.tokens,
        "overlap_tokens": c.overlap_tokens,
    } for c in chunks]