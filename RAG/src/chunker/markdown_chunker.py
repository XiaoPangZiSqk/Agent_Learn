"""
Markdown 切分器
"""
from dataclasses import dataclass
from core_line.chunk import chunk_tool


@dataclass
class Chunk:
    core: str                                   # 本体文本
    heading_path: tuple[str, ...]
    start: int                                  # 在源文档中的字符偏移
    end: int
    block_types: tuple[str, ...]
    prefix: str = ""                            # 重叠区（上一块句尾/标题行/代码末行）
    overlap_tokens: int = 0
    index: int = -1
    fragment: bool = False                      # 碎块标记：合并后仍 < FRAGMENT_TOKENS，检索侧降权

    @property
    def text(self) -> str:
        return (self.prefix + "\n\n" + self.core) if self.prefix else self.core

    @property
    def tokens(self) -> int:
        return chunk_tool.predic_token(self.text)


# ============================================================
# 切分器
# ============================================================
def _fence_wrap(text: str) -> tuple[str, str]:
    """识别已包裹围栏的代码块，返回 (头, 尾)，用于超长时按行重包。"""
    first, sep, rest = text.partition("\n")
    m = chunk_tool._FENCE.match(first)
    if not m or not sep:
        return None
    marker = m.group(2)[0] * 3
    last = rest.split("\n")[-1].strip()
    tail = last if chunk_tool._FENCE_LINE.match(last) else marker
    return first, tail


class MarkdownChunker:
    """边界感知的 Markdown 切分器。

    target   : 目标块大小（软预算）
    max_size : 硬上限，超过必须在句子/行级切开
    min_size : 小于该值的块尝试并入相邻块（前后均可，跨根标题允许）
    overlap  : 相邻块重叠预算（句尾吸附句子边界；标题/代码块取标题行或代码末行）

    adaptive : 按文档形态自适应参数（代码/围栏密集文档自动放大 target 装箱预算）；
               显式传入 target/max_size/min_size 任一参数时自适应自动关闭
    """

    def __init__(self, target: int | None = None, max_size: int | None = None,
                 min_size: int | None = None, overlap: int = chunk_tool.OVERLAP,
                 adaptive: bool = True):
        self._base_target = target if target is not None else chunk_tool.CHUNK_SIZE
        self._base_max = max_size if max_size is not None else chunk_tool.MAX_CHUNK_SIZE
        self._base_min = min_size if min_size is not None else chunk_tool.MIN_CHUNK_SIZE
        assert 0 < self._base_target <= self._base_max, "target 必须 <= max_size"
        self.overlap = overlap
        # 显式定制参数表示调用方有自己的预算，不做形态自适应
        self.adaptive = adaptive and (target is None and max_size is None and min_size is None)
        self.target = self._base_target
        self.max_size = self._base_max
        self.min_size = self._base_min

    # ---------- 文档形态自适应 ----------
    def _apply_doc_profile(self, md: str, blocks: list[chunk_tool.Block]) -> None:
        """按围栏/表格密度选择生效参数；纯散文保持默认。

        代码密集文档（围栏+表格字符占比 >= CODE_DENSE_THRESHOLD）：
            target 400 -> 600，让相邻代码块装进同块；min_size 同步下调。
        每次 chunk() 调用从基准参数重算，不会污染后续文档。
        """
        target, max_size, min_size = self._base_target, self._base_max, self._base_min
        if self.adaptive and self._is_code_dense(md, blocks):
            target = chunk_tool.CODE_DENSE_TARGET
            max_size = chunk_tool.CODE_DENSE_MAX
            min_size = min(chunk_tool.CODE_DENSE_MIN, min_size)
        self.target, self.max_size, self.min_size = target, max_size, min_size

    @staticmethod
    def _is_code_dense(md: str, blocks: list[chunk_tool.Block]) -> bool:
        """围栏/表格字符占比检测（frontmatter/hr 不计）。"""
        total = max(len(md), 1)
        dense = sum(len(b.content) for b in blocks if b.type in ("code", "table"))
        return dense / total >= chunk_tool.CODE_DENSE_THRESHOLD

    # ---------- 主入口 ----------
    def chunk(self, md: str) -> list[Chunk]:
        blocks = chunk_tool.parse_markdown(md)
        self._apply_doc_profile(md, blocks)
        out: list[Chunk] = []
        for path, section in self._split_sections(blocks):
            units = self._pack_blocks(section, path) 	#section:list[Block] , path:(str, ...)
            out.extend(self._apply_overlap(units))
        out = self._merge_small(out)
        self._mark_fragments(out)
        for i, c in enumerate(out):
            c.index = i
        return out

    # ---------- 标题分节：每个标题开启新节 ----------
    @staticmethod
    def _split_sections(blocks: list[chunk_tool.Block]) -> list[tuple[tuple[str, ...], list[chunk_tool.Block]]]:
        """
        找出某一标题路径下的所有block，封装成(path, Blocks)加入到list中。
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
    def _pack_blocks(self, blocks: list[chunk_tool.Block], path) -> list[Chunk]:
        """
        将某一标题路径下的所有blocks组装成Chunk, 可能有多个。

        param:
            blocks:某一标题路径下的所有blocks
            path: 某一标题路径
        """
        chunks: list[Chunk] = []    # 组装的Chunk
        cur: list[chunk_tool.Block] = []	# 组合成chunk所需的block块
        cur_tokens = 0

        def flush() -> None:
            """将多个block合成 一个chunk"""

            # nonlocal 是 Python 中用于嵌套函数（闭包）场景的关键字，作用是让内层函数能够修改外层函数（非全局）作用域中的变量。
            nonlocal cur, cur_tokens    # 声明使用外层的 cur,cur_tokens
            if not cur:
                return
            items = [(b.content, b.start, b.end) for b in cur]
            chunks.append(self._make_unit(items, path,
                                          tuple(b.type for b in cur)))
            cur, cur_tokens = [], 0

        for b in blocks:
            tk = chunk_tool.predic_token(b.content)
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
    def _split_atomic(self, b: chunk_tool.Block, path) -> list[Chunk]:
        if b.type == "code":
            return self._split_code(b, path)
        if b.type == "table":
            return self._split_table(b, path)
        return self._split_lines(b, path)

    def _split_code(self, b: chunk_tool.Block, path) -> list[Chunk]:
        lines = b.content.split("\n")
        if len(lines) < 3:
            return self._split_lines(b, path)
        head, tail = lines[0], lines[-1]
        if not chunk_tool._FENCE_LINE.match(tail):
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

    def _split_table(self, b: chunk_tool.Block, path) -> list[Chunk]:
        lines = b.content.split("\n")
        if len(lines) < 3:
            return self._split_lines(b, path)
        head, delim, rows = lines[0], lines[1], lines[2:]
        if chunk_tool.predic_token(head + delim) > self.max_size:
            return self._split_lines(b, path)

        head_tk = chunk_tool.predic_token(head + "\n" + delim)
        groups, cur, cur_t = [], [], head_tk
        for r in rows:                          # 行级装箱
            tk = chunk_tool.predic_token(r)
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

    def _split_lines(self, b: chunk_tool.Block, path) -> list[Chunk]:
        items, cursor = [], 0
        for ln in b.content.split("\n"):
            idx = b.content.find(ln, cursor) if ln else cursor
            if idx < 0:
                idx = cursor
            items.append((ln, b.start + idx, b.start + idx + len(ln))) #(原始字符，起始位置，结束位置)
            cursor = idx + len(ln) + 1
        return self._pack_lines(items, path, b.type)

    def _split_long(self, b: chunk_tool.Block, path) -> list[Chunk]:
        sents = chunk_tool.split_sentences(b.content)    # 将该block的content按句子切分。
        if not sents:
            return self._split_lines(b, path)
        joiner = "" if chunk_tool._HAS_CJK.search(b.content) else " "
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
            tk = chunk_tool.predic_token(text)
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
            tk = chunk_tool.predic_token(text)
            if cur and cur_t + tk > self.target:
                flush()
            if not cur:
                span[0] = s
            cur.append(text)
            span[1] = e
            cur_t += tk
        flush()
        return chunks

    # ---------- 重叠：句尾吸附句子边界，标题/代码块取标题行或代码末行 ----------
    def _apply_overlap(self, units: list[Chunk]) -> list[Chunk]:
        prev: Chunk|None = None
        for u in units:
            if prev is not None and self.overlap > 0:
                tail = self._overlap_prefix(prev)
                if tail:
                    u.prefix = tail
                    u.overlap_tokens = chunk_tool.predic_token(tail)
            prev = u
        return units

    def _overlap_prefix(self, prev: Chunk) -> str:
        """重叠前缀来源（按上一块的块类型降级，代码/围栏密集文档也能获得上下文衔接）：

        1. 含 paragraph/list/quote -> 尾部句子（吸附句子边界，原有行为）；
        2. 含 heading -> 标题行作为上下文锚点（取最贴近当前块的那行）；
        3. 含 code -> 代码末行（按行截取，不按句切，避免破坏代码语义）。
        """
        types = set(prev.block_types)
        if types & {"paragraph", "list", "quote"}:
            return self._tail(prev.core, self.overlap)
        if "heading" in types:
            heading_lines = [ln.strip() for ln in prev.core.split("\n") if chunk_tool._HEADING.match(ln)]
            if heading_lines:
                return heading_lines[-1]
        if "code" in types:
            return self._code_tail(prev.core, self.overlap)
        return ""

    def _code_tail(self, text: str, budget: int) -> str:
        """取代码块末尾若干行（行级，不做句子切分；剔除围栏标记行）。"""
        lines = [ln for ln in text.split("\n")
                 if ln.strip() and not chunk_tool._FENCE.match(ln)]
        picked: list[str] = []
        total = 0
        for ln in reversed(lines):
            tk = chunk_tool.predic_token(ln)
            if picked and total + tk > budget:
                break
            picked.append(ln)
            total += tk
        picked.reverse()
        return "\n".join(picked)

    def _tail(self, text: str, budget: int) -> str:
        sents = chunk_tool.split_sentences(text)
        picked: list[str] = []
        total = 0
        for s in reversed(sents):
            tk = chunk_tool.predic_token(s)
            if picked and total + tk > budget:
                break
            picked.append(s)
            total += tk
        picked.reverse()
        joiner = "" if chunk_tool._HAS_CJK.search(text) else " "
        return joiner.join(picked)

    # ---------- 小块合并：避免碎片（放宽版） ----------
    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """小块合并（放宽版，解决 9~30 token 碎块）：

        - 小于 min_size 的块先尝试并入前块：允许跨根标题，合并后 <= max_size；
        - 前块放不下则并入后块（小块在前，保持原文顺序）；
        - 两者都放不下则保持独立，由 _mark_fragments() 打碎块标记。
        """
        # 前向合并：小块并入前块（跨根标题允许）
        merged: list[Chunk] = []
        for c in chunks:
            if merged and self._can_merge(merged[-1], c) \
                    and chunk_tool.predic_token(c.core) < self.min_size:
                self._absorb(merged[-1], c)
            else:
                merged.append(c)

        # 后向合并：仍过小的块并入后块
        out: list[Chunk] = []
        i, n = 0, len(merged)
        while i < n:
            c = merged[i]
            if (chunk_tool.predic_token(c.core) < self.min_size
                    and i + 1 < n and self._can_merge(c, merged[i + 1])):
                self._absorb(merged[i + 1], c, prepend=True)
                i += 1                      # c 已并入 merged[i+1]，跳过
                continue
            out.append(c)
            i += 1
        return out

    def _can_merge(self, a: Chunk, b: Chunk) -> bool:
        return chunk_tool.predic_token(a.core + "\n\n" + b.core) <= self.max_size

    @staticmethod
    def _absorb(host: Chunk, other: Chunk, prepend: bool = False) -> None:
        """将 other 并入 host：prepend=True 时 other 在前（后向合并）。"""
        if prepend:
            host.core = other.core + "\n\n" + host.core
            host.start = min(host.start, other.start)
            host.block_types = other.block_types + host.block_types
        else:
            host.core = host.core + "\n\n" + other.core
            host.end = max(host.end, other.end)
            host.block_types = host.block_types + other.block_types
        host.overlap_tokens = 0
        host.prefix = ""

    # ---------- 碎块兜底标注 ----------
    def _mark_fragments(self, chunks: list[Chunk]) -> None:
        """合并后仍 < FRAGMENT_TOKENS 的块打碎块标记，检索侧降权（见 HybridRetriever）。"""
        for c in chunks:
            c.fragment = chunk_tool.predic_token(c.core) < chunk_tool.FRAGMENT_TOKENS


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
        "fragment": c.fragment,
    } for c in chunks]