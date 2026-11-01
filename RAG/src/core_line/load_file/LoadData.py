"""
加载Markdown、PDF、Word等文档。

内存策略：
1. ``load_file`` 返回生成器，同一时刻内存中只有一份文档内容；
2. 目录遍历使用惰性递归（边扫描边产出），不预先收集全部路径；
3. 单文件超过 ``max_file_size`` 直接跳过，避免超大文件撑爆内存；
4. 单个文件读取/解析失败只记录日志，不中断整批加载。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# 单文件大小上限：50MB，超过则跳过
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024

# 文本解码编码候选，按顺序尝试
TEXT_ENCODINGS = ("utf-8-sig", "gb18030")

# 后缀 -> 处理方法名
HANDLERS = {
    ".md": "_load_markdown",
    ".markdown": "_load_markdown",
    ".pdf": "_load_pdf",
    ".docx": "_load_docx",
}


@dataclass(slots=True)
class Document:
    """一份文档：正文 + 元数据。"""

    page_content: str
    metadata: dict = field(default_factory=dict)


class LoadData:
    def __init__(self, max_file_size: int = DEFAULT_MAX_FILE_SIZE):
        self.max_file_size = max_file_size

    def load_file(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
    ) -> Iterator[Document]:
        """批量载入目录下的文档，或载入单个文件。

        以生成器方式逐个产出 ``Document``，调用方遍历结束后内存即可释放，
        不会因为目录中文档多而累积占用。

        :param path: 文件或目录路径
        :param recursive: 目录时是否递归子目录
        """
        root = Path(path).expanduser()

        # 若路径指定一个文件，则载入单文件
        if root.is_file():
            yield from self._load_single(root)
            return

        if not root.is_dir():
            raise FileNotFoundError(f"路径不存在或不是文件/目录: {root}")

        pattern = "**/*" if recursive else "*"
        for file_path in root.glob(pattern):
            if not file_path.is_file() or self._is_hidden(file_path, root):
                continue
            yield from self._load_single(file_path)

    def _load_single(self, file_path: Path) -> Iterator[Document]:
        """按后缀分发到具体解析方法，异常不向上冒泡。"""
        handler_name = HANDLERS.get(file_path.suffix.lower())
        if handler_name is None:
            logger.debug("跳过不支持的文件类型: %s", file_path)
            return

        try:
            size = file_path.stat().st_size
        except OSError as exc:
            logger.warning("无法读取文件信息，已跳过 %s: %s", file_path, exc)
            return

        if size == 0:
            logger.warning("空文件，已跳过: %s", file_path)
            return
        if size > self.max_file_size:
            logger.warning(
                "文件超过大小上限(%d bytes)，已跳过: %s (%d bytes)",
                self.max_file_size,
                file_path,
                size,
            )
            return

        try:
            # getattr(obj, name) 用字符串来动态获取属性/方法
            yield from getattr(self, handler_name)(file_path)
        except NotImplementedError:
            logger.warning("暂未实现该类文件的解析，已跳过: %s", file_path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("解析文件失败，已跳过 %s: %s", file_path, exc)



    def _load_markdown(self, file_path: Path) -> Iterator[Document]:
        """载入单个 Markdown 文件，整篇作为一个 Document 产出。"""
        text, encoding = self._read_text(file_path)
        if not text.strip():
            logger.warning("Markdown 内容为空，已跳过: %s", file_path)
            return

        metadata = self._build_metadata(file_path)
        metadata["encoding"] = encoding
        yield Document(page_content=text, metadata=metadata)



    def _load_pdf(self, file_path: Path) -> Iterator[Document]:
        raise NotImplementedError



    def _load_docx(self, file_path: Path) -> Iterator[Document]:
        raise NotImplementedError



    @staticmethod
    def _read_text(file_path: Path) -> tuple[str, str]:
        """按候选编码读取文本，统一换行符，返回 (内容, 实际编码)。"""
        for encoding in TEXT_ENCODINGS:
            try:
                with file_path.open("r", encoding=encoding) as fp:
                    return fp.read().replace("\r\n", "\n").replace("\r", "\n"), encoding
            except UnicodeDecodeError:
                logger.debug("按 %s 解码失败，尝试其它编码: %s", encoding, file_path)

        # 所有候选编码都失败时兜底，保证不因个别坏字符丢失整篇内容
        with file_path.open("r", encoding="utf-8", errors="replace") as fp:
            text = fp.read().replace("\r\n", "\n").replace("\r", "\n")
        return text, "utf-8(replace)"

    @staticmethod
    def _build_metadata(file_path: Path) -> dict:
        stat = file_path.stat()     # 返回 os.stat_result
        return {
            "source": str(file_path.resolve()),                 # 源路径
            "file_name": file_path.name,                        # 文件名
            "file_type": file_path.suffix.lower().lstrip("."),  # 文件类型
            "size_bytes": stat.st_size,                         # 文件大小
        }

    @staticmethod
    def _is_hidden(file_path: Path, root: Path) -> bool:
        """跳过隐藏文件/隐藏目录下的文件。"""
        try:
            parts = file_path.relative_to(root).parts
        except ValueError:
            parts = file_path.parts
        return any(part.startswith(".") for part in parts)
