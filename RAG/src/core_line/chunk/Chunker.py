"""
将加载的文档进行节分：

"""
from dataclasses import dataclass, field
from load_file.LoadData import Document
import tiktoken

CHUNK_SIZE = 1024
OVERLAP = 128


@dataclass
class Chunk :
    """一块数据：快内容 + 元数据"""

    chunk_content: str
    metadata: dict = field(default_factory=dict)


class Chunker :
    def __init__(self, document: Document):
        """
        param:
            document: 由load_file.LoadData.py产出的Document
        """
        self.document = document

    def _predic_token(text: str) -> int:
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

    def fixed_len_chunk(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[Chunk]:
        """固定长度切分
        param:
            chunk_size: 块大小
            overlap: 重叠部分

        核心流程：
            1. 用与嵌入模型一致的分词器把文本编码成 Token 序列 ids      --      这里使用的是tiktoken, openai开源的分词器, 在真正嵌入时会与该分词器不符，需改正
            2. 步长 step = chunk_size - overlap(overlap=0 时 step = chunk_size)
            3. 以 i = 0, step, 2 x step ... 滑动，取 ids[i : i+chunk_size]
            4. 逐块解码回文本，记录块序号与首尾字符偏移（供引用定位）
            5. 丢弃纯空白块；尾部长度不足 chunk_size/3 的块并入前一块
        """

        chunks, start = [], 0
        sum_token = self._predic_token(self.document.page_content)
        while start < sum_token:
            end = start + chunk_size
            mdata = {
                "source":self.document.metadata.get("source", "未知来源"),
                "file_name":self.document.metadata.get("file_name", "default")
            }
            chunk = Chunk(self.document.page_content[start:end], mdata)
            chunks.append(chunk)
            if end >= sum_token:
                break
            start = end - overlap   # 关键：步长 = chunk_size - overlap
        return chunks
    