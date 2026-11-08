"""
RAG入口文件
"""

import sys
from pathlib import Path

from core_line.config import ConfigError, preflight
from core_line.load_file.load_data import LoadData
from chunker.markdown_chunker import MarkdownChunker, to_records

# 启动预检：必需配置（key / model / base_url）缺失或非法时，
# 立刻报出具体变量名，而不是拖到第一次 API 调用才失败（见 core_line/config.py）
try:
    _config = preflight()
except ConfigError as exc:
    print(f"[配置预检失败] {exc}")
    sys.exit(1)
print(f"[配置] 预检通过 | {_config.summary()}")

file_path = Path(__file__).resolve().parents[1]/"data/load_file/"


md_load = LoadData()


md_document = md_load.load_file(file_path)

# for document in md_document :
#     print(document.page_content)
#     print("".join("=" for _ in range(200)))

md_chunker = MarkdownChunker()

for document in md_document :
    chunks = md_chunker.chunk(document.page_content)
    datas = to_records(chunks, document.metadata["file_name"])
    for data in datas :
        print(f"doc_id:{data['doc_id']}。 路径：{data['heading_path']}\n 内容：{data['text']}\n\n")

    print("".join("=" for _ in range(200)))