"""
RAG入口文件
"""

from pathlib import Path
from core_line.load_file.load_data import LoadData
from chunker.markdown_chunker import MarkdownChunker, to_records

file_path = Path(__file__).resolve().parents[1]/"data/load_file/结构化切分-demo-结果.md"


md_load = LoadData()

md_document = md_load.load_file(file_path)

md_chunker = MarkdownChunker()

for document in md_document :
    chunks = md_chunker.chunk(document.page_content)
    datas = to_records(chunks, document.metadata["file_name"])
    for data in datas :
        print(f"doc_id:{data['doc_id']}。 路径：{data['heading_path']}\n 内容：{data['text']}\n\n")