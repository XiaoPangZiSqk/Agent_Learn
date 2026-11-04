# 核心链路

该文件夹是RAG的核心链路模块，包含: 
> 加载数据(load_file) -> 切分(chunk) -> 向量化(embedding) -> 索引构建/存储(vector_store) -> 召回/检索(retrive) -> 融合与重排 -> 生成

```
构建期（离线脚本）
  LoadData → MarkdownChunker → to_records(records)
                                   ├── store.add_records(records)   → Chroma（向量通道）
                                   └── bm25.build(records)          → data/bm25_index/（关键词通道）

查询期（服务）
  bm25.load()          ← 只读本地缓存，不访问 Chroma 语料
  store.query(...)     ← 向量通道
  rrf(两路结果)         ← 融合
```
