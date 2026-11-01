# 核心链路

该文件夹是RAG的核心链路模块，包含: 
> 加载数据(load_file) -> 切分(chunk) -> 向量化(embedding) -> 索引构建/存储(vector_store) -> 召回/检索(retrive) -> 融合与重排 -> 生成