"""
RAG入口文件：组合离线端构建与在线端查询。

两个子命令：
    build   离线端构建（offline_build.py）：加载 → 切分 → 双通道落库
            （向量 Chroma + 关键词 FTS5）
    query   在线端查询（online_query.py）：双路召回 → RRF 融合 → 交叉重排 → 生成

用法（在 RAG/src 目录下执行）：
    python RagApp.py build [--source DIR] [--rebuild]   # 离线构建索引
    python RagApp.py query "K-means 的原理是什么？"      # 单次查询
    python RagApp.py query                              # 交互式循环，q 退出

各子命令的参数定义在 offline_build.py / online_query.py 内（单一来源），
本入口只做分发，参数原样透传；子命令帮助见：
    python RagApp.py build -h / python RagApp.py query -h

也可以直接调用两个模块：
    python offline_build.py / python online_query.py
"""

from __future__ import annotations

import sys

from offline_build import main as build_main
from online_query import main as query_main

USAGE = __doc__

COMMANDS = {
    "build": build_main,
    "query": query_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE.strip())
        return 0

    command, rest = argv[0], argv[1:]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"[错误] 未知子命令: {command}", file=sys.stderr)
        print(USAGE.strip(), file=sys.stderr)
        return 2

    # 子命令参数原样透传，由各模块的 argparse 解析（help/校验不重复定义）
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())
