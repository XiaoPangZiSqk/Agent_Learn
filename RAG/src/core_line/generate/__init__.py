"""生成层：上下文组装 + 答案生成与自校验。

两个子模块：
    - context_assembler: 将 CrossEncoderReranker 精排结果组装为结构化上下文
    - generator:         基于 DeepSeek API（OpenAI SDK）的生成与自校验，
                          支持 CoT / Self-consistency / Reflexion、
                          引用核验、NLI 一致性、SSE 流式输出
"""

from core_line.generate.context_assembler import (
    AssembledContext,
    ContextAssembler,
    ContextChunk,
)
from core_line.generate.generator import (
    GenerationResult,
    Generator,
    SSEEvent,
)

__all__ = [
    "AssembledContext",
    "ContextAssembler",
    "ContextChunk",
    "GenerationResult",
    "Generator",
    "SSEEvent",
]
