"""
统一配置管理：百炼 embedding / rerank / base_url 三处配置收敛到本模块。

职责：
1. 统一 Key 读取链（BaiLianEmbeder 与 BaiLianRerankScorer 共用同一解析函数）：
       BAILIAN_RERANK_API_KEY -> BAILIAN_EMBEDDING_API_KEY -> DASHSCOPE_API_KEY
   任一变量命中即生效，按此顺序优先。
2. embedding 模型 / rerank 模型 / 百炼 base_url 的集中读取与规范化
   （兼容模式地址自动转换为 dashscope SDK 可用的原生 /api/v1 基址）。
3. 启动预检 preflight()：一次性校验必需配置（key、model、base_url），
   缺失/非法立即报出具体环境变量名，而不是拖到第一次 API 调用才失败。

全部环境变量清单及各自读取方见 RAG/.env.example。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()   # 以本文件位置向上查找 RAG/.env（与各模块原有行为一致）

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- 常量

#: 百炼 API Key 统一读取链（按优先级排列）
BAILIAN_KEY_ENVS: tuple[str, ...] = (
    "BAILIAN_RERANK_API_KEY",
    "BAILIAN_EMBEDDING_API_KEY",
    "DASHSCOPE_API_KEY",
)

DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_DS_MODEL = "deepseek-flash"
DEFAULT_DS_BASE_URL = "https://api.deepseek.com"


class ConfigError(RuntimeError):
    """配置缺失或非法；message 中包含具体环境变量名。"""


# ----------------------------------------------------------------- Key 解析（embedder / rerank 共用）

def _locate_bailian_key(explicit: str | None = None) -> tuple[str, str] | None:
    """返回 (来源, Key)：来源为变量名或 "explicit"；未命中返回 None。"""
    if explicit is not None and str(explicit).strip():
        return "explicit", str(explicit).strip()
    for name in BAILIAN_KEY_ENVS:
        value = os.getenv(name)
        if value and value.strip():
            return name, value.strip()
    return None


def find_bailian_api_key(explicit: str | None = None) -> str | None:
    """按统一顺序解析百炼 API Key，未配置返回 None（不抛错）。

    param:
        explicit: 显式传入的 Key，优先于环境变量（空串视同未传）

    读取顺序：BAILIAN_RERANK_API_KEY -> BAILIAN_EMBEDDING_API_KEY -> DASHSCOPE_API_KEY
    """
    pair = _locate_bailian_key(explicit)
    return pair[1] if pair else None


def resolve_bailian_api_key(explicit: str | None = None) -> str:
    """同 find_bailian_api_key()，但缺失时抛 ConfigError（报出全部候选变量名）。"""
    key = find_bailian_api_key(explicit)
    if key is None:
        raise ConfigError(bailian_key_hint())
    return key


def bailian_key_hint() -> str:
    """Key 缺失时的标准提示语（含具体变量名与读取顺序）。"""
    return (
        "缺少百炼 API Key：请依次尝试设置环境变量 "
        + " -> ".join(BAILIAN_KEY_ENVS)
        + "（任一命中即生效），或显式传入 api_key"
    )


# ----------------------------------------------------------------- 模型 / 端点读取

def get_embedding_model() -> str:
    """嵌入模型名：BAILIAN_EMBEDDING_MODEL，缺省 text-embedding-v4。"""
    return (os.getenv("BAILIAN_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip()


def get_rerank_model() -> str:
    """重排模型名：BAILIAN_RERANK_MODEL，缺省 qwen3-rerank（gte-rerank 已下线勿用）。"""
    return (os.getenv("BAILIAN_RERANK_MODEL") or DEFAULT_RERANK_MODEL).strip()


def get_bailian_base_url() -> str | None:
    """百炼 API 基址（dashscope 原生协议），缺省 None 表示走 SDK 公网默认端点。

    BAILIAN_BASE_URL 允许两种写法：
    - 兼容模式地址（如 https://ws-xxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1）：
      自动转换为原生基址 .../api/v1（dashscope SDK 仅走原生协议路径）；
    - 原生地址（以 /api/v1 结尾）：原样使用。
    """
    raw = (os.getenv("BAILIAN_BASE_URL") or "").strip()
    if not raw:
        return None
    base = raw.rstrip("/")
    marker = base.find("/compatible-mode/")
    if marker != -1:
        head = base[:marker]
        version = base[marker + len("/compatible-mode/"):].split("/")[0] or "v1"
        base = f"{head}/api/{version}"
        logger.debug("BAILIAN_BASE_URL 兼容模式地址已转换为原生基址: %s", base)
    return base


# ----------------------------------------------------------------- 缓存 / 生成模式

_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # RAG/
DEFAULT_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "rag_cache.db"


def get_cache_enabled() -> bool:
    """语义缓存总开关：RAG_CACHE_ENABLED，缺省开启。"""
    return (os.getenv("RAG_CACHE_ENABLED") or "true").strip().lower() not in ("", "0", "false", "no", "off")


def get_cache_path() -> Path:
    """语义缓存库文件路径：RAG_CACHE_PATH，缺省 data/cache/rag_cache.db。"""
    raw = (os.getenv("RAG_CACHE_PATH") or "").strip()
    return Path(raw or DEFAULT_CACHE_PATH).expanduser()


def get_cache_ttl_days() -> int:
    """缓存有效期（天）：RAG_CACHE_TTL_DAYS，缺省 30；<=0 表示永不过期。

    非法值按缺省 30 处理（预检会单独报出该变量名）。
    """
    raw = (os.getenv("RAG_CACHE_TTL_DAYS") or "").strip()
    if not raw:
        return 30
    try:
        return int(raw)
    except ValueError:
        return 30


def get_generator_fast_mode() -> bool:
    """生成快速模式开关：GENERATOR_FAST_MODE，缺省关闭。

    开启后：SC 采样固定 1 次、跳过 NLI 检查、禁用 Reflexion 与润色（延迟最低档）。
    """
    return (os.getenv("GENERATOR_FAST_MODE") or "false").strip().lower() in ("1", "true", "yes", "on")


def get_generator_sc_model() -> str | None:
    """Self-consistency 采样用轻量模型名：GENERATOR_SC_MODEL，缺省 None（与主模型相同）。"""
    return (os.getenv("GENERATOR_SC_MODEL") or "").strip() or None


# ----------------------------------------------------------------- 启动预检

@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """预检通过后的配置快照，便于启动日志输出与问题排查。"""

    bailian_key_source: str          # 实际命中的变量名（或 "explicit" / "未配置"）
    embedding_model: str
    rerank_model: str
    bailian_base_url: str | None     # None = SDK 公网默认端点
    ds_model: str
    ds_base_url: str
    cache_enabled: bool = True       # 语义缓存开关
    fast_mode: bool = False          # 生成快速模式（SC=1、跳过 NLI/Reflexion）
    sc_model: str | None = None      # SC 采样轻量模型（None = 同主模型）

    def summary(self) -> str:
        return (
            f"百炼Key来源={self.bailian_key_source} | embedding={self.embedding_model} | "
            f"rerank={self.rerank_model} | base_url={self.bailian_base_url or 'SDK默认端点'} | "
            f"ds={self.ds_model}@{self.ds_base_url} | "
            f"cache={'on' if self.cache_enabled else 'off'} | "
            f"fast_mode={'on' if self.fast_mode else 'off'} | "
            f"sc_model={self.sc_model or '同主模型'}"
        )


def collect_config_problems(*, need_bailian: bool = True, need_generation: bool = True) -> list[str]:
    """返回全部配置问题清单（每条含具体变量名），空列表表示通过。"""
    problems: list[str] = []

    if need_bailian:
        if find_bailian_api_key() is None:
            problems.append(bailian_key_hint())
        for name in ("BAILIAN_EMBEDDING_MODEL", "BAILIAN_RERANK_MODEL"):
            value = os.getenv(name)
            if value is not None and not value.strip():
                problems.append(f"{name} 已设置但为空：请填写模型名，或删除该行以使用默认值")
        raw_base = (os.getenv("BAILIAN_BASE_URL") or "").strip()
        if raw_base and not raw_base.startswith(("http://", "https://")):
            problems.append(f"BAILIAN_BASE_URL 非法（需以 http:// 或 https:// 开头）：{raw_base!r}")

    if need_generation:
        if not (os.getenv("DS_API_KEY") or "").strip():
            problems.append("缺少 DeepSeek API Key：请设置 DS_API_KEY（生成模块 generate/generator.py 读取）")
        for name in ("DS_MODEL", "DS_BASE_URL"):
            value = os.getenv(name)
            if value is not None and not value.strip():
                problems.append(f"{name} 已设置但为空：请填写值，或删除该行以使用默认值")

    # 性能/缓存配置（可选项，一旦设置即校验）
    ttl_raw = (os.getenv("RAG_CACHE_TTL_DAYS") or "").strip()
    if ttl_raw:
        try:
            int(ttl_raw)
        except ValueError:
            problems.append(f"RAG_CACHE_TTL_DAYS 需为整数（单位天，<=0 表示永不过期），当前为 {ttl_raw!r}")
    fast_raw = (os.getenv("GENERATOR_FAST_MODE") or "").strip()
    if fast_raw and fast_raw.lower() not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        problems.append(f"GENERATOR_FAST_MODE 需为 true/false，当前为 {fast_raw!r}")

    return problems


def preflight(*, need_bailian: bool = True, need_generation: bool = True) -> ResolvedConfig:
    """启动预检：校验必需配置（key、model、base_url），缺失即报出具体变量名。

    param:
        need_bailian:    校验百炼（embedding + rerank + base_url）配置
        need_generation: 校验 DeepSeek 生成配置（DS_API_KEY 等）

    return:
        ResolvedConfig 配置快照

    raise:
        ConfigError: 存在缺失/非法配置，message 一次性列出全部问题（含变量名）
    """
    problems = collect_config_problems(need_bailian=need_bailian, need_generation=need_generation)
    if problems:
        raise ConfigError(
            f"配置预检未通过，共 {len(problems)} 项：\n  - " + "\n  - ".join(problems)
        )

    pair = _locate_bailian_key()
    resolved = ResolvedConfig(
        bailian_key_source=pair[0] if pair else "未配置",
        embedding_model=get_embedding_model(),
        rerank_model=get_rerank_model(),
        bailian_base_url=get_bailian_base_url(),
        ds_model=(os.getenv("DS_MODEL") or DEFAULT_DS_MODEL).strip(),
        ds_base_url=(os.getenv("DS_BASE_URL") or DEFAULT_DS_BASE_URL).strip(),
        cache_enabled=get_cache_enabled(),
        fast_mode=get_generator_fast_mode(),
        sc_model=get_generator_sc_model(),
    )
    logger.info("配置预检通过：%s", resolved.summary())
    return resolved
