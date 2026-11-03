"""网关配置。

配置来源优先级：进程环境变量 > `.env` 文件。

模型注册表使用动态键（`MODEL_<ID>_PROVIDER` 等），pydantic-settings 无法静态
声明，因此这里额外提供 :func:`raw_env`，供 :mod:`src.services.llm_adaptor.registry`
读取原始键值。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ENV_FILE = ".env"


def raw_env(env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """合并 `.env` 与进程环境变量（后者覆盖前者）。"""
    values: dict[str, str] = {}
    path = Path(env_file)
    if path.is_file():
        values.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    values.update(os.environ)
    return values


class Settings(BaseSettings):
    """静态配置项。

    注意：`MODEL_CHAIN` 以字符串形式读取后由 :attr:`model_chain_list` 解析，
    避免 pydantic-settings 对复杂类型强制 JSON 解析。
    """

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- 模型选择 ---------------------------------------------------------
    # 是否允许 responses 方式调用（实验特性，默认关闭，不做 chat→responses 模拟）
    enable_responses_mode: bool = False
    # 降级链，逗号分隔，靠前者优先：deepseek-chat,gpt-4o-mini
    model_chain: str = ""
    # MODEL_CHAIN 缺省时的兼容写法
    main_model: str | None = None
    backup_model: str | None = None

    # --- 超时（秒） -------------------------------------------------------
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    # 流式：首个事件到达的截止时间
    first_token_timeout: float = 20.0
    # 流式：事件间隔超过该值视为上游卡死
    idle_timeout: float = 60.0

    # --- 重试与熔断 -------------------------------------------------------
    # 429 时的最大尝试次数（含首次），指数退避
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    # 连续失败多少次后熔断
    circuit_failure_threshold: int = 3
    # 熔断冷却时间（秒）
    circuit_cooldown: float = 30.0

    # --- 入口限流 ---------------------------------------------------------
    # 同时在飞（含 SSE 流）的请求数上限，超过直接返回 429，不排队。
    # 这是**进程内**计数：多 worker 部署时总上限 = 该值 × worker 数。
    # 设为 0 或负数表示关闭限流。
    max_concurrent_requests: int = 10
    # 429 响应里 Retry-After 的建议等待秒数
    overload_retry_after: float = 1.0

    # --- 结构化输出（JSON Schema 校验 + 有限次修复重试） -------------------
    # 总开关：关闭后请求里的 structured_output 会被拒绝，而不是静默忽略。
    structured_output_enabled: bool = True
    # 校验不通过时，**除首次之外**额外重试的次数（与 retry_max_attempts 独立）。
    # 0 表示只校验不重试；总尝试次数 = 1 + 该值（或请求内的 max_repair_attempts）。
    schema_repair_max_attempts: int = 1
    # 单次修复请求里回灌给上游的错误文本上限，防止上下文爆炸
    schema_repair_max_error_chars: int = 2000
    # 回灌时原样带回的错误输出上限
    schema_repair_max_output_chars: int = 4000
    # 请求自带/注册表加载的 schema 大小与嵌套深度上限，超限直接 400
    schema_max_bytes: int = 16384
    schema_max_depth: int = 12
    # finish_reason=length（输出被截断）时是否仍重试：通常应先让调用方调大
    # max_tokens，同样预算下重试大概率还是截断。
    schema_repair_on_length: bool = False
    # 修复重试是否累加 usage（影响计费口径）。false 时只保留最后一次的 usage。
    schema_sum_usage: bool = True
    # 具名 schema 注册表目录：SCHEMAS_DIR=<id>.json
    schemas_dir: str = "schemas"

    @property
    def model_chain_list(self) -> list[str]:
        """解析后的候选模型优先级链。"""
        items = [item.strip() for item in self.model_chain.split(",") if item.strip()]
        if items:
            return items
        legacy = (self.main_model, self.backup_model)
        return [item.strip() for item in legacy if item and item.strip()]


def get_settings() -> Settings:
    """读取一次配置（不做缓存，便于测试注入）。"""
    return Settings()
