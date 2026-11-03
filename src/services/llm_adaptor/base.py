"""适配器契约：能力声明 + 部署描述 + 适配器协议。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Protocol, runtime_checkable

from src.services.llm_adaptor.types import LLMRequest, LLMResult, StreamEvent


class UpstreamMode(str, Enum):
    """上游厂商实际暴露的协议。由部署配置决定，不是客户端请求决定的。"""

    CHAT = "chat"
    RESPONSES = "responses"


class Cap(IntFlag):
    """部署能力位。降级前据此裁剪参数，避免 backup 因不支持的参数而 400。

    ``JSON_SCHEMA`` 表示“能吃 ``response_format``”（含厂商自定义的 json_object /
    json_schema），不保证是语法级约束；``STRICT_JSON`` 特指支持 OpenAI
    严格 ``json_schema`` 语义（约束解码）的部署。二者分开声明，才能在不支持
    严格模式的模型上退化到“提示 + 校验 + 修复”。
    """

    NONE = 0
    TOOLS = 1
    VISION = 2
    JSON_SCHEMA = 4
    STREAM_USAGE = 8
    STRICT_JSON = 16


CAP_NAMES: dict[str, Cap] = {
    "tools": Cap.TOOLS,
    "vision": Cap.VISION,
    "json_schema": Cap.JSON_SCHEMA,
    "stream_usage": Cap.STREAM_USAGE,
    "strict_json": Cap.STRICT_JSON,
}

# 默认能力保持保守：严格 json_schema 需要显式声明（不是所有厂商都支持）
DEFAULT_CAPS = Cap.TOOLS | Cap.JSON_SCHEMA | Cap.STREAM_USAGE


@dataclass(frozen=True)
class Deployment:
    """一个可调用的上游部署（逻辑模型 → 厂商 + base_url + 凭据）。"""

    model: str
    provider: str
    upstream_model: str
    mode: UpstreamMode
    base_url: str
    api_key: str
    caps: Cap = DEFAULT_CAPS
    timeout: float | None = None
    # 输出长度上限的字段名：多数厂商为 max_tokens，OpenAI 推理模型为 max_completion_tokens
    max_tokens_field: str = "max_tokens"
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.mode.value}:{self.upstream_model}"


@runtime_checkable
class LLMAdapter(Protocol):
    """适配器只需实现两个方法：非流式与流式。"""

    mode: UpstreamMode

    async def complete(self, deployment: Deployment, request: LLMRequest) -> LLMResult: ...

    def stream(self, deployment: Deployment, request: LLMRequest) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
