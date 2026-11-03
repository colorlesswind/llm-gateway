"""适配层内部统一数据结构（IR）。

客户端协议（chat / responses）与上游协议之间的差异，全部收敛在
:mod:`src.services.llm_adaptor.translator` 中；适配器只面向这里的 IR。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.services.llm_adaptor.errors import GatewayError

StreamEventType = Literal["delta", "done", "error"]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_upstream(cls, data: dict[str, Any] | None) -> "Usage | None":
        """兼容 chat（prompt/completion）与 responses（input/output）两种字段命名。"""
        if not data:
            return None
        prompt = data.get("prompt_tokens") or data.get("input_tokens") or 0
        completion = data.get("completion_tokens") or data.get("output_tokens") or 0
        total = data.get("total_tokens") or (prompt + completion)
        if not (prompt or completion or total):
            return None
        return cls(
            prompt_tokens=int(prompt),
            completion_tokens=int(completion),
            total_tokens=int(total),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def merge(self, other: Usage | None) -> None:
        """累加多轮（修复重试）产生的 token 用量。"""
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class Message:
    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """转换为 OpenAI chat 消息格式（厂商私有字段原样透传）。"""
        payload: dict[str, Any] = dict(self.extra)
        payload["role"] = self.role
        if self.content is not None:
            payload["content"] = self.content
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass
class LLMRequest:
    """与具体厂商无关的调用参数。"""

    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    # 厂商私有参数透传（仅在同类厂商之间降级时安全）
    extra: dict[str, Any] = field(default_factory=dict)
    # 是否需要上游返回流式 usage
    stream_usage: bool = True


@dataclass
class LLMResult:
    text: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_calls: list[dict[str, Any]] | None = None
    upstream_id: str | None = None
    upstream_model: str | None = None


@dataclass
class StreamEvent:
    type: StreamEventType
    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    error: GatewayError | None = None
    # 以下字段由降级层在 done 事件上补齐
    provider: str | None = None
    model: str | None = None
    upstream_model: str | None = None
    fallback: bool = False


@dataclass
class StructuredOutcome:
    """结构化输出的最终结果（供出口翻译与诊断使用）。"""

    schema_name: str
    schema_id: str | None
    mode: Literal["strict", "json_object", "prompt_only"]
    valid: bool
    attempts: int
    value: Any = None
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_name": self.schema_name,
            "mode": self.mode,
            "valid": self.valid,
            "attempts": self.attempts,
        }
        if self.schema_id:
            data["schema_id"] = self.schema_id
        if include_value and self.value is not None:
            data["parsed"] = self.value
        if self.violations:
            data["violations"] = self.violations
        return data


@dataclass
class GatewayResult:
    """一次成功调用的完整上下文（供出口翻译使用）。"""

    result: LLMResult
    deployment: Any  # Deployment（避免 base ↔ types 循环导入）
    model: str  # 请求方指定的逻辑模型名
    mode: Any  # UpstreamMode
    fallback: bool = False
    # 结构化输出上下文（未启用时为 None）；attempts 含首次调用
    structured: StructuredOutcome | None = None
    attempts: int = 1
