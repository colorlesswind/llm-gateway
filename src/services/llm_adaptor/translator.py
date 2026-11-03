"""协议翻译：统一请求/响应 ↔ 内部 IR ↔ 各对外形态。

这里的转换是网关的“边缘”，适配器只面对 IR。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from src.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    GenerateRequest,
)
from src.schemas import Usage as APIUsage
from src.services.llm_adaptor.base import Deployment, UpstreamMode
from src.services.llm_adaptor.errors import InvalidRequestError
from src.services.llm_adaptor.structured import StructuredSpec
from src.services.llm_adaptor.types import (
    GatewayResult,
    LLMRequest,
    Message,
    StreamEvent,
)
from src.services.llm_adaptor.types import (
    Usage as IRUsage,
)

# GenerateRequest 中由网关处理的字段，不会作为厂商参数透传
GATEWAY_FIELDS = frozenset(
    {
        "model",
        "mode",
        "messages",
        "input",
        "instructions",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "structured_output",
        "prompt",
    }
)


def _content_text(content: Any) -> str | list[Any] | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)


def _messages_from_input(items: list[dict[str, Any]]) -> list[Message]:
    messages: list[Message] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text" and item.get("text"):
            messages.append(Message(role="user", content=str(item["text"])))
            continue
        role = item.get("role")
        if not role:
            continue
        messages.append(
            Message(
                role=str(role),
                content=_content_text(item.get("content")),
                name=item.get("name"),
            )
        )
    return messages


def build_llm_request(
    payload: GenerateRequest,
    *,
    mode: UpstreamMode,
    structured: StructuredSpec | None = None,
) -> LLMRequest:
    """把对外请求转换为与厂商无关的 IR。"""
    messages: list[Message] = []
    # system 内容合并成一条，保持“指令 → schema → 对话”的顺序且不浪费消息位
    system_parts: list[str] = []
    if payload.instructions:
        system_parts.append(payload.instructions)
    if structured is not None and structured.inject_prompt:
        system_parts.append(structured.prompt_text)
    if system_parts:
        messages.append(Message(role="system", content="\n\n".join(system_parts)))

    if payload.messages:
        for item in payload.messages:
            messages.append(
                Message(
                    role=item.role,
                    content=item.content,
                    name=item.name,
                    tool_call_id=item.tool_call_id,
                    tool_calls=item.tool_calls,
                    extra=dict(item.model_extra or {}),
                )
            )
    elif payload.input is not None:
        if isinstance(payload.input, str):
            messages.append(Message(role="user", content=payload.input))
        else:
            messages.extend(_messages_from_input(payload.input))

    if not messages:
        raise InvalidRequestError("messages 与 input 不能同时为空")

    extra = {
        key: value
        for key, value in (payload.model_extra or {}).items()
        if key not in GATEWAY_FIELDS and value is not None
    }

    # structured_output 是权威契约，它的 response_format 覆盖任何直接透传值
    response_format = payload.response_format
    if structured is not None:
        response_format = structured.response_format

    return LLMRequest(
        messages=messages,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        stop=payload.stop,
        tools=payload.tools,
        tool_choice=payload.tool_choice,
        response_format=response_format,
        extra=extra,
    )


def build_generate_request(payload: ChatCompletionRequest) -> GenerateRequest:
    """Chat Completions 兼容入口 → 统一请求。"""
    if payload.stream and payload.structured_output is not None:
        raise InvalidRequestError("structured_output 仅支持非流式调用（stream=false）")
    return GenerateRequest(
        model=payload.model,
        mode="chat",
        messages=payload.messages,
        stream=payload.stream,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        structured_output=payload.structured_output,
        prompt=payload.prompt,
        **(payload.model_extra or {}),
    )


# ---------------------------------------------------------------------------
# 统一出口
# ---------------------------------------------------------------------------
def to_generate_payload(gateway_result: GatewayResult) -> dict[str, Any]:
    result = gateway_result.result
    deployment = gateway_result.deployment
    structured = gateway_result.structured
    return {
        "id": result.upstream_id or f"gen-{uuid.uuid4().hex}",
        "object": "generation",
        "created": int(time.time()),
        "model": gateway_result.model,
        "upstream_model": deployment.upstream_model,
        "provider": deployment.provider,
        "mode": gateway_result.mode.value,
        "fallback": gateway_result.fallback,
        "output": {
            "text": result.text,
            "finish_reason": result.finish_reason,
            "tool_calls": result.tool_calls,
            "json": (
                structured.value
                if structured is not None and structured.valid
                else None
            ),
            "schema_valid": structured.valid if structured is not None else None,
            "attempts": gateway_result.attempts,
        },
        "usage": (result.usage or IRUsage()).to_dict(),
        "structured": structured.to_dict() if structured is not None else None,
    }


def gateway_headers(gateway_result: GatewayResult) -> dict[str, str]:
    headers = {
        "X-Gateway-Model": gateway_result.model,
        "X-Gateway-Provider": gateway_result.deployment.provider,
        "X-Gateway-Upstream-Model": gateway_result.deployment.upstream_model,
        "X-Gateway-Mode": gateway_result.mode.value,
        "X-Gateway-Fallback": "true" if gateway_result.fallback else "false",
    }
    structured = gateway_result.structured
    if structured is not None:
        headers["X-Gateway-Schema-Mode"] = structured.mode
        headers["X-Gateway-Schema-Valid"] = "true" if structured.valid else "false"
        headers["X-Gateway-Schema-Attempts"] = str(structured.attempts)
    return headers


def encode_sse(event: StreamEvent, *, requested_model: str) -> str:
    """统一 SSE 事件：调用方无需区分 chat / responses。"""
    if event.type == "delta":
        data: dict[str, Any] = {"type": "delta", "text": event.text}
        if event.tool_calls:
            data["tool_calls"] = event.tool_calls
    elif event.type == "done":
        data = {
            "type": "done",
            "model": event.model or requested_model,
            "provider": event.provider,
            "upstream_model": event.upstream_model,
            "mode": None,
            "fallback": event.fallback,
            "finish_reason": event.finish_reason,
            "usage": event.usage.to_dict() if event.usage else None,
        }
    else:
        payload = event.error.to_dict()["error"] if event.error else {
            "code": "internal_error",
            "message": "未知错误",
            "retryable": False,
        }
        data = {"type": "error", **payload}
    return f"event: {event.type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Chat Completions 兼容出口
# ---------------------------------------------------------------------------
def to_chat_completion(gateway_result: GatewayResult) -> ChatCompletionResponse:
    result = gateway_result.result
    message = ChatMessage(
        role="assistant",
        content=result.text,
        tool_calls=result.tool_calls,
    )
    usage = result.usage or IRUsage()
    x_gateway: dict[str, Any] | None = None
    if gateway_result.structured is not None:
        x_gateway = {"structured": gateway_result.structured.to_dict()}
    return ChatCompletionResponse(
        id=result.upstream_id or f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=gateway_result.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=result.finish_reason or "stop",
            )
        ],
        usage=APIUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        ),
        x_gateway=x_gateway,
    )


def chat_chunk(event: StreamEvent, *, chunk_id: str, model: str) -> str:
    """把内部事件编码为 OpenAI ``chat.completion.chunk``。"""
    created = int(time.time())
    if event.type == "delta":
        delta: dict[str, Any] = {"content": event.text} if event.text else {}
        if event.tool_calls:
            delta["tool_calls"] = event.tool_calls
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    elif event.type == "done":
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": event.finish_reason or "stop"}
            ],
        }
        if event.usage:
            payload["usage"] = event.usage.to_dict()
    else:
        payload = {
            "error": event.error.to_dict()["error"]
            if event.error
            else {"code": "internal_error", "message": "未知错误"},
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Responses 兼容出口
# ---------------------------------------------------------------------------
def to_response_object(gateway_result: GatewayResult) -> dict[str, Any]:
    result = gateway_result.result
    deployment: Deployment = gateway_result.deployment
    return {
        "id": result.upstream_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "model": gateway_result.model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.text}],
            }
        ],
        "usage": (result.usage or IRUsage()).to_dict(),
        "metadata": {
            "provider": deployment.provider,
            "upstream_model": deployment.upstream_model,
            "mode": gateway_result.mode.value,
            "fallback": gateway_result.fallback,
        },
        "x_gateway": (
            {"structured": gateway_result.structured.to_dict()}
            if gateway_result.structured is not None
            else None
        ),
    }


__all__ = [
    "build_generate_request",
    "build_llm_request",
    "chat_chunk",
    "encode_sse",
    "gateway_headers",
    "to_chat_completion",
    "to_generate_payload",
    "to_response_object",
]
