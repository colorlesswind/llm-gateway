"""OpenAI Responses API 原生适配器。

只在 ``MODE=responses`` 的部署上使用；不做 chat ↔ responses 的模拟转换。
如需使用，必须显式设置 ``ENABLE_RESPONSES_MODE=true``。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.services.llm_adaptor.base import Cap, Deployment, UpstreamMode
from src.services.llm_adaptor.errors import (
    ProviderConnectionError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.services.llm_adaptor.providers._client import UpstreamClient, iter_sse, loads
from src.services.llm_adaptor.types import (
    LLMRequest,
    LLMResult,
    Message,
    StreamEvent,
    Usage,
)

_RESPONSES_PATH = "/responses"


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or "")
        return "".join(parts)
    return str(content)


def _finish_reason(payload: dict[str, Any]) -> str | None:
    if payload.get("status") == "completed":
        return "stop"
    details = payload.get("incomplete_details") or {}
    reason = details.get("reason")
    if reason == "max_output_tokens":
        return "length"
    if reason == "content_filter":
        return "content_filter"
    return reason


class OpenAIResponsesAdapter:
    mode = UpstreamMode.RESPONSES

    def __init__(
        self,
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = UpstreamClient(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- 请求构造 -----------------------------------------------------------
    def _payload(
        self, deployment: Deployment, request: LLMRequest, *, stream: bool
    ) -> dict[str, Any]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                instructions.append(_text_of(message.content))
                continue
            item = self._input_item(message)
            if item is not None:
                items.append(item)

        payload: dict[str, Any] = {"model": deployment.upstream_model, "input": items}
        if instructions:
            payload["instructions"] = "\n\n".join(part for part in instructions if part)
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        text_format = _text_format(request.response_format, deployment)
        if text_format is not None:
            payload["text"] = {"format": text_format}
        payload.update(request.extra)
        payload["model"] = deployment.upstream_model
        payload["input"] = items
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _input_item(message: Message) -> dict[str, Any] | None:
        if message.role == "tool":
            return {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": _text_of(message.content),
            }
        content_type = "output_text" if message.role == "assistant" else "input_text"
        return {
            "type": "message",
            "role": message.role,
            "content": [{"type": content_type, "text": _text_of(message.content)}],
        }

    # -- 非流式 -------------------------------------------------------------
    async def complete(self, deployment: Deployment, request: LLMRequest) -> LLMResult:
        payload = self._payload(deployment, request, stream=False)
        data = await self._client.post_json(deployment, _RESPONSES_PATH, payload)
        return LLMResult(
            text=self._extract_text(data),
            finish_reason=_finish_reason(data),
            usage=Usage.from_upstream(data.get("usage")),
            upstream_id=data.get("id"),
            upstream_model=data.get("model"),
        )

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for item in payload.get("output") or []:
            if item.get("type") not in (None, "message"):
                continue
            for part in item.get("content") or []:
                if part.get("type") in {"output_text", "text"}:
                    parts.append(part.get("text") or "")
        return "".join(parts)

    # -- 流式 ---------------------------------------------------------------
    async def stream(
        self, deployment: Deployment, request: LLMRequest
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(deployment, request, stream=True)
        usage: Usage | None = None
        finish_reason: str | None = None

        async with self._client.open_stream(deployment, _RESPONSES_PATH, payload) as response:
            try:
                async for event_name, data in iter_sse(response):
                    chunk = loads(data)
                    if chunk is None:
                        continue
                    event_type = chunk.get("type") or event_name

                    if event_type == "response.output_text.delta":
                        delta = chunk.get("delta")
                        if delta:
                            yield StreamEvent(type="delta", text=delta)
                    elif event_type in {"response.completed", "response.incomplete"}:
                        body = chunk.get("response") or {}
                        usage = Usage.from_upstream(body.get("usage")) or usage
                        finish_reason = _finish_reason(body)
                    elif event_type in {"response.failed", "error"}:
                        raise _error_from_event(chunk)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"流式读取超时：{exc}") from exc
            except httpx.HTTPError as exc:
                raise ProviderConnectionError(f"流式读取中断：{exc}") from exc

        yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)


def _text_format(
    response_format: dict[str, Any] | None, deployment: Deployment
) -> dict[str, Any] | None:
    """Responses 协议的结构化输出字段是 ``text.format``，与 chat 不同。

    这里做与 chat 适配器一致的**能力裁剪**：能力不足时降级而不是让上游 400。
    """
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        if deployment.caps & Cap.STRICT_JSON:
            spec = response_format.get("json_schema") or {}
            return {
                "type": "json_schema",
                "name": spec.get("name") or "output",
                "schema": spec.get("schema") or {},
                "strict": True,
            }
        if deployment.caps & Cap.JSON_SCHEMA:
            return {"type": "json_object"}
        return None
    if deployment.caps & Cap.JSON_SCHEMA:
        return response_format
    return None


def _error_from_event(chunk: dict[str, Any]) -> ProviderServerError:
    body = chunk.get("response") or {}
    error = chunk.get("error") or body.get("error") or {}
    if isinstance(error, dict):
        message = str(error.get("message") or error)
        code = str(error.get("code") or error.get("type") or "unknown")
    else:
        message, code = str(error), "unknown"
    return ProviderServerError(f"上游流式响应返回错误（{code}）：{message}")
