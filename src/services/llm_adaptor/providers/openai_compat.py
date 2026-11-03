"""OpenAI Chat Completions 兼容适配器。

DeepSeek、Kimi、Qwen、Ollama、vLLM 等均可复用本适配器，差异仅在 base_url。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.services.llm_adaptor.base import Cap, Deployment, UpstreamMode
from src.services.llm_adaptor.errors import (
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.services.llm_adaptor.providers._client import UpstreamClient, iter_sse, loads
from src.services.llm_adaptor.types import LLMRequest, LLMResult, StreamEvent, Usage

logger = logging.getLogger(__name__)

_CHAT_PATH = "/chat/completions"


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
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(item.get("text") or "")
        return "".join(parts)
    return str(content)


class OpenAICompatAdapter:
    mode = UpstreamMode.CHAT

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
        payload: dict[str, Any] = {
            "model": deployment.upstream_model,
            "messages": [message.to_payload() for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload[deployment.max_tokens_field] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop

        # 按目标部署的能力裁剪参数，避免降级后因不支持的字段被 400
        if request.tools and deployment.caps & Cap.TOOLS:
            payload["tools"] = request.tools
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice

        response_format = _upstream_response_format(request.response_format, deployment)
        if response_format is not None:
            payload["response_format"] = response_format

        # 厂商私有参数透传（仅同类厂商之间降级时安全）
        payload.update(request.extra)
        payload["model"] = deployment.upstream_model
        payload["messages"] = [message.to_payload() for message in request.messages]

        if stream:
            payload["stream"] = True
            if deployment.caps & Cap.STREAM_USAGE:
                payload["stream_options"] = {"include_usage": True}
        return payload

    # -- 非流式 -------------------------------------------------------------
    async def complete(self, deployment: Deployment, request: LLMRequest) -> LLMResult:
        payload = self._payload(deployment, request, stream=False)
        data = await self._client.post_json(deployment, _CHAT_PATH, payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderServerError("上游返回的 choices 为空")
        choice = choices[0]
        message = choice.get("message") or {}
        return LLMResult(
            text=_text_of(message.get("content")),
            finish_reason=choice.get("finish_reason"),
            usage=Usage.from_upstream(data.get("usage")),
            tool_calls=message.get("tool_calls") or None,
            upstream_id=data.get("id"),
            upstream_model=data.get("model"),
        )

    # -- 流式 ---------------------------------------------------------------
    async def stream(
        self, deployment: Deployment, request: LLMRequest
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(deployment, request, stream=True)
        usage: Usage | None = None
        finish_reason: str | None = None

        async with self._client.open_stream(deployment, _CHAT_PATH, payload) as response:
            try:
                async for _, data in iter_sse(response):
                    if data == "[DONE]":
                        break
                    chunk = loads(data)
                    if chunk is None:
                        continue
                    if chunk.get("error"):
                        raise _error_from_chunk(chunk["error"])

                    chunk_usage = Usage.from_upstream(chunk.get("usage"))
                    if chunk_usage:
                        usage = chunk_usage

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        text = delta.get("content")
                        if text:
                            yield StreamEvent(type="delta", text=_text_of(text))
                        if delta.get("tool_calls"):
                            yield StreamEvent(type="delta", tool_calls=delta["tool_calls"])
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"流式读取超时：{exc}") from exc
            except httpx.HTTPError as exc:
                raise ProviderConnectionError(f"流式读取中断：{exc}") from exc

        yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)


def _upstream_response_format(
    response_format: dict[str, Any] | None, deployment: Deployment
) -> dict[str, Any] | None:
    """按部署能力下发 ``response_format``。

    - 部署声明了 ``strict_json``：原样下发（含 strict 语义）；
    - 只声明了 ``json_schema``：退化为 ``json_object``，**不假装** strict；
    - 都不支持：不下发，改由提示词 + 网关校验兜底（调用方从响应头能看到）。
    """
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        if deployment.caps & Cap.STRICT_JSON:
            return response_format
        if deployment.caps & Cap.JSON_SCHEMA:
            logger.debug(
                "模型 %s 不支持 strict json_schema，退化为 json_object", deployment.model
            )
            return {"type": "json_object"}
        return None
    if deployment.caps & Cap.JSON_SCHEMA:
        return response_format
    return None


def _error_from_chunk(error: Any) -> ProviderError:
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("msg") or error)
        code = str(error.get("code") or error.get("type") or "unknown")
    else:
        message = str(error)
        code = "unknown"
    return ProviderServerError(f"上游流式响应返回错误（{code}）：{message}")
