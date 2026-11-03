"""把已上传的模板渲染进统一请求。

时机很重要：必须在调用 ``gateway.complete()`` / ``gateway.stream()`` **之前**完成。
``build_sse_response`` 会先预取首事件才返回 ``StreamingResponse``，如果渲染放到
适配层内部，失败时响应头已经发出，状态码再也改不回来。

注入发生在 ``GenerateRequest`` 这一层（而不是 translator），因此结构化输出的
修复重试（以 ``request.messages`` 为基底重建）会自动带上模板内容。

**互斥约定**：模板与请求里已有的 system 内容（``instructions``、``messages`` /
``input`` 中的 ``role=system``）不允许同时出现，直接 400。不同厂商对「多条
system 消息」的处理不一致（按序拼接 / 只取第一条 / 直接 400），为避免这种
不可移植性，网关要求 system 提示只能有一个来源：要么自己传，要么用模板。
"""

from __future__ import annotations

import logging
from typing import Any

from src.schemas import ChatMessage, GenerateRequest
from src.services.llm_adaptor.errors import AuthenticationError, InvalidRequestError
from src.services.prompts.errors import PromptRenderError, PromptStoreUnavailable
from src.services.prompts.service import PromptService, RenderedPrompt

logger = logging.getLogger(__name__)


def prompt_headers(info: RenderedPrompt | None) -> dict[str, str]:
    """响应头里只放可公开的元信息，正文与变量值一律不出现在头/日志中。"""
    if info is None:
        return {}
    return {
        "X-Gateway-Prompt-Id": info.prompt_id,
        "X-Gateway-Prompt-Version": str(info.version),
        "X-Gateway-Prompt-Bytes": str(info.rendered_bytes),
    }


def _combine(base: str, text: str, position: str) -> str:
    base = base or ""
    if not base:
        return text
    if not text:
        return base
    return f"{text}\n\n{base}" if position == "prepend" else f"{base}\n\n{text}"


def _inject_user_message(
    message: ChatMessage, text: str, position: str
) -> ChatMessage:
    content = message.content
    if not isinstance(content, str):
        raise PromptRenderError(
            "target=user 要求 user 消息的内容是纯文本",
            details={"role": message.role},
        )
    return message.model_copy(update={"content": _combine(content, text, position)})


def _existing_system_source(payload: GenerateRequest) -> str | None:
    """返回请求里已有的 system 内容来源；没有则返回 ``None``。

    只认「有实际内容」的来源：空字符串 / ``content=None`` 的 system 消息不算冲突。
    """
    if payload.instructions and payload.instructions.strip():
        return "instructions"
    for item in payload.messages or []:
        if item.role == "system" and (item.content or "").strip():
            return "messages"
    if isinstance(payload.input, list):
        for item in payload.input:
            if not isinstance(item, dict) or item.get("role") != "system":
                continue
            content = item.get("content") or item.get("text")
            if isinstance(content, str) and content.strip():
                return "input"
    return None


async def render_into_payload(
    payload: GenerateRequest,
    user_id: str | None,
    service: PromptService | None,
) -> tuple[GenerateRequest, RenderedPrompt | None]:
    """按 ``payload.prompt`` 渲染并写回 payload；未带 prompt 时原样返回。"""
    injection = payload.prompt
    if injection is None:
        return payload, None

    # 先做无依赖的契约校验：模板与已有 system 内容互斥（客户端问题，不需要查库）
    conflict_source = _existing_system_source(payload)
    if conflict_source is not None:
        raise InvalidRequestError(
            "prompt 与请求中已有的 system 内容互斥："
            f"检测到 {conflict_source}。网关保证上游只收到一条 system 消息，"
            "请把原有 system 提示并入模板正文后重试。",
            code="prompt_instructions_conflict",
            details={"conflict_source": conflict_source},
        )

    if service is None:
        raise PromptStoreUnavailable(
            "提示词模板存储未配置（DATABASE_URL 为空），无法引用模板"
        )
    if not user_id:
        raise AuthenticationError(
            "使用提示词模板需要 X-User-Id 请求头", code="missing_user_id"
        )

    text, info = await service.resolve(user_id, injection)
    # prompt 已被消费：清空后再交给 gateway，避免任何路径把它透传给上游
    stripped = payload.model_copy(update={"prompt": None})
    position = injection.position

    if injection.target in ("system", "instructions"):
        merged = _combine(stripped.instructions or "", text, position)
        result = stripped.model_copy(update={"instructions": merged})
    elif stripped.messages:
        index = next(
            (i for i, item in enumerate(stripped.messages) if item.role == "user"),
            None,
        )
        if index is None:
            raise PromptRenderError("target=user 但请求中没有 user 消息")
        messages = list(stripped.messages)
        messages[index] = _inject_user_message(messages[index], text, position)
        result = stripped.model_copy(update={"messages": messages})
    elif isinstance(stripped.input, str):
        result = stripped.model_copy(
            update={"input": _combine(stripped.input, text, position)}
        )
    elif isinstance(stripped.input, list):
        items: list[Any] = [
            dict(item) if isinstance(item, dict) else item
            for item in stripped.input
        ]
        for item in items:
            if not isinstance(item, dict):
                continue
            is_user = item.get("role") == "user"
            is_text = item.get("type") == "input_text"
            if not (is_user or is_text):
                continue
            field = "content" if is_user else "text"
            value = item.get(field)
            if not isinstance(value, str):
                raise PromptRenderError(
                    "target=user 要求 user 输入是纯文本", details={"field": field}
                )
            item[field] = _combine(value, text, position)
            result = stripped.model_copy(update={"input": items})
            break
        else:
            raise PromptRenderError("target=user 但 input 中没有 user 消息")
    else:
        raise PromptRenderError("target=user 但请求中没有可注入的内容")

    logger.info(
        "prompt injected user=%s id=%s version=%s target=%s bytes=%d tokens≈%d",
        user_id,
        info.prompt_id,
        info.version,
        info.target,
        info.rendered_bytes,
        info.token_estimate,
    )
    return result, info


__all__ = ["prompt_headers", "render_into_payload"]
