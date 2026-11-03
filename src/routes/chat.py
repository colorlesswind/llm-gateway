"""Chat Completions 兼容入口。

推荐使用统一的 ``/{api_version}/generate``；本路由保留是为了让现成的
OpenAI SDK 客户端可以直接指向网关。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Path, status
from fastapi.responses import JSONResponse

from src.routes._sse import build_sse_response
from src.routes.deps import Gateway, OptionalPrompts, OptionalUserId
from src.schemas import ChatCompletionRequest, ChatCompletionResponse
from src.services.llm_adaptor.translator import (
    build_generate_request,
    chat_chunk,
    gateway_headers,
    to_chat_completion,
)
from src.services.prompts import prompt_headers, render_into_payload

router = APIRouter(tags=["chat"])


@router.post(
    "/{api_version}/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Chat Completions（兼容入口）",
    description="以 chat completions 形状调用下游模型；建议改用 /generate。",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "请求参数或调用方式不受支持"},
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "网关并发已达上限，请按 Retry-After 稍后重试"
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "全部候选模型均调用失败"},
    },
)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    gateway: Gateway,
    user_id: OptionalUserId,
    prompts: OptionalPrompts,
    api_version: str = Path(..., description="API 版本，例如 v1"),
):
    unified = build_generate_request(payload)
    # 未带 prompt 时 user_id 仍可为空，保持对旧客户端的兼容
    unified, prompt_usage = await render_into_payload(unified, user_id, prompts)
    extra_headers = prompt_headers(prompt_usage)

    if payload.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = payload.model
        return await build_sse_response(
            gateway.stream(unified),
            lambda event: chat_chunk(event, chunk_id=chunk_id, model=model),
            terminator="data: [DONE]\n\n",
            headers=extra_headers,
        )

    result = await gateway.complete(unified)
    return JSONResponse(
        content=to_chat_completion(result).model_dump(exclude_none=True),
        headers={**gateway_headers(result), **extra_headers},
    )
