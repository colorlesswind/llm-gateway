"""统一生成接口（网关主入口）。

调用方不需要关心底层是 chat completions 还是 responses：在请求里给出
``messages`` 或 ``input`` 即可，``mode`` 缺省为 ``auto``（沿用模型自身的上游协议）。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Path, status
from fastapi.responses import JSONResponse

from src.routes._sse import build_sse_response
from src.routes.deps import Gateway, UserId
from src.schemas import GenerateRequest, GenerateResponse
from src.services.llm_adaptor import InvalidRequestError
from src.services.llm_adaptor.translator import (
    encode_sse,
    gateway_headers,
    to_generate_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.post(
    "/{api_version}/generate",
    response_model=GenerateResponse,
    summary="统一生成（流式 / 非流式）",
    description=(
        "单一入口：传入 messages（chat 风格）或 input（responses 风格），"
        "网关负责选择模型与上游协议、失败降级，并统一流式事件的形状。"
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "请求参数或调用方式不受支持"},
        status.HTTP_401_UNAUTHORIZED: {
            "description": "缺少或非法的 X-User-Id（missing_user_id / invalid_user_id）"
        },
        status.HTTP_404_NOT_FOUND: {"description": "模型未注册"},
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "网关并发已达上限，请按 Retry-After 稍后重试"
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "全部候选模型均调用失败"},
    },
)
async def generate(
    payload: GenerateRequest,
    gateway: Gateway,
    user_id: UserId,
    api_version: str = Path(..., description="API 版本，例如 v1"),
):
    # 鉴权已由 UserId 依赖完成（缺失/非法直接 401，不会触到上游）。
    # 当前只用做日志归因；后续的按用户限流/配额会复用这个标识。
    logger.debug(
        "generate user=%s model=%s mode=%s stream=%s",
        user_id,
        payload.model,
        payload.mode,
        payload.stream,
    )

    if payload.stream:
        if payload.structured_output is not None:
            # 流式无法“收回已发出的 delta”，校验只能覆盖非流式返回
            raise InvalidRequestError(
                "structured_output 仅支持非流式调用；请设置 stream=false"
            )
        requested_model = payload.model or ""
        return await build_sse_response(
            gateway.stream(payload),
            lambda event: encode_sse(event, requested_model=requested_model),
        )

    result = await gateway.complete(payload)
    return JSONResponse(
        content=to_generate_payload(result),
        headers=gateway_headers(result),
    )
