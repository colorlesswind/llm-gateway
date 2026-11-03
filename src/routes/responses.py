"""Responses 兼容入口。

只在 ``ENABLE_RESPONSES_MODE=true`` 且目标模型的上游协议为 responses 时可用；
网关不做 chat → responses 的事件模拟，因此本入口仅支持非流式调用，
需要流式请使用统一的 ``/{api_version}/generate``。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import JSONResponse

from src.routes.deps import Gateway
from src.schemas import GenerateRequest, ResponseObject, ResponseRequest
from src.services.llm_adaptor.translator import gateway_headers, to_response_object

router = APIRouter(tags=["responses"])


@router.post(
    "/{api_version}/responses",
    response_model=ResponseObject,
    summary="Responses（兼容入口，仅非流式）",
    description=(
        "以 responses 形状调用下游模型。需要 ENABLE_RESPONSES_MODE=true，"
        "且目标模型的上游协议为 responses。"
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "responses 方式未启用或不支持"},
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "网关并发已达上限，请按 Retry-After 稍后重试"
        },
        status.HTTP_501_NOT_IMPLEMENTED: {"description": "本入口不支持流式"},
    },
)
async def create_response(
    payload: ResponseRequest,
    gateway: Gateway,
    api_version: str = Path(..., description="API 版本，例如 v1"),
):
    if payload.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "responses 流式不支持经由本兼容入口；"
                f"请使用 /{api_version}/generate 并设置 stream=true"
            ),
        )

    unified = GenerateRequest(
        model=payload.model,
        mode="responses",
        input=payload.input,
        instructions=payload.instructions,
        temperature=payload.temperature,
        max_tokens=payload.max_output_tokens,
        stream=False,
        structured_output=payload.structured_output,
        **(payload.model_extra or {}),
    )
    result = await gateway.complete(unified)
    return JSONResponse(
        content=to_response_object(result),
        headers=gateway_headers(result),
    )
