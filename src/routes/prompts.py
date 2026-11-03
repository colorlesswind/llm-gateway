"""提示词管理接口（按用户隔离）。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Path, status

from src.schemas import Prompt, PromptCreateRequest, PromptListResponse

router = APIRouter(tags=["prompts"])

UserId = Annotated[
    str,
    Header(
        alias="X-User-Id",
        min_length=1,
        description="用户标识；后续将替换为鉴权得到的身份",
    ),
]


@router.post(
    "/{api_version}/prompts",
    response_model=Prompt,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="上传提示词",
    description="接收用户提交的提示词文档，之后可按条件注入到发往 LLM 的 messages 中。",
    responses={
        status.HTTP_501_NOT_IMPLEMENTED: {"description": "业务逻辑尚未实现"},
    },
)
async def create_prompt(
    payload: PromptCreateRequest,
    user_id: UserId,
    api_version: str = Path(..., description="API 版本，例如 v1"),
) -> Prompt:
    # TODO: 校验并持久化提示词，生成 id 与时间戳
    # TODO: 设计注入条件与优先级字段
    # TODO: 处理同用户重名、内容大小限制
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="prompts 写入逻辑尚未实现",
    )


@router.get(
    "/{api_version}/prompts",
    response_model=PromptListResponse,
    summary="获取当前用户的提示词列表",
)
async def list_prompts(
    user_id: UserId,
    api_version: str = Path(..., description="API 版本，例如 v1"),
) -> PromptListResponse:
    # TODO: 从存储中查询该 user_id 下的全部提示词
    # TODO: 支持分页与按启用状态过滤
    return PromptListResponse(data=[], total=0)


@router.get(
    "/{api_version}/prompts/{prompt_id}",
    response_model=Prompt,
    summary="获取指定提示词",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "提示词不存在或不属于当前用户"},
        status.HTTP_501_NOT_IMPLEMENTED: {"description": "业务逻辑尚未实现"},
    },
)
async def get_prompt(
    prompt_id: str,
    user_id: UserId,
    api_version: str = Path(..., description="API 版本，例如 v1"),
) -> Prompt:
    # TODO: 按 prompt_id 查询，并校验归属该 user_id
    # TODO: 不存在时返回 404，避免泄露其他用户的数据
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="prompts 查询逻辑尚未实现",
    )
