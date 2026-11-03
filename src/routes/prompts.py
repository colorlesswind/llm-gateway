"""提示词管理接口（按用户隔离）。

模板正文可以很长（几十 KB ~ 几百 KB），因此：

- 列表接口只返回摘要（不含正文），详情接口才返回全文；
- 修改正文会生成新版本，旧版本仍可按 ``version`` 引用（结果可复现）；
- 不属于当前用户的模板一律 404，避免通过状态码探测他人数据。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from src.routes.deps import Prompts, UserId
from src.schemas import (
    Prompt,
    PromptCreateRequest,
    PromptListResponse,
    PromptUpdateRequest,
)

router = APIRouter(tags=["prompts"])

ApiVersion = Annotated[str, Path(description="API 版本，例如 v1")]
PromptId = Annotated[
    str, Path(min_length=1, max_length=64, description="模板 ID（ptpl_...）")
]

NOT_FOUND = {"description": "模板不存在或不属于当前用户"}
STORE_UNAVAILABLE = {
    "description": "模板存储未配置或不可用（prompt_store_unavailable）"
}


@router.post(
    "/{api_version}/prompts",
    response_model=Prompt,
    status_code=status.HTTP_201_CREATED,
    summary="上传提示词模板",
    description=(
        "保存模板正文并返回唯一的 ``id``。占位符写作 ``{{name}}``，"
        "上传时会解析出变量列表，之后在 /generate、/chat/completions、/responses "
        "里通过 ``prompt.id`` + ``prompt.variables`` 引用。"
    ),
    responses={
        status.HTTP_409_CONFLICT: {"description": "同名模板已存在或模板数量超限"},
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "模板正文或渲染结果超限"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: STORE_UNAVAILABLE,
    },
)
async def create_prompt(
    payload: PromptCreateRequest,
    user_id: UserId,
    prompts: Prompts,
    api_version: ApiVersion,
) -> Prompt:
    return await prompts.create(user_id, payload)


@router.get(
    "/{api_version}/prompts",
    response_model=PromptListResponse,
    summary="获取当前用户的提示词列表",
    description="返回摘要列表（不含正文），支持分页与按启用状态过滤。",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: STORE_UNAVAILABLE},
)
async def list_prompts(
    user_id: UserId,
    prompts: Prompts,
    api_version: ApiVersion,
    limit: int = Query(default=20, ge=1, le=100, description="每页条数"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
    enabled: bool | None = Query(
        default=None, description="仅返回启用 / 停用的模板"
    ),
) -> PromptListResponse:
    data, total = await prompts.list(
        user_id, limit=limit, offset=offset, enabled=enabled
    )
    return PromptListResponse(data=data, total=total, limit=limit, offset=offset)


@router.get(
    "/{api_version}/prompts/{prompt_id}",
    response_model=Prompt,
    summary="获取指定提示词",
    responses={
        status.HTTP_404_NOT_FOUND: NOT_FOUND,
        status.HTTP_503_SERVICE_UNAVAILABLE: STORE_UNAVAILABLE,
    },
)
async def get_prompt(
    prompt_id: PromptId,
    user_id: UserId,
    prompts: Prompts,
    api_version: ApiVersion,
    version: int | None = Query(
        default=None, ge=1, description="指定历史版本；缺省返回最新版本"
    ),
) -> Prompt:
    return await prompts.get(user_id, prompt_id, version=version)


@router.patch(
    "/{api_version}/prompts/{prompt_id}",
    response_model=Prompt,
    summary="更新提示词（正文变更生成新版本）",
    description=(
        "可通过 ``expected_version`` 做乐观锁；正文哈希变化时 head_version +1，"
        "旧版本仍可按 version 读取。"
    ),
    responses={
        status.HTTP_404_NOT_FOUND: NOT_FOUND,
        status.HTTP_409_CONFLICT: {"description": "名称冲突 / 版本冲突 / 数量超限"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "模板正文超限"},
        status.HTTP_503_SERVICE_UNAVAILABLE: STORE_UNAVAILABLE,
    },
)
async def update_prompt(
    prompt_id: PromptId,
    payload: PromptUpdateRequest,
    user_id: UserId,
    prompts: Prompts,
    api_version: ApiVersion,
) -> Prompt:
    return await prompts.update(user_id, prompt_id, payload)


@router.delete(
    "/{api_version}/prompts/{prompt_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除提示词（软删除）",
    responses={
        status.HTTP_404_NOT_FOUND: NOT_FOUND,
        status.HTTP_503_SERVICE_UNAVAILABLE: STORE_UNAVAILABLE,
    },
)
async def delete_prompt(
    prompt_id: PromptId,
    user_id: UserId,
    prompts: Prompts,
    api_version: ApiVersion,
) -> None:
    await prompts.delete(user_id, prompt_id)

