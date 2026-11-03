"""模型列表接口。"""

from __future__ import annotations

from fastapi import APIRouter, Path

from src.routes.deps import Gateway
from src.schemas import ModelInfo, ModelListResponse

router = APIRouter(tags=["models"])


@router.get(
    "/{api_version}/models",
    response_model=ModelListResponse,
    summary="获取可用模型",
    description="返回注册表中所有可调用的模型（含厂商 API 与本地 ollama 模型）。",
)
async def list_models(
    gateway: Gateway,
    api_version: str = Path(..., description="API 版本，例如 v1"),
) -> ModelListResponse:
    data = [
        ModelInfo(
            id=item["id"],
            owned_by=item["provider"],
            provider=item["provider"],
        )
        for item in gateway.registry.describe()
    ]
    return ModelListResponse(data=data)
