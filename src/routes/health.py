"""部署相关的存活与就绪检查。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from src.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


async def check_database(request: Request) -> tuple[str, str | None]:
    """返回 ``(状态, 说明)``，状态取 ``ok / unavailable / not_configured``。

    提示词模板是可选的：``DATABASE_URL`` 为空时进程仍然可以处理纯模型调用，
    因此这里必须区分「未配置」与「配置了但连不上」，不能用同一个 503 掩盖。
    """
    service = getattr(request.app.state, "prompt_service", None)
    if service is None:
        return "not_configured", "未配置 DATABASE_URL，提示词模板能力不可用"
    if await service.ping():
        return "ok", None
    return "unavailable", "数据库连接不可用"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="存活检查",
    description="仅表示进程仍在运行，不检查下游依赖。",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="就绪检查",
    description="检查服务是否已具备处理业务请求的条件（数据库 + 模型注册表）。",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
async def ready(request: Request, response: Response) -> ReadyResponse:
    gateway = getattr(request.app.state, "llm_gateway", None)
    models = len(gateway.registry.all_deployments) if gateway is not None else 0
    providers = "ok" if models else "not_configured"

    database, detail = await check_database(request)
    if database != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="not_ready",
            database=database,  # type: ignore[arg-type]
            providers=providers,
            models=models,
            detail=detail,
        )
    return ReadyResponse(
        status="ready", database="ok", providers=providers, models=models
    )
