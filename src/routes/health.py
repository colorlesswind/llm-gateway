"""部署相关的存活与就绪检查。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from src.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


async def check_database() -> bool:
    """检查数据库连接是否就绪。

    TODO: 接入真实数据库后实现连接探测（例如执行一次 SELECT 1），
    失败时返回 False，并区分“未配置”与“连接不可用”。
    """
    return False


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

    # TODO: 数据库接入后，替换为真实的连接池/会话探测
    database_ready = await check_database()
    if not database_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="not_ready",
            database="not_configured",
            providers=providers,
            models=models,
            detail="数据库连接尚未接入",
        )
    return ReadyResponse(
        status="ready", database="ok", providers=providers, models=models
    )
