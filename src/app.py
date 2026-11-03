"""应用装配。

放在 ``src/app.py`` 而不是 ``src/main.py``，是为了让测试可以导入
:func:`create_app` 并注入配置，而不会触发 ``src/main.py`` 的模块级实例化
（那一步会在配置缺失时立即失败，符合“启动期 fail fast”的预期）。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import Settings
from src.middleware import ConcurrencyLimitMiddleware
from src.routes import api_router
from src.routes.deps import USER_ID_HEADER
from src.services.llm_adaptor import (
    AuthenticationError,
    GatewayError,
    LLMGateway,
)
from src.services.prompts import PromptService, SqlPromptStore
from src.services.ratelimit import InflightLimiter

logger = logging.getLogger(__name__)


def _build_prompt_service(settings: Settings) -> PromptService | None:
    """``DATABASE_URL`` 为空时不构造存储，prompts 相关能力返回 503。"""
    url = (settings.database_url or "").strip()
    if not url:
        return None
    return PromptService(SqlPromptStore(url), settings)


def _log_value(value: str, limit: int = 64) -> str:
    """把不可信的 header 值压成单行安全字符串，避免日志注入。"""
    cleaned = "".join(" " if ord(ch) < 32 or ord(ch) == 127 else ch for ch in value)
    return cleaned[:limit] or "-"


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    env: dict[str, str] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    # 启动期完成注册表校验：配置缺失时直接抛 ConfigurationError，而不是等首个请求
    gateway = LLMGateway.create(settings, transport=transport, env=env)
    prompt_service = _build_prompt_service(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 建表失败不阻断启动（与 /ready 的 database 三态保持一致），
        # 但会把原因写到日志，避免“静默 503”。
        service = app.state.prompt_service
        if service is not None:
            try:
                await service.initialize()
            except GatewayError as exc:
                logger.error("提示词模板库初始化失败，相关能力不可用：%s", exc)
        yield
        await app.state.llm_gateway.aclose()
        if app.state.prompt_service is not None:
            await app.state.prompt_service.aclose()

    app = FastAPI(
        title="LLM Gateway",
        version="0.2.0",
        description=(
            "统一管理厂商模型 API 与本地 ollama 模型调用的网关："
            "对外一个统一入口，内部按 chat / responses 协议与降级链调度。"
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.llm_gateway = gateway
    # None 表示未配置数据库：prompts 接口 503，生成接口仍可正常调用
    app.state.prompt_service = prompt_service

    # 入口并发限流：计数在进程内，多 worker 部署时总上限 = 该值 × worker 数。
    # 配置 <= 0 表示关闭（此时 app.state.inflight_limiter 为 None）。
    limiter: InflightLimiter | None = None
    if settings.max_concurrent_requests > 0:
        limiter = InflightLimiter(settings.max_concurrent_requests)
        app.add_middleware(
            ConcurrencyLimitMiddleware,
            limiter=limiter,
            retry_after=settings.overload_retry_after,
        )
    app.state.inflight_limiter = limiter

    @app.exception_handler(GatewayError)
    async def _handle_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        logger.warning(
            "%s %s user=%s -> %s",
            request.method,
            request.url.path,
            _log_value(request.headers.get("x-user-id", "")),
            exc,
        )
        headers = (
            {"Retry-After": f"{exc.retry_after:g}"}
            if exc.retry_after is not None
            else None
        )
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_dict(), headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ):
        """把 ``X-User-Id`` 的校验失败统一成 401，其余仍保持 422。"""
        for error in exc.errors():
            loc = error.get("loc") or ()
            if (
                len(loc) == 2
                and loc[0] == "header"
                and str(loc[1]).lower() == USER_ID_HEADER.lower()
            ):
                code = (
                    "missing_user_id"
                    if error.get("type") in {"missing", "string_too_short"}
                    else "invalid_user_id"
                )
                auth_error = AuthenticationError(
                    "X-User-Id 请求头缺失或非法", code=code
                )
                return JSONResponse(
                    status_code=auth_error.status_code,
                    content=auth_error.to_dict(),
                )
        return await request_validation_exception_handler(request, exc)

    app.include_router(api_router)
    return app
