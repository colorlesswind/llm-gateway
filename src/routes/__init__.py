"""路由聚合：所有对外接口在此挂载到同一个 APIRouter。"""

from fastapi import APIRouter

from src.routes import chat, generate, health, models, prompts, responses

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router)
# 统一入口（推荐）
api_router.include_router(generate.router)
# 兼容入口
api_router.include_router(chat.router)
api_router.include_router(responses.router)
api_router.include_router(prompts.router)

__all__ = ["api_router"]
