"""路由层依赖。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from src.services.llm_adaptor import AuthenticationError, LLMGateway

# 客户端身份标识的请求头
USER_ID_HEADER = "X-User-Id"
# 上限留足空间给 UUID / 邮箱 / 租户前缀，同时避免超长字符串进日志和后续的限流键
MAX_USER_ID_LENGTH = 128


def parse_user_id(raw: str | None) -> str:
    """校验并规范化客户端标识。

    本轮只做「形状」校验（非空、长度、无控制字符）：身份来自调用方自己
    声明的 header，**并不是可信凭证** —— 任何客户端都可以随便填一个值。
    接入真实鉴权（API key / JWT / session）后，这里应改为从凭证中解析身份，
    header 只保留作调试辅助。
    """
    user_id = (raw or "").strip()
    if not user_id:
        raise AuthenticationError("缺少 X-User-Id 请求头", code="missing_user_id")
    if len(user_id) > MAX_USER_ID_LENGTH:
        raise AuthenticationError(
            f"X-User-Id 超长（上限 {MAX_USER_ID_LENGTH} 字符）",
            code="invalid_user_id",
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in user_id):
        # 控制字符会污染日志，也会破坏后续把 user_id 当作限流键/指标标签的假设
        raise AuthenticationError("X-User-Id 含非法字符", code="invalid_user_id")
    return user_id


def require_user_id(
    x_user_id: Annotated[
        str | None,
        Header(alias=USER_ID_HEADER, description="客户端唯一标识（必填）"),
    ] = None,
) -> str:
    """准入依赖：缺失或非法时抛 401（而不是 FastAPI 默认的 422）。"""
    return parse_user_id(x_user_id)


def get_gateway(request: Request) -> LLMGateway:
    gateway = getattr(request.app.state, "llm_gateway", None)
    if gateway is None:  # pragma: no cover - 正常启动流程不会出现
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM 网关尚未初始化",
        )
    return gateway


Gateway = Annotated[LLMGateway, Depends(get_gateway)]
UserId = Annotated[str, Depends(require_user_id)]

__all__ = [
    "MAX_USER_ID_LENGTH",
    "USER_ID_HEADER",
    "Gateway",
    "UserId",
    "get_gateway",
    "parse_user_id",
    "require_user_id",
]
