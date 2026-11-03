"""纯 ASGI 中间件：LLM 入口的并发限流。

为什么不用 ``BaseHTTPMiddleware``
--------------------------------
它会给响应额外套一层任务组与内存流，已知会干扰流式响应的取消语义
（:mod:`src.routes._sse` 依赖 ``finally`` 及时关闭上游流、避免继续计费）。
所以这里直接实现 ASGI 应用协议。

为什么放在中间件而不是路由依赖
------------------------------
FastAPI 的 ``yield`` 依赖在**响应发出之前**就会执行退出码，而
``StreamingResponse`` 的 body 是在那之后才迭代的 —— 用依赖释放名额会让
流式请求几乎立刻「归还」，限流对 SSE 完全失效。中间件包住的
``await self.app(...)`` 覆盖了整个响应体发送过程，生命周期才是对的。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.services.llm_adaptor.errors import GatewayConcurrencyLimited
from src.services.ratelimit import InflightLimiter

logger = logging.getLogger(__name__)

# 真正会打到上游模型的三类入口：generate（推荐）/ chat 兼容 / responses 兼容。
# 只作用于这三条路径：``/health``、``/ready``、``/v1/models`` 不受影响，
# 避免限流把健康检查一起拒掉、导致探针失败后反复重启。
_LIMITED_PATH = re.compile(r"^/[^/]+/(?:generate|chat/completions|responses)$")


def is_llm_entry_path(path: str) -> bool:
    """判断某个请求路径是否需要计入并发限流。"""
    return _LIMITED_PATH.match(path.rstrip("/") or "/") is not None


class ConcurrencyLimitMiddleware:
    """并发已满时立刻返回 429，不排队等待。"""

    def __init__(
        self,
        app: ASGIApp,
        limiter: InflightLimiter,
        *,
        is_limited: Callable[[str], bool] = is_llm_entry_path,
        retry_after: float = 1.0,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.is_limited = is_limited
        self.retry_after = retry_after

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.is_limited(scope["path"]):
            await self.app(scope, receive, send)
            return

        lease = self.limiter.try_acquire()
        if lease is None:
            logger.warning(
                "并发上限 %d 已满，拒绝 %s %s",
                self.limiter.limit,
                scope.get("method"),
                scope["path"],
            )
            await self._reject(scope, receive, send)
            return

        try:
            # 正常结束、抛异常、客户端断连（取消）都会走到 finally，
            # 且覆盖 SSE 整条流的发送过程。
            await self.app(scope, receive, send)
        finally:
            lease.release()

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        # 先把请求体读完再回包：直接断开会让部分客户端（keep-alive / HTTP/2）
        # 收到连接重置而不是干净的 429。读取出错时不阻塞拒绝。
        try:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if not message.get("more_body", False):
                    break
        except Exception:  # pragma: no cover - 客户端提前断开
            logger.debug("读取被拒绝请求的 body 时出错", exc_info=True)

        error = GatewayConcurrencyLimited(
            f"网关并发请求已达上限（{self.limiter.limit}），请稍后重试",
            retry_after=self.retry_after,
        )
        response = JSONResponse(
            status_code=error.status_code,
            content=error.to_dict(),
            headers={"Retry-After": f"{self.retry_after:g}"},
        )
        await response(scope, receive, send)


__all__ = ["ConcurrencyLimitMiddleware", "is_llm_entry_path"]
