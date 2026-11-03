"""SSE 响应构造工具。

要点：
1. **先取首事件再建响应**——这样在真正开始流式输出前发生的错误（鉴权失败、
   全部模型失败等）仍能返回正确的 HTTP 状态码，而不是 200 + 一个错误帧；
2. 首事件若本身是 ``error``，同样上升为 HTTP 错误；
3. 客户端断连时通过 ``finally`` 关闭上游流，避免继续计费。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping

from fastapi.responses import StreamingResponse

from src.services.llm_adaptor import GatewayError, aclose_stream
from src.services.llm_adaptor.types import StreamEvent

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def build_sse_response(
    events: AsyncIterator[StreamEvent],
    encoder: Callable[[StreamEvent], str],
    *,
    terminator: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> StreamingResponse:
    try:
        first: StreamEvent | None = await events.__anext__()
    except StopAsyncIteration:
        first = None
    except BaseException:
        await aclose_stream(events)
        raise

    if first is not None and first.type == "error" and first.error is not None:
        await aclose_stream(events)
        raise first.error

    async def body() -> AsyncIterator[str]:
        try:
            if first is not None:
                yield encoder(first)
            async for event in events:
                yield encoder(event)
            if terminator is not None:
                yield terminator
        finally:
            await aclose_stream(events)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, **(headers or {})},
    )


__all__ = ["SSE_HEADERS", "GatewayError", "build_sse_response"]
