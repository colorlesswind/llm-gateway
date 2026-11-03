"""异步事件流工具。

``AsyncIterator`` 协议只保证 ``__anext__``，而 ``aclose`` 属于更窄的
``AsyncGenerator``。调用方为了保持最宽契约（只要求能被迭代）无法直接关闭，
这里统一提供「尽力关闭」的实现。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


async def aclose_stream(events: AsyncIterator[Any]) -> None:
    """尽力关闭异步事件流。

    - 非 async generator（没有 ``aclose``）直接跳过，不报 ``AttributeError``；
    - 重复关闭 / 已耗尽（``RuntimeError``、``GeneratorExit``）视为正常；
    - 关闭阶段的其它异常只记日志，不影响已建立的响应或待重抛的原始异常。
    """
    closer = getattr(events, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except (RuntimeError, GeneratorExit):
        pass
    except Exception:  # pragma: no cover - 关闭阶段异常不应影响主流程
        logger.debug("关闭上游流时出错", exc_info=True)


__all__ = ["aclose_stream"]
