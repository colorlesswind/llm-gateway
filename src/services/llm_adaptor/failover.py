"""降级、限流退避与熔断。

策略（与设计约定一致）：

1. 只有可重试错误（连接失败 / 超时 / 5xx / 429）才降级；
2. 429 在**同一部署**上指数退避重试，最多 ``RETRY_MAX_ATTEMPTS`` 次（含首次）；
3. 400/401/403/内容拒答等不可重试错误直接抛出，不换模型；
4. 流式响应一旦发出首个事件就不再降级，只下发 ``error`` 事件。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from src.config import Settings
from src.services.llm_adaptor.base import Deployment, LLMAdapter, UpstreamMode
from src.services.llm_adaptor.errors import (
    AllDeploymentsFailed,
    GatewayError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from src.services.llm_adaptor.types import LLMRequest, LLMResult, StreamEvent
from src.services.llm_adaptor.utils import aclose_stream

logger = logging.getLogger(__name__)


@dataclass
class Attempt:
    """一次成功的调用结果，附带实际服务的部署。"""

    deployment: Deployment
    fallback: bool
    result: LLMResult


@dataclass
class _Health:
    failures: int = 0
    opened_until: float = 0.0


class FailoverClient:
    def __init__(
        self, adapters: dict[UpstreamMode, LLMAdapter], settings: Settings
    ) -> None:
        self._adapters = adapters
        self._settings = settings
        self._health: dict[str, _Health] = {}

    # -- 熔断 ---------------------------------------------------------------
    def _is_open(self, deployment: Deployment) -> bool:
        health = self._health.get(deployment.key)
        if health is None or health.opened_until == 0.0:
            return False
        if health.opened_until > time.monotonic():
            return True
        health.opened_until = 0.0
        return False

    def _record_success(self, deployment: Deployment) -> None:
        self._health.pop(deployment.key, None)

    def _record_failure(self, deployment: Deployment) -> None:
        health = self._health.setdefault(deployment.key, _Health())
        health.failures += 1
        if health.failures >= self._settings.circuit_failure_threshold:
            health.opened_until = time.monotonic() + self._settings.circuit_cooldown
            health.failures = 0
            logger.error(
                "模型 %s 连续失败，熔断 %.0fs",
                deployment.model,
                self._settings.circuit_cooldown,
            )

    def _backoff(self, attempt: int, error: ProviderRateLimitedError) -> float:
        if error.retry_after is not None:
            return min(error.retry_after, self._settings.retry_max_delay)
        delay = self._settings.retry_base_delay * (2 ** (attempt - 1))
        delay = min(delay, self._settings.retry_max_delay)
        return delay * (0.5 + random.random() / 2)

    # -- 非流式 -------------------------------------------------------------
    async def complete(
        self, chain: Sequence[Deployment], request: LLMRequest
    ) -> Attempt:
        errors: list[GatewayError] = []
        for index, deployment in enumerate(chain):
            if self._is_open(deployment):
                logger.warning("模型 %s 处于熔断状态，跳过", deployment.model)
                continue
            attempt = 0
            while True:
                try:
                    result = await self._adapters[deployment.mode].complete(
                        deployment, request
                    )
                except ProviderRateLimitedError as error:
                    attempt += 1
                    self._record_failure(deployment)
                    if attempt >= self._settings.retry_max_attempts:
                        errors.append(error)
                        break
                    delay = self._backoff(attempt, error)
                    logger.warning(
                        "模型 %s 被限流，%.2fs 后重试（第 %d 次）",
                        deployment.model,
                        delay,
                        attempt,
                    )
                    await asyncio.sleep(delay)
                    continue
                except GatewayError as error:
                    self._record_failure(deployment)
                    if not error.retryable:
                        raise
                    errors.append(error)
                    logger.warning(
                        "模型 %s 调用失败，尝试降级：%s", deployment.model, error
                    )
                    break
                self._record_success(deployment)
                return Attempt(deployment, index > 0, result)
        raise AllDeploymentsFailed(errors)

    # -- 流式 ---------------------------------------------------------------
    async def stream(
        self,
        chain: Sequence[Deployment],
        request: LLMRequest,
        *,
        requested_model: str,
    ) -> AsyncIterator[StreamEvent]:
        errors: list[GatewayError] = []
        for index, deployment in enumerate(chain):
            if self._is_open(deployment):
                logger.warning("模型 %s 处于熔断状态，跳过", deployment.model)
                continue
            attempt = 0
            while True:
                started = False
                generator = self._adapters[deployment.mode].stream(deployment, request)
                try:
                    while True:
                        timeout = (
                            self._settings.idle_timeout
                            if started
                            else self._settings.first_token_timeout
                        )
                        try:
                            event = await asyncio.wait_for(generator.__anext__(), timeout)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            phase = "流式空闲" if started else "首包"
                            raise ProviderTimeoutError(
                                f"{phase}超时（{timeout}s）", upstream_status=None
                            ) from exc

                        started = True
                        if event.type == "done":
                            event.provider = deployment.provider
                            event.model = requested_model
                            event.upstream_model = deployment.upstream_model
                            event.fallback = index > 0
                        yield event
                except ProviderRateLimitedError as error:
                    attempt += 1
                    self._record_failure(deployment)
                    if started:
                        yield _error_event(error, deployment, requested_model)
                        return
                    if attempt >= self._settings.retry_max_attempts:
                        errors.append(error)
                        break
                    delay = self._backoff(attempt, error)
                    logger.warning(
                        "模型 %s 被限流，%.2fs 后重试（第 %d 次）",
                        deployment.model,
                        delay,
                        attempt,
                    )
                    await asyncio.sleep(delay)
                    continue
                except GatewayError as error:
                    self._record_failure(deployment)
                    if started:
                        # 已经向客户端输出内容，不能再换模型
                        logger.error(
                            "模型 %s 流式输出中断（已发送首包），不再降级：%s",
                            deployment.model,
                            error,
                        )
                        yield _error_event(error, deployment, requested_model)
                        return
                    if not error.retryable:
                        raise
                    errors.append(error)
                    logger.warning(
                        "模型 %s 流式调用失败，尝试降级：%s", deployment.model, error
                    )
                    break
                else:
                    self._record_success(deployment)
                    return
                finally:
                    await aclose_stream(generator)
        raise AllDeploymentsFailed(errors)


def _error_event(
    error: GatewayError, deployment: Deployment, requested_model: str
) -> StreamEvent:
    return StreamEvent(
        type="error",
        error=error,
        provider=deployment.provider,
        model=requested_model,
        upstream_model=deployment.upstream_model,
    )


__all__ = ["Attempt", "FailoverClient"]
