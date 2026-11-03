"""统一网关入口：解析候选链 → 构造 IR → 调用（含降级）→ 出口翻译。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx

from src.config import Settings, raw_env
from src.schemas import GenerateRequest
from src.services.llm_adaptor.base import Cap, Deployment, LLMAdapter, UpstreamMode
from src.services.llm_adaptor.errors import (
    InvalidRequestError,
    OutputSchemaValidationFailed,
)
from src.services.llm_adaptor.failover import Attempt, FailoverClient
from src.services.llm_adaptor.providers import (
    OpenAICompatAdapter,
    OpenAIResponsesAdapter,
)
from src.services.llm_adaptor.registry import ModelRegistry
from src.services.llm_adaptor.structured import (
    StructuredSpec,
    build_repair_messages,
    extract_json,
    prepare_structured,
    validate_instance,
)
from src.services.llm_adaptor.translator import build_llm_request
from src.services.llm_adaptor.types import (
    GatewayResult,
    LLMRequest,
    StreamEvent,
    StructuredOutcome,
)
from src.services.llm_adaptor.types import (
    Usage as IRUsage,
)

logger = logging.getLogger(__name__)


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        failover: FailoverClient,
        adapters: dict[UpstreamMode, LLMAdapter],
    ) -> None:
        self.settings = settings
        self.registry = registry
        self._failover = failover
        self._adapters = adapters

    @classmethod
    def create(
        cls,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        env: dict[str, str] | None = None,
    ) -> LLMGateway:
        registry = ModelRegistry.from_env(
            raw_env() if env is None else env,
            chain=settings.model_chain_list,
            enable_responses=settings.enable_responses_mode,
        )
        registry.validate()

        adapters: dict[UpstreamMode, LLMAdapter] = {
            UpstreamMode.CHAT: OpenAICompatAdapter(
                connect_timeout=settings.connect_timeout,
                read_timeout=settings.read_timeout,
                transport=transport,
            ),
            UpstreamMode.RESPONSES: OpenAIResponsesAdapter(
                connect_timeout=settings.connect_timeout,
                read_timeout=settings.read_timeout,
                transport=transport,
            ),
        }
        return cls(settings, registry, FailoverClient(adapters, settings), adapters)

    async def aclose(self) -> None:
        for adapter in self._adapters.values():
            await adapter.aclose()

    # -- 非流式 -------------------------------------------------------------
    async def complete(self, payload: GenerateRequest) -> GatewayResult:
        if payload.structured_output is not None and payload.response_format is not None:
            raise InvalidRequestError(
                "structured_output 与 response_format 不能同时使用"
            )

        mode, chain = self.registry.resolve(payload.model, payload.mode)
        spec = prepare_structured(
            payload.structured_output,
            settings=self.settings,
            deployments=chain,
        )
        if spec is None:
            request = build_llm_request(payload, mode=mode)
            attempt = await self._failover.complete(chain, request)
            return self._result(payload, mode, chain, attempt)

        if spec.mode == "strict":
            # 严格模式只允许声明了 strict_json 的部署，避免静默退化
            chain = [d for d in chain if d.caps & Cap.STRICT_JSON]
        request = build_llm_request(payload, mode=mode, structured=spec)
        return await self._complete_structured(payload, mode, chain, request, spec)

    async def _complete_structured(
        self,
        payload: GenerateRequest,
        mode: UpstreamMode,
        chain: list[Deployment],
        request: LLMRequest,
        spec: StructuredSpec,
    ) -> GatewayResult:
        """校验 + 有限次修复重试（位于降级层之上，不污染熔断计数）。"""
        base_messages = list(request.messages)
        usage = IRUsage()
        attempts = 0
        preferred: Deployment | None = None
        attempt: Attempt | None = None
        violations: list[dict] = []

        while attempts < spec.max_attempts:
            attempts += 1
            # 优先命中上一次成功的部署：同一模型对同一 schema 的重试成功率更高，
            # 只有它报错时才由 failover 往下降级。
            current = (
                chain
                if preferred is None
                else [preferred, *[d for d in chain if d.key != preferred.key]]
            )
            attempt = await self._failover.complete(current, request)
            preferred = attempt.deployment
            if self.settings.schema_sum_usage:
                usage.merge(attempt.result.usage)

            value = None
            try:
                value = extract_json(attempt.result.text)
            except ValueError as exc:
                violations = [{"path": "$", "reason": str(exc)}]
            else:
                violations = validate_instance(value, spec.schema)

            if not violations:
                outcome = StructuredOutcome(
                    schema_name=spec.name,
                    schema_id=spec.schema_id,
                    mode=spec.mode,
                    valid=True,
                    attempts=attempts,
                    value=value,
                )
                return self._result(
                    payload,
                    mode,
                    chain,
                    attempt,
                    attempts=attempts,
                    structured=outcome,
                    usage=self._final_usage(usage, attempt),
                )

            logger.warning(
                "结构化输出校验失败（第 %d/%d 次，失败原因：%s）",
                attempts,
                spec.max_attempts,
                "; ".join(f"{item['path']}: {item['reason']}" for item in violations[:3]),
            )
            if attempt.result.finish_reason == "length" and not (
                self.settings.schema_repair_on_length
            ):
                logger.warning("输出被截断（finish_reason=length），不再浪费重试预算")
                break
            if attempts >= spec.max_attempts:
                break
            request = replace(
                request,
                messages=build_repair_messages(
                    base_messages,
                    attempt.result.text,
                    violations,
                    max_error_chars=self.settings.schema_repair_max_error_chars,
                    max_output_chars=self.settings.schema_repair_max_output_chars,
                ),
            )

        assert attempt is not None  # spec.max_attempts >= 1
        final_usage = self._final_usage(usage, attempt)
        if spec.on_invalid == "return_raw":
            outcome = StructuredOutcome(
                schema_name=spec.name,
                schema_id=spec.schema_id,
                mode=spec.mode,
                valid=False,
                attempts=attempts,
                violations=violations,
            )
            return self._result(
                payload,
                mode,
                chain,
                attempt,
                attempts=attempts,
                structured=outcome,
                usage=final_usage,
            )

        raise OutputSchemaValidationFailed(
            f"尝试 {attempts} 次后输出仍不符合 schema {spec.name!r}",
            attempts=attempts,
            schema_name=spec.name,
            schema_id=spec.schema_id,
            violations=violations,
            finish_reason=attempt.result.finish_reason,
        )

    def _final_usage(self, accumulated: IRUsage, attempt: Attempt) -> IRUsage:
        """按配置决定是“所有尝试累加”还是“只算最后一次”。"""
        if self.settings.schema_sum_usage:
            return accumulated
        return attempt.result.usage or IRUsage()

    def _result(
        self,
        payload: GenerateRequest,
        mode: UpstreamMode,
        chain: list[Deployment],
        attempt: Attempt,
        *,
        attempts: int = 1,
        structured: StructuredOutcome | None = None,
        usage: IRUsage | None = None,
    ) -> GatewayResult:
        result = (
            attempt.result
            if usage is None
            else replace(attempt.result, usage=usage)
        )
        return GatewayResult(
            result=result,
            deployment=attempt.deployment,
            model=payload.model or chain[0].model,
            mode=mode,
            # 以“最终服务的部署是否等于链首”为准：修复重试会把上次成功的部署
            # 提到链首，attempt.fallback 在第二轮会变成 False，导致降级信号丢失。
            fallback=attempt.deployment.key != chain[0].key,
            structured=structured,
            attempts=attempts,
        )

    # -- 流式 ---------------------------------------------------------------
    async def stream(self, payload: GenerateRequest) -> AsyncIterator[StreamEvent]:
        mode, chain = self.registry.resolve(payload.model, payload.mode)
        request = build_llm_request(payload, mode=mode)
        requested_model = payload.model or chain[0].model
        async for event in self._failover.stream(
            chain, request, requested_model=requested_model
        ):
            yield event
