"""适配层错误体系。

分层原则：

- ``GatewayError`` 之上是“可直接映射为 HTTP 响应”的错误；
- ``retryable`` 决定是否允许降级到下一个候选模型；
- 不可重试的错误（400/401/403/内容拒答）**不得**触发降级，避免掩盖配置或参数问题。
"""

from __future__ import annotations

import json
from typing import Any

MAX_MESSAGE_LENGTH = 500


def _truncate(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= MAX_MESSAGE_LENGTH else f"{text[:MAX_MESSAGE_LENGTH]}..."


def _extract_message(body: str) -> str:
    """尽力从上游错误响应里取出人类可读的错误信息。"""
    if not body:
        return "上游未返回错误详情"
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return _truncate(body)
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg")
            if message:
                return _truncate(str(message))
        if isinstance(error, str):
            return _truncate(error)
        message = payload.get("message") or payload.get("detail")
        if message:
            return _truncate(str(message))
    return _truncate(body)


class GatewayError(Exception):
    """所有网关错误的基类。"""

    code = "internal_error"
    status_code = 500
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
        upstream_status: int | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
        self.upstream_status = upstream_status
        self.retry_after = retry_after
        # 附加诊断信息（例如 schema 校验违例），会平铺到 error 对象里
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.upstream_status is not None:
            error["upstream_status"] = self.upstream_status
        for key, value in self.details.items():
            error.setdefault(key, value)
        return {"error": error}

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"[{self.code}] {self.message}"


# ---------------------------------------------------------------------------
# 网关自身错误
# ---------------------------------------------------------------------------
class ConfigurationError(GatewayError):
    code = "configuration_error"
    status_code = 500
    retryable = False


class ModelNotFoundError(GatewayError):
    code = "model_not_found"
    status_code = 404
    retryable = False


class UnsupportedModeError(GatewayError):
    code = "unsupported_mode"
    status_code = 400
    retryable = False


class InvalidRequestError(GatewayError):
    code = "invalid_request"
    status_code = 400
    retryable = False


class AuthenticationError(GatewayError):
    """调用方身份缺失或不合法。

    具体原因用更细的 ``code`` 区分（``missing_user_id`` / ``invalid_user_id``），
    便于客户端定位问题，而不是笼统的 401。
    """

    code = "unauthorized"
    status_code = 401
    retryable = False


class GatewayConcurrencyLimited(GatewayError):
    """网关自身并发已满（不是上游限流）。

    ``code`` 必须与 :class:`ProviderRateLimitedError` 区分，否则客户端会把
    网关的过载当成上游 429，套用错误的退避/降级策略。

    ``retryable`` 在这里沿用**对外**语义（客户端可以稍后重试）。该错误只在
    入口中间件抛出，不会进入 ``FailoverClient``，因此不会触发模型降级。
    """

    code = "gateway_concurrency_limited"
    status_code = 429
    retryable = True


# ---------------------------------------------------------------------------
# 结构化输出错误
# ---------------------------------------------------------------------------
class InvalidOutputSchema(GatewayError):
    """调用方提供的 schema 本身不合法（非法 JSON Schema、超大、过深、不存在）。

    这是**客户端问题**：重试同一个请求没有意义，直接 400。
    """

    code = "invalid_output_schema"
    status_code = 400
    retryable = False


class UnsupportedStructuredOutput(GatewayError):
    """请求要求了当前配置无法满足的结构化输出能力。

    典型场景：``mode=strict`` 但降级链里没有任何声明 ``strict_json`` 能力的
    部署。这里**不做静默降级**，否则调用方会以为拿到了语法级保证。
    """

    code = "unsupported_structured_output"
    status_code = 400
    retryable = False


class OutputSchemaValidationFailed(GatewayError):
    """上游在既定的重试预算内始终产不出符合 schema 的输出。

    属于上游产出问题（502）：同参数重放大概率还是错，因此 ``retryable=False``，
    但把 attempts / 违例摘要透出，便于调用方定位与申诉。
    """

    code = "output_schema_validation_failed"
    status_code = 502
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        schema_name: str | None = None,
        schema_id: str | None = None,
        violations: list[dict[str, Any]] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        details: dict[str, Any] = {"attempts": attempts}
        if schema_name:
            details["schema_name"] = schema_name
        if schema_id:
            details["schema_id"] = schema_id
        if violations:
            details["violations"] = violations
        if finish_reason:
            details["finish_reason"] = finish_reason
        super().__init__(message, details=details)


# ---------------------------------------------------------------------------
# 上游错误
# ---------------------------------------------------------------------------
class ProviderError(GatewayError):
    code = "upstream_error"
    status_code = 502
    retryable = True


class ProviderConnectionError(ProviderError):
    code = "upstream_connection_error"
    retryable = True


class ProviderTimeoutError(ProviderError):
    code = "upstream_timeout"
    retryable = True


class ProviderRateLimitedError(ProviderError):
    code = "upstream_rate_limited"
    retryable = True


class ProviderServerError(ProviderError):
    code = "upstream_server_error"
    retryable = True


class ProviderAuthError(ProviderError):
    code = "upstream_auth_error"
    retryable = False


class ProviderRejectedError(ProviderError):
    """上游 4xx 参数/请求错误：换模型没有意义，禁止降级。"""

    code = "upstream_invalid_request"
    status_code = 400
    retryable = False


class ProviderContentFilterError(ProviderError):
    """内容审核拒答：是“成功的失败”，重试会重复计费。"""

    code = "upstream_content_filter"
    status_code = 422
    retryable = False


class AllDeploymentsFailed(GatewayError):
    code = "all_deployments_failed"
    status_code = 502
    retryable = False

    def __init__(self, errors: list[GatewayError]) -> None:
        self.errors = errors
        detail = "；".join(f"{err.code}: {err.message}" for err in errors[-3:])
        # 全部因限流失败时，把最长的退避建议透传给客户端
        retry_after = max(
            (err.retry_after for err in errors if err.retry_after is not None),
            default=None,
        )
        super().__init__(
            f"全部候选模型均调用失败（{detail or '无可用候选'}）",
            retry_after=retry_after,
        )


def error_from_status(
    status: int,
    body: str = "",
    *,
    retry_after: float | None = None,
) -> GatewayError:
    """把上游 HTTP 状态码映射为错误类型。"""
    detail = _extract_message(body)
    if status in (401, 403):
        return ProviderAuthError(
            f"上游鉴权失败（HTTP {status}）：{detail}", upstream_status=status
        )
    if status == 429:
        return ProviderRateLimitedError(
            f"上游限流（HTTP 429）：{detail}",
            upstream_status=status,
            retry_after=retry_after,
        )
    if status in (408, 504):
        return ProviderTimeoutError(
            f"上游超时（HTTP {status}）：{detail}", upstream_status=status
        )
    if status >= 500:
        return ProviderServerError(
            f"上游服务异常（HTTP {status}）：{detail}", upstream_status=status
        )
    return ProviderRejectedError(
        f"上游拒绝了请求（HTTP {status}）：{detail}", upstream_status=status
    )
