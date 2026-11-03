"""厂商适配层。

对外只暴露 :class:`~src.services.llm_adaptor.gateway.LLMGateway`：

    gateway = LLMGateway.create(settings)
    result = await gateway.complete(payload)          # 非流式
    async for event in gateway.stream(payload): ...   # 流式
"""

from src.services.llm_adaptor.base import Cap, Deployment, UpstreamMode
from src.services.llm_adaptor.errors import (
    AllDeploymentsFailed,
    AuthenticationError,
    ConfigurationError,
    GatewayConcurrencyLimited,
    GatewayError,
    InvalidOutputSchema,
    InvalidRequestError,
    ModelNotFoundError,
    OutputSchemaValidationFailed,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderServerError,
    ProviderTimeoutError,
    UnsupportedModeError,
    UnsupportedStructuredOutput,
)
from src.services.llm_adaptor.gateway import LLMGateway
from src.services.llm_adaptor.registry import ModelRegistry
from src.services.llm_adaptor.types import (
    GatewayResult,
    LLMRequest,
    LLMResult,
    Message,
    StreamEvent,
    Usage,
)
from src.services.llm_adaptor.utils import aclose_stream

__all__ = [
    "AllDeploymentsFailed",
    "AuthenticationError",
    "Cap",
    "ConfigurationError",
    "Deployment",
    "GatewayConcurrencyLimited",
    "GatewayError",
    "GatewayResult",
    "InvalidOutputSchema",
    "InvalidRequestError",
    "LLMGateway",
    "LLMRequest",
    "LLMResult",
    "Message",
    "ModelNotFoundError",
    "ModelRegistry",
    "OutputSchemaValidationFailed",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderRateLimitedError",
    "ProviderRejectedError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "StreamEvent",
    "UnsupportedModeError",
    "UnsupportedStructuredOutput",
    "UpstreamMode",
    "Usage",
    "aclose_stream",
]
