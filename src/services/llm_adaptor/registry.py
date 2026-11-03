"""模型注册表：把 `.env` 里的模型配置解析为可调用的部署列表。

约定（键名全部大写，模型 ID 中的 `-`/`.` 转写为 `_`）：

    MODEL_CHAIN=deepseek-chat,gpt-4o-mini,qwen-max

    MODEL_DEEPSEEK_CHAT_PROVIDER=deepseek          # 必填
    MODEL_DEEPSEEK_CHAT_UPSTREAM=deepseek-chat     # 选填，默认等于模型 ID
    MODEL_DEEPSEEK_CHAT_MODE=chat                  # chat | responses，默认 chat
    MODEL_DEEPSEEK_CHAT_CAPS=tools,json_schema     # 选填，默认全部能力
    MODEL_DEEPSEEK_CHAT_TIMEOUT=30                 # 选填，覆盖单次调用读超时

    DEEPSEEK_API_KEY=sk-xxx                        # <PROVIDER>_API_KEY
    DEEPSEEK_BASE_URL=https://api.deepseek.com/v1  # <PROVIDER>_BASE_URL
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.services.llm_adaptor.base import (
    CAP_NAMES,
    DEFAULT_CAPS,
    Cap,
    Deployment,
    UpstreamMode,
)
from src.services.llm_adaptor.errors import (
    ConfigurationError,
    ModelNotFoundError,
    UnsupportedModeError,
)

logger = logging.getLogger(__name__)

MODEL_PREFIX = "MODEL_"
_FIELDS = (
    "PROVIDER",
    "UPSTREAM",
    "MODE",
    "CAPS",
    "TIMEOUT",
    "MAX_TOKENS_FIELD",
)

# 已知厂商的默认 base_url，未配置 <PROVIDER>_BASE_URL 时使用
KNOWN_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}

# 本地部署不需要 API Key
KEYLESS_PROVIDERS = frozenset({"ollama", "vllm", "lmstudio"})


def _env_key(model_id: str, field: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", model_id).upper()
    return f"{MODEL_PREFIX}{normalized}_{field}"


def _normalize_model_id(raw: str) -> str:
    return raw.strip().lower().replace("_", "-")


def _collect_model_ids(env: dict[str, str]) -> set[str]:
    """扫描环境变量，反推已注册的模型 ID。"""
    ids: set[str] = set()
    for key, value in env.items():
        if not value or not key.upper().startswith(MODEL_PREFIX):
            continue
        name = key[len(MODEL_PREFIX) :].upper()
        for field in _FIELDS:
            suffix = f"_{field}"
            if name.endswith(suffix):
                ids.add(_normalize_model_id(name[: -len(suffix)]))
                break
    return ids


def _parse_caps(raw: str | None) -> Cap:
    if raw is None or not raw.strip():
        return DEFAULT_CAPS
    if raw.strip().lower() in {"none", "-"}:
        return Cap.NONE
    caps = Cap.NONE
    for name in raw.split(","):
        token = name.strip().lower()
        if not token:
            continue
        if token not in CAP_NAMES:
            raise ConfigurationError(
                f"未知能力标识 {token!r}，可用值：{', '.join(CAP_NAMES)} 或 none"
            )
        caps |= CAP_NAMES[token]
    return caps


def _provider_credentials(env: dict[str, str], provider: str) -> tuple[str, str]:
    prefix = re.sub(r"[^0-9A-Za-z]+", "_", provider).upper()
    api_key = (env.get(f"{prefix}_API_KEY") or "").strip()
    base_url = (env.get(f"{prefix}_BASE_URL") or "").strip()
    if not base_url:
        base_url = KNOWN_BASE_URLS.get(provider.lower(), "")
    if not base_url:
        raise ConfigurationError(
            f"厂商 {provider} 缺少 base_url，请配置 {prefix}_BASE_URL"
        )
    if not api_key and provider.lower() not in KEYLESS_PROVIDERS:
        raise ConfigurationError(
            f"厂商 {provider} 缺少 API Key，请配置 {prefix}_API_KEY"
        )
    return base_url, api_key


class ModelRegistry:
    """模型 → 部署的映射，以及按请求解析候选降级链。"""

    def __init__(
        self,
        deployments: dict[str, Deployment],
        chain: list[str],
        *,
        enable_responses: bool,
    ) -> None:
        self._deployments = deployments
        self._chain = chain
        self._enable_responses = enable_responses

    # -- 构建 ---------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        env: dict[str, str],
        *,
        chain: list[str],
        enable_responses: bool,
    ) -> ModelRegistry:
        deployments: dict[str, Deployment] = {}
        for model_id in sorted(_collect_model_ids(env)):
            provider = (env.get(_env_key(model_id, "PROVIDER")) or "").strip()
            if not provider:
                raise ConfigurationError(
                    f"模型 {model_id} 缺少 PROVIDER，请配置 {_env_key(model_id, 'PROVIDER')}"
                )
            upstream = (env.get(_env_key(model_id, "UPSTREAM")) or "").strip() or model_id
            raw_mode = (env.get(_env_key(model_id, "MODE")) or "chat").strip().lower()
            if raw_mode not in (UpstreamMode.CHAT.value, UpstreamMode.RESPONSES.value):
                raise ConfigurationError(
                    f"模型 {model_id} 的 MODE 只能是 chat 或 responses，当前为 {raw_mode!r}"
                )
            base_url, api_key = _provider_credentials(env, provider)
            raw_timeout = (env.get(_env_key(model_id, "TIMEOUT")) or "").strip()
            try:
                timeout = float(raw_timeout) if raw_timeout else None
            except ValueError as exc:
                raise ConfigurationError(
                    f"模型 {model_id} 的 TIMEOUT 不是合法数字：{raw_timeout!r}"
                ) from exc

            deployments[model_id] = Deployment(
                model=model_id,
                provider=provider,
                upstream_model=upstream,
                mode=UpstreamMode(raw_mode),
                base_url=base_url,
                api_key=api_key,
                caps=_parse_caps(env.get(_env_key(model_id, "CAPS"))),
                timeout=timeout,
                max_tokens_field=(
                    env.get(_env_key(model_id, "MAX_TOKENS_FIELD")) or "max_tokens"
                ).strip(),
            )

        resolved_chain = chain or list(deployments)
        return cls(deployments, resolved_chain, enable_responses=enable_responses)

    # -- 查询 ---------------------------------------------------------------
    @property
    def deployments(self) -> list[Deployment]:
        """按降级链顺序返回可调用部署。"""
        return [self._deployments[m] for m in self._chain if m in self._deployments]

    @property
    def all_deployments(self) -> list[Deployment]:
        """链内优先 + 链外已注册模型，用于 /models 与就绪检查。"""
        in_chain = set(self._chain)
        ordered = [self._deployments[m] for m in self._chain if m in self._deployments]
        extra = [d for m, d in sorted(self._deployments.items()) if m not in in_chain]
        return ordered + extra

    @property
    def chain(self) -> list[str]:
        return list(self._chain)

    def get(self, model: str) -> Deployment:
        deployment = self._deployments.get(model)
        if deployment is None:
            available = ", ".join(sorted(self._deployments)) or "（空）"
            raise ModelNotFoundError(
                f"模型 {model!r} 未在注册表中配置；已注册模型：{available}"
            )
        return deployment

    # -- 校验 ---------------------------------------------------------------
    def validate(self) -> None:
        """启动期校验：宁可起不来，也不要等第一个请求才 500。"""
        if not self._deployments:
            raise ConfigurationError(
                "没有注册任何模型，请至少配置一组 MODEL_<ID>_PROVIDER 与对应厂商凭据"
            )
        missing = [m for m in self._chain if m not in self._deployments]
        if missing:
            raise ConfigurationError(
                f"MODEL_CHAIN 中的模型未注册：{', '.join(missing)}"
            )
        if not self.deployments:
            raise ConfigurationError(
                "MODEL_CHAIN 未匹配到任何已注册模型，请检查模型 ID 拼写"
            )

        duplicates = [m for m in self._chain if self._chain.count(m) > 1]
        if duplicates:
            raise ConfigurationError(
                f"MODEL_CHAIN 存在重复模型：{', '.join(sorted(set(duplicates)))}"
            )

        usable = self.deployments
        if len(usable) > 1 and usable[0].provider == usable[1].provider:
            logger.warning(
                "降级链前两个模型同属厂商 %s，厂商级故障时降级无法生效",
                usable[0].provider,
            )
        if not self._enable_responses:
            responses_only = [d.model for d in usable if d.mode is UpstreamMode.RESPONSES]
            if responses_only:
                logger.warning(
                    "ENABLE_RESPONSES_MODE=false，模型 %s 仅支持 responses 方式，"
                    "当前无法被调用",
                    ", ".join(responses_only),
                )

    # -- 解析 ---------------------------------------------------------------
    def resolve(
        self, model: str | None, requested_mode: str
    ) -> tuple[UpstreamMode, list[Deployment]]:
        """确定上游协议与候选降级链（按优先级）。

        - ``requested_mode="auto"``：沿用首模型的协议；
        - 显式指定协议时按协议过滤候选，但**不会**偷偷换掉用户点名的模型；
        - 不做 chat ↔ responses 模拟：不支持就明确报错。
        """
        if not self._deployments:
            raise ConfigurationError("没有注册任何模型")
        head_id = model or (self._chain[0] if self._chain else None)
        if not head_id:
            raise ConfigurationError("MODEL_CHAIN 为空且请求未指定模型")
        head = self.get(head_id)

        mode = head.mode if requested_mode == "auto" else UpstreamMode(requested_mode)

        if mode is UpstreamMode.RESPONSES and not self._enable_responses:
            raise UnsupportedModeError(
                "responses 方式调用当前被禁用（ENABLE_RESPONSES_MODE=false），"
                "请改用 chat 方式或启用该开关",
                code="responses_mode_disabled",
            )
        if model is not None and head.mode is not mode:
            raise UnsupportedModeError(
                f"模型 {model} 的上游协议为 {head.mode.value}，无法以 {mode.value} 方式调用"
            )

        ordered = [head_id] + [m for m in self._chain if m != head_id]
        chain: list[Deployment] = []
        seen: set[str] = set()
        for model_id in ordered:
            deployment = self._deployments.get(model_id)
            if deployment is None or model_id in seen or deployment.mode is not mode:
                continue
            seen.add(model_id)
            chain.append(deployment)
        if not chain:
            raise UnsupportedModeError(
                f"没有支持 {mode.value} 方式调用的已注册模型"
            )
        return mode, chain

    def describe(self) -> list[dict[str, Any]]:
        """供 /models 接口使用。"""
        return [
            {
                "id": deployment.model,
                "provider": deployment.provider,
                "mode": deployment.mode.value,
                "upstream_model": deployment.upstream_model,
            }
            for deployment in self.all_deployments
        ]
