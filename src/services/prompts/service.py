"""模板业务层：校验、配额、渲染编排与进程内缓存。"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Any

from src.config import Settings
from src.schemas import (
    Prompt,
    PromptCreateRequest,
    PromptInjection,
    PromptSummary,
    PromptUpdateRequest,
)
from src.services.prompts import render as renderer
from src.services.prompts.errors import (
    PromptDisabled,
    PromptNotFound,
    PromptQuotaExceeded,
    PromptTooLarge,
    PromptVariablesInvalid,
)
from src.services.prompts.store import SqlPromptStore


@dataclass(frozen=True)
class RenderedPrompt:
    """一次渲染的可观测信息（**不含正文**，可直接进日志/响应头）。"""

    prompt_id: str
    version: int
    name: str
    content_hash: str
    rendered_bytes: int
    token_estimate: int
    variables: list[str]
    target: str
    position: str


def _record_to_prompt(record: dict[str, Any]) -> Prompt:
    return Prompt(
        id=record["id"],
        user_id=record["user_id"],
        name=record["name"],
        content=record["content"],
        description=record["description"],
        enabled=record["enabled"],
        version=record["version"],
        content_hash=record["content_hash"],
        variables=record["variables"],
        token_estimate=record["token_estimate"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


def _record_to_summary(record: dict[str, Any]) -> PromptSummary:
    return PromptSummary(
        id=record["id"],
        user_id=record["user_id"],
        name=record["name"],
        description=record["description"],
        enabled=record["enabled"],
        version=record["version"],
        content_hash=record["content_hash"],
        variables=record["variables"],
        token_estimate=record["token_estimate"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


class PromptService:
    """模板的读写与渲染入口。"""

    def __init__(self, store: SqlPromptStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        # key: (user_id, prompt_id, version or 0) -> (expires_at, Prompt)
        self._cache: OrderedDict[tuple[str, str, int], tuple[float, Prompt]] = (
            OrderedDict()
        )
        # 缓存正文的总字节数，用于按字节预算淘汰
        self._cache_bytes = 0

    # -- 生命周期 ---------------------------------------------------------
    async def initialize(self) -> None:
        """建表（幂等）。失败时抛 ``PromptStoreUnavailable``，由调用方决定是否降级。"""
        await self._store.ensure_schema()

    async def ping(self) -> bool:
        return await self._store.ping()

    async def aclose(self) -> None:
        await self._store.aclose()

    # -- CRUD -------------------------------------------------------------
    async def create(self, user_id: str, payload: PromptCreateRequest) -> Prompt:
        limit = self._settings.prompt_max_per_user
        if limit > 0 and await self._store.count(user_id) >= limit:
            raise PromptQuotaExceeded(
                f"模板数量已达上限（{limit}）", details={"limit": limit}
            )

        version_data = self._prepare(payload.content)
        self._check_declared(payload.variables, version_data["variables"])
        record = await self._store.create(
            user_id=user_id,
            prompt_id=f"ptpl_{uuid.uuid4().hex}",
            name=payload.name,
            description=payload.description,
            enabled=payload.enabled,
            version_data=version_data,
        )
        return _record_to_prompt(record)

    async def get(
        self, user_id: str, prompt_id: str, *, version: int | None = None
    ) -> Prompt:
        record = await self._store.get(user_id, prompt_id, version=version)
        if record is None:
            # 不存在与不属于当前用户统一 404，避免探测他人模板
            raise PromptNotFound(f"模板不存在：{prompt_id}")
        return _record_to_prompt(record)

    async def list(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        enabled: bool | None = None,
    ) -> tuple[list[PromptSummary], int]:
        records, total = await self._store.list(
            user_id, limit=limit, offset=offset, enabled=enabled
        )
        return [_record_to_summary(item) for item in records], total

    async def update(
        self, user_id: str, prompt_id: str, payload: PromptUpdateRequest
    ) -> Prompt:
        provided = payload.model_fields_set
        changes: dict[str, Any] = {}
        version_data: dict[str, Any] | None = None

        if "name" in provided and payload.name is not None:
            changes["name"] = payload.name
        if "description" in provided:
            changes["description"] = payload.description
        if "enabled" in provided and payload.enabled is not None:
            changes["enabled"] = payload.enabled
        if payload.content is not None:
            version_data = self._prepare(payload.content)

        if payload.variables is not None:
            if version_data is not None:
                declared = version_data["variables"]
            else:
                current = await self.get(user_id, prompt_id)
                declared = current.variables
            self._check_declared(payload.variables, declared)

        record = await self._store.update(
            user_id,
            prompt_id,
            changes=changes,
            version_data=version_data,
            expected_version=payload.expected_version,
        )
        if record is None:
            raise PromptNotFound(f"模板不存在：{prompt_id}")
        # 更新后本进程立刻读到新版本；其他 worker 最多在 TTL 后生效
        self._invalidate(user_id, prompt_id)
        return _record_to_prompt(record)

    async def delete(self, user_id: str, prompt_id: str) -> None:
        deleted = await self._store.delete(user_id, prompt_id)
        if not deleted:
            raise PromptNotFound(f"模板不存在：{prompt_id}")
        self._invalidate(user_id, prompt_id)

    # -- 渲染 -------------------------------------------------------------
    async def resolve(
        self, user_id: str, injection: PromptInjection
    ) -> tuple[str, RenderedPrompt]:
        prompt = await self._load(user_id, injection.id, injection.version)
        if not prompt.enabled:
            raise PromptDisabled(f"模板已停用：{prompt.id}")

        text = renderer.render_template(
            prompt.content, injection.variables, on_missing=injection.on_missing
        )
        size = renderer.byte_length(text)
        limit = self._settings.prompt_max_rendered_bytes
        if limit > 0 and size > limit:
            raise PromptTooLarge(
                f"渲染后的提示词超过上限（{size} > {limit} 字节）",
                details={"rendered_bytes": size, "limit": limit},
            )

        info = RenderedPrompt(
            prompt_id=prompt.id,
            version=prompt.version,
            name=prompt.name,
            content_hash=prompt.content_hash,
            rendered_bytes=size,
            token_estimate=renderer.estimate_tokens(text),
            variables=sorted(injection.variables),
            target=injection.target,
            position=injection.position,
        )
        return text, info

    # -- 内部 -------------------------------------------------------------
    async def _load(
        self, user_id: str, prompt_id: str, version: int | None
    ) -> Prompt:
        ttl = self._settings.prompt_cache_ttl
        key = (user_id, prompt_id, version or 0)

        if ttl > 0:
            cached = self._cache.get(key)
            if cached is not None:
                expires_at, prompt = cached
                if expires_at > monotonic():
                    self._cache.move_to_end(key)
                    return prompt
                self._drop(key)

        record = await self._store.get(user_id, prompt_id, version=version)
        if record is None:
            raise PromptNotFound(f"模板不存在：{prompt_id}")
        prompt = _record_to_prompt(record)

        if ttl > 0:
            self._put(key, prompt)
        return prompt

    def _invalidate(self, user_id: str, prompt_id: str) -> None:
        for key in [k for k in self._cache if k[:2] == (user_id, prompt_id)]:
            self._drop(key)

    def _put(self, key: tuple[str, str, int], prompt: Prompt) -> None:
        ttl = self._settings.prompt_cache_ttl
        self._cache[key] = (monotonic() + ttl, prompt)
        self._cache_bytes += renderer.byte_length(prompt.content)
        self._cache.move_to_end(key)

        max_entries = max(1, self._settings.prompt_cache_max_entries)
        max_bytes = max(0, self._settings.prompt_cache_max_bytes)
        while self._cache and (
            len(self._cache) > max_entries
            or (max_bytes > 0 and self._cache_bytes > max_bytes)
        ):
            oldest = next(iter(self._cache))
            self._drop(oldest)

    def _drop(self, key: tuple[str, str, int]) -> None:
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache_bytes -= renderer.byte_length(cached[1].content)

    def _prepare(self, content: str) -> dict[str, Any]:
        """上传/更新时统一做的正文校验与派生字段计算。"""
        size = renderer.byte_length(content)
        limit = self._settings.prompt_max_bytes
        if limit > 0 and size > limit:
            raise PromptTooLarge(
                f"模板正文超过上限（{size} > {limit} 字节）",
                details={"content_bytes": size, "limit": limit},
            )
        variables = renderer.validate_template(content)
        return {
            "content": content,
            "content_bytes": size,
            "content_hash": renderer.content_hash(content),
            "variables": variables,
            "token_estimate": renderer.estimate_tokens(content),
        }

    @staticmethod
    def _check_declared(
        declared: list[str] | None, actual: list[str]
    ) -> None:
        if declared is None:
            return
        declared_set = {item.strip() for item in declared if item.strip()}
        actual_set = set(actual)
        if declared_set == actual_set:
            return
        raise PromptVariablesInvalid(
            "声明的变量与模板中的占位符不一致",
            details={
                "missing_declaration": sorted(actual_set - declared_set),
                "undeclared": sorted(declared_set - actual_set),
            },
        )


__all__ = ["PromptService", "RenderedPrompt"]
