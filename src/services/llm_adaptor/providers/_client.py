"""上游 HTTP 客户端封装：超时、鉴权头、错误映射、SSE 解析。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from src.services.llm_adaptor.base import Deployment, join_url
from src.services.llm_adaptor.errors import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    error_from_status,
)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class UpstreamClient:
    """按厂商复用的 ``httpx.AsyncClient``（连接池共享，不要每次请求新建）。"""

    def __init__(
        self,
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._default_timeout = httpx.Timeout(
            read_timeout, connect=connect_timeout, write=connect_timeout, pool=connect_timeout
        )
        self._client = httpx.AsyncClient(
            timeout=self._default_timeout,
            transport=transport,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _timeout(self, deployment: Deployment) -> httpx.Timeout:
        if deployment.timeout is None:
            return self._default_timeout
        return httpx.Timeout(
            deployment.timeout,
            connect=self._default_timeout.connect,
            write=self._default_timeout.write,
            pool=self._default_timeout.pool,
        )

    @staticmethod
    def _headers(deployment: Deployment) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if deployment.api_key:
            headers["Authorization"] = f"Bearer {deployment.api_key}"
        headers.update(deployment.extra_headers)
        return headers

    async def post_json(
        self, deployment: Deployment, path: str, payload: dict
    ) -> dict:
        url = join_url(deployment.base_url, path)
        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=self._headers(deployment),
                timeout=self._timeout(deployment),
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"请求 {url} 超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderConnectionError(f"连接 {url} 失败：{exc}") from exc

        if response.status_code >= 400:
            raise error_from_status(
                response.status_code,
                response.text,
                retry_after=_retry_after(response),
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                "上游返回了非 JSON 响应", upstream_status=response.status_code
            ) from exc

    @asynccontextmanager
    async def open_stream(
        self, deployment: Deployment, path: str, payload: dict
    ) -> AsyncIterator[httpx.Response]:
        url = join_url(deployment.base_url, path)
        headers = self._headers(deployment)
        headers["Accept"] = "text/event-stream"
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout(deployment),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise error_from_status(
                        response.status_code,
                        body,
                        retry_after=_retry_after(response),
                    )
                yield response
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"连接 {url} 超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderConnectionError(f"连接 {url} 失败：{exc}") from exc


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """解析 SSE：产出 ``(event 名, data 内容)``。"""
    event_name: str | None = None
    data_lines: list[str] = []
    async for raw in response.aiter_lines():
        line = raw.rstrip("\r")
        if not line.strip():
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def loads(data: str) -> dict | None:
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
