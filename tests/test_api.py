"""接口与适配层测试。

全部离线：用 ``httpx.MockTransport`` 模拟上游厂商，不产生任何真实调用。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings
from src.middleware import ConcurrencyLimitMiddleware, is_llm_entry_path
from src.routes.deps import MAX_USER_ID_LENGTH, parse_user_id
from src.services.llm_adaptor import AuthenticationError
from src.services.ratelimit import InflightLimiter

BASE_ENV = {
    "DEEPSEEK_API_KEY": "sk-deepseek",
    "DEEPSEEK_BASE_URL": "https://deepseek.test/v1",
    "OPENAI_API_KEY": "sk-openai",
    "OPENAI_BASE_URL": "https://openai.test/v1",
    "MODEL_DEEPSEEK_CHAT_PROVIDER": "deepseek",
    "MODEL_DEEPSEEK_CHAT_UPSTREAM": "deepseek-chat",
    "MODEL_DEEPSEEK_CHAT_MODE": "chat",
    "MODEL_GPT_MINI_PROVIDER": "openai",
    "MODEL_GPT_MINI_UPSTREAM": "gpt-mini",
    "MODEL_GPT_MINI_MODE": "chat",
    "MODEL_GPT_RESP_PROVIDER": "openai",
    "MODEL_GPT_RESP_UPSTREAM": "gpt-resp",
    "MODEL_GPT_RESP_MODE": "responses",
}


def build_client(
    handler,
    *,
    chain: str = "deepseek-chat,gpt-mini",
    enable_responses: bool = False,
    user_id: str | None = "test-user",
    env: dict[str, str] | None = None,
    **overrides,
) -> TestClient:
    settings = Settings(
        _env_file=None,
        model_chain=chain,
        enable_responses_mode=enable_responses,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        **overrides,
    )
    merged_env = {**BASE_ENV, **(env or {})}
    app = create_app(
        settings, transport=httpx.MockTransport(handler), env=merged_env
    )
    headers = {"X-User-Id": user_id} if user_id is not None else None
    return TestClient(app, headers=headers)


def chat_ok(text: str = "hello", finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "upstream",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )


async def _sse(text: str):
    yield text.encode("utf-8")


CHAT_SSE = (
    'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
    'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"lo"},'
    '"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,'
    '"total_tokens":3}}\n\n'
    "data: [DONE]\n\n"
)


# ---------------------------------------------------------------------------
# 基础接口
# ---------------------------------------------------------------------------
def test_health_is_alive() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_database_and_providers() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["database"] == "not_configured"
    assert body["providers"] == "ok"
    assert body["models"] == 3


def test_models_lists_registry() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert ids == {"deepseek-chat", "gpt-mini", "gpt-resp"}


# ---------------------------------------------------------------------------
# 统一入口：非流式
# ---------------------------------------------------------------------------
def test_generate_non_stream_returns_text() -> None:
    client = build_client(lambda request: chat_ok("你好"))
    response = client.post("/v1/generate", json={"input": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["output"]["text"] == "你好"
    assert body["model"] == "deepseek-chat"
    assert body["provider"] == "deepseek"
    assert body["fallback"] is False
    assert body["usage"]["total_tokens"] == 3
    assert response.headers["X-Gateway-Fallback"] == "false"


def test_generate_requires_messages_or_input() -> None:
    client = build_client(lambda request: chat_ok())
    assert client.post("/v1/generate", json={}).status_code == 422


def test_generate_unknown_model_returns_404() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.post(
        "/v1/generate", json={"model": "not-exist", "input": "hi"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# responses 模式（开关 + 不做模拟）
# ---------------------------------------------------------------------------
def test_responses_mode_disabled_by_default() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.post(
        "/v1/generate", json={"input": "hi", "mode": "responses"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "responses_mode_disabled"


def test_responses_mode_rejects_chat_only_model() -> None:
    client = build_client(lambda request: chat_ok(), enable_responses=True)
    response = client.post(
        "/v1/generate",
        json={"model": "gpt-mini", "input": "hi", "mode": "responses"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_mode"


def test_responses_native_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-resp"
        assert payload["max_output_tokens"] == 16
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "model": "gpt-resp",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "from responses"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            },
        )

    client = build_client(handler, chain="gpt-resp", enable_responses=True)
    response = client.post(
        "/v1/generate",
        json={"model": "gpt-resp", "input": "hi", "mode": "responses", "max_tokens": 16},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output"]["text"] == "from responses"
    assert body["mode"] == "responses"
    assert body["usage"]["total_tokens"] == 7


# ---------------------------------------------------------------------------
# 降级与重试
# ---------------------------------------------------------------------------
def test_failover_on_5xx() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if "deepseek.test" in request.url.host:
            return httpx.Response(503, json={"error": {"message": "main down"}})
        return chat_ok("from backup")

    client = build_client(handler)
    response = client.post("/v1/generate", json={"input": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["output"]["text"] == "from backup"
    assert body["model"] == "deepseek-chat"
    assert body["upstream_model"] == "gpt-mini"
    assert body["fallback"] is True
    assert response.headers["X-Gateway-Fallback"] == "true"
    assert calls == ["deepseek.test", "openai.test"]


def test_non_retryable_error_does_not_failover() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(400, json={"error": {"message": "bad param"}})

    client = build_client(handler)
    response = client.post("/v1/generate", json={"input": "hi"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_invalid_request"
    assert calls == ["deepseek.test"]


def test_rate_limit_retries_then_failover() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if "deepseek.test" in request.url.host:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return chat_ok("from backup")

    client = build_client(handler)
    response = client.post("/v1/generate", json={"input": "hi"})
    assert response.status_code == 200
    assert response.json()["fallback"] is True
    assert calls.count("deepseek.test") == 3  # RETRY_MAX_ATTEMPTS 默认 3
    assert calls.count("openai.test") == 1


def test_all_deployments_failed_returns_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    client = build_client(handler)
    response = client.post("/v1/generate", json={"input": "hi"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "all_deployments_failed"


# ---------------------------------------------------------------------------
# 流式
# ---------------------------------------------------------------------------
def test_generate_streaming() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(CHAT_SSE),
        )

    client = build_client(handler)
    with client.stream(
        "POST", "/v1/generate", json={"input": "hi", "stream": True}
    ) as response:
        assert response.status_code == 200
        body = "".join(chunk for chunk in response.iter_text())

    assert "event: delta" in body
    assert '"text": "Hel"' in body
    assert '"text": "lo"' in body
    assert "event: done" in body
    assert '"fallback": false' in body


def test_stream_error_after_first_chunk_does_not_failover() -> None:
    calls: list[str] = []

    async def broken():
        yield (
            'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
        ).encode("utf-8")
        raise httpx.ReadError("connection reset")

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=broken(),
        )

    client = build_client(handler)
    with client.stream(
        "POST", "/v1/generate", json={"input": "hi", "stream": True}
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert '"text": "partial"' in body
    assert "event: error" in body
    assert calls == ["deepseek.test"]  # 已输出首包，不再降级


# ---------------------------------------------------------------------------
# 兼容入口
# ---------------------------------------------------------------------------
def test_chat_completions_compat() -> None:
    client = build_client(lambda request: chat_ok("compat"))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "compat"


def test_responses_endpoint_requires_flag() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.post("/v1/responses", json={"model": "gpt-resp", "input": "hi"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "responses_mode_disabled"


def test_responses_endpoint_rejects_streaming() -> None:
    client = build_client(lambda request: chat_ok(), chain="gpt-resp", enable_responses=True)
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-resp", "input": "hi", "stream": True},
    )
    assert response.status_code == 501


# ---------------------------------------------------------------------------
# 启动期校验
# ---------------------------------------------------------------------------
def test_missing_api_key_fails_fast() -> None:
    env = {
        "MODEL_BROKEN_PROVIDER": "deepseek",
        "DEEPSEEK_BASE_URL": "https://deepseek.test/v1",
    }
    with pytest.raises(Exception) as excinfo:
        create_app(Settings(_env_file=None, model_chain="broken"), env=env)
    assert "DEEPSEEK_API_KEY" in str(excinfo.value)


def test_unknown_chain_entry_fails_fast() -> None:
    with pytest.raises(Exception) as excinfo:
        create_app(
            Settings(_env_file=None, model_chain="deepseek-chat,ghost"),
            env=BASE_ENV,
        )
    assert "ghost" in str(excinfo.value)


def test_get_prompt_without_database_returns_unavailable() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.get("/v1/prompts/abc", headers={"X-User-Id": "user-1"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "prompt_store_unavailable"


@pytest.mark.parametrize(
    "path",
    ["/{api_version}/chat/completions", "/{api_version}/responses"],
)
def test_openapi_documents_models(path: str) -> None:
    client = build_client(lambda request: chat_ok())
    schema = client.get("/openapi.json").json()
    assert path in schema["paths"]


def test_openapi_documents_prompt_user_header() -> None:
    client = build_client(lambda request: chat_ok())
    schema = client.get("/openapi.json").json()
    params = schema["paths"]["/{api_version}/prompts"]["get"]["parameters"]
    assert any(p["name"] == "X-User-Id" and p["required"] for p in params)


# ---------------------------------------------------------------------------
# 入口并发限流
# ---------------------------------------------------------------------------
PAYLOAD = {"messages": [{"role": "user", "content": "hi"}]}


def build_async_app(handler, **overrides):
    settings = Settings(
        _env_file=None,
        model_chain="deepseek-chat,gpt-mini",
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        **overrides,
    )
    return create_app(settings, transport=httpx.MockTransport(handler), env=BASE_ENV)


def async_client(app, user_id: str | None = "test-user"):
    headers = {"X-User-Id": user_id} if user_id is not None else None
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.test",
        headers=headers,
    )


async def _run_asgi(app, path: str = "/v1/generate", body: bytes = b"{}"):
    """手工驱动一次 ASGI 调用，用于观察流式请求在中间件里的生命周期。"""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"gateway.test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("gateway.test", 80),
    }
    sent: list[dict] = []
    consumed = False

    async def receive() -> dict:
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()  # 之后保持连接，不再有请求体
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return status, payload


def test_concurrency_limit_returns_429_when_full() -> None:
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            entered.set()
            await finish.wait()
            return chat_ok()

        app = build_async_app(handler, max_concurrent_requests=1)
        limiter = app.state.inflight_limiter
        async with async_client(app) as client:
            first = asyncio.create_task(client.post("/v1/generate", json=PAYLOAD))
            await entered.wait()  # 首个请求已占住名额并卡在上游
            health = await client.get("/health")
            rejected = await client.post("/v1/generate", json=PAYLOAD)
            inflight_busy = limiter.inflight
            finish.set()
            ok = await first
        return ok, health, rejected, inflight_busy, limiter.inflight

    ok, health, rejected, inflight_busy, inflight_after = asyncio.run(scenario())

    assert ok.status_code == 200
    assert health.status_code == 200  # 健康检查不受限流影响
    assert inflight_busy == 1
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "gateway_concurrency_limited"
    assert rejected.headers["Retry-After"] == "1"
    assert inflight_after == 0  # 名额已归还


def test_streaming_request_holds_slot_until_body_finishes() -> None:
    """SSE 必须整条流结束后才归还名额，而不是拿到首包就归还。"""

    async def scenario():
        limiter = InflightLimiter(1)
        started, finish = asyncio.Event(), asyncio.Event()

        async def streaming_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {"type": "http.response.body", "body": b"chunk-1", "more_body": True}
            )
            started.set()
            await finish.wait()  # 模拟一条迟迟不结束的 SSE
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = ConcurrencyLimitMiddleware(streaming_app, limiter)
        task = asyncio.create_task(_run_asgi(middleware))
        await started.wait()
        inflight_while_streaming = limiter.inflight

        rejected_status, rejected_body = await _run_asgi(middleware)
        finish.set()
        await task
        return inflight_while_streaming, rejected_status, rejected_body, limiter.inflight

    (
        inflight_while_streaming,
        rejected_status,
        rejected_body,
        inflight_after,
    ) = asyncio.run(scenario())

    assert inflight_while_streaming == 1  # 流未结束，名额没有被提前归还
    assert rejected_status == 429
    assert (
        json.loads(rejected_body)["error"]["code"] == "gateway_concurrency_limited"
    )
    assert inflight_after == 0


def test_slot_released_when_request_is_cancelled() -> None:
    """客户端断开 / 任务被取消时名额必须归还。"""

    async def scenario():
        limiter = InflightLimiter(1)
        started = asyncio.Event()

        async def hanging_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            started.set()
            await asyncio.Event().wait()  # 永不结束

        middleware = ConcurrencyLimitMiddleware(hanging_app, limiter)
        task = asyncio.create_task(_run_asgi(middleware))
        await started.wait()
        assert limiter.inflight == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return limiter.inflight

    assert asyncio.run(scenario()) == 0


def test_slot_released_after_upstream_failure() -> None:
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        app = build_async_app(handler, max_concurrent_requests=1)
        limiter = app.state.inflight_limiter
        async with async_client(app) as client:
            first = await client.post("/v1/generate", json=PAYLOAD)
            second = await client.post("/v1/generate", json=PAYLOAD)
        return first, second, limiter.inflight

    first, second, inflight = asyncio.run(scenario())

    assert first.status_code == 502
    assert second.status_code == 502  # 名额已归还，所以不是 429
    assert inflight == 0


def test_concurrency_limit_can_be_disabled() -> None:
    client = build_client(lambda request: chat_ok(), max_concurrent_requests=0)
    assert client.app.state.inflight_limiter is None
    assert client.post("/v1/generate", json=PAYLOAD).status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/v1/generate",
        "/v2/generate",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/generate/",
    ],
)
def test_llm_entry_paths_are_limited(path: str) -> None:
    assert is_llm_entry_path(path)


@pytest.mark.parametrize(
    "path",
    ["/health", "/ready", "/v1/models", "/v1/prompts/abc", "/", "/v1/generate/x"],
)
def test_non_llm_paths_are_not_limited(path: str) -> None:
    assert not is_llm_entry_path(path)


# ---------------------------------------------------------------------------
# /generate 鉴权（X-User-Id）
# ---------------------------------------------------------------------------
def test_generate_rejects_missing_user_id() -> None:
    client = build_client(lambda request: chat_ok(), user_id=None)
    response = client.post("/v1/generate", json=PAYLOAD)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_user_id"


def test_generate_rejects_blank_user_id() -> None:
    client = build_client(lambda request: chat_ok(), user_id=None)
    response = client.post("/v1/generate", json=PAYLOAD, headers={"X-User-Id": "   "})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_user_id"


def test_generate_rejects_too_long_user_id() -> None:
    client = build_client(lambda request: chat_ok(), user_id=None)
    response = client.post(
        "/v1/generate",
        json=PAYLOAD,
        headers={"X-User-Id": "a" * (MAX_USER_ID_LENGTH + 1)},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_user_id"


@pytest.mark.parametrize("value", ["bad\x01user", "bad\x7fuser"])
def test_parse_user_id_rejects_control_characters(value: str) -> None:
    with pytest.raises(AuthenticationError) as excinfo:
        parse_user_id(value)
    assert excinfo.value.code == "invalid_user_id"


def test_generate_accepts_valid_user_id() -> None:
    client = build_client(lambda request: chat_ok("authed"), user_id="user-1")
    response = client.post("/v1/generate", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["output"]["text"] == "authed"


def test_generate_header_name_is_case_insensitive() -> None:
    client = build_client(lambda request: chat_ok(), user_id=None)
    response = client.post("/v1/generate", json=PAYLOAD, headers={"x-user-id": "user-2"})
    assert response.status_code == 200


def test_generate_streaming_also_requires_user_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(CHAT_SSE))

    client = build_client(handler, user_id=None)
    with client.stream(
        "POST", "/v1/generate", json={**PAYLOAD, "stream": True}
    ) as response:
        assert response.status_code == 401


def test_openapi_documents_generate_user_header() -> None:
    client = build_client(lambda request: chat_ok())
    operation = client.get("/openapi.json").json()["paths"][
        "/{api_version}/generate"
    ]["post"]
    assert any(p["name"] == "X-User-Id" for p in operation["parameters"])
    assert "401" in operation["responses"]
