"""结构化输出（JSON Schema 校验 + 有限次修复重试）测试。

全部离线：上游用 ``httpx.MockTransport`` 模拟。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from test_api import build_client, chat_ok

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
    },
    "required": ["name", "price"],
    "additionalProperties": False,
}

VALID_TEXT = '{"name": "Widget", "price": 9.9}'
MISSING_PRICE_TEXT = '{"name": "Widget"}'


def _payload(**overrides) -> dict:
    structured = {"schema": INVOICE_SCHEMA}
    structured.update(overrides)
    return {"input": "hi", "structured_output": structured}


def _chat_handler(texts: list[str], calls: list[dict], finish_reason: str = "stop"):
    """按调用次序返回 ``texts`` 里的内容，并记录上行 payload。"""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        index = min(len(calls) - 1, len(texts) - 1)
        return chat_ok(texts[index], finish_reason=finish_reason)

    return handler


# ---------------------------------------------------------------------------
# 通过校验
# ---------------------------------------------------------------------------
def test_valid_json_passes_without_retry() -> None:
    calls: list[dict] = []
    client = build_client(_chat_handler([VALID_TEXT], calls))

    response = client.post("/v1/generate", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["output"]["json"] == {"name": "Widget", "price": 9.9}
    assert body["output"]["schema_valid"] is True
    assert body["output"]["attempts"] == 1
    assert body["structured"]["valid"] is True
    assert response.headers["X-Gateway-Schema-Valid"] == "true"
    assert response.headers["X-Gateway-Schema-Attempts"] == "1"
    assert response.headers["X-Gateway-Schema-Mode"] == "json_object"
    assert len(calls) == 1
    # 模型没声明 strict_json：退化到 json_object，并靠提示词补齐约束
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in calls[0]["messages"][0]["content"]


def test_markdown_fence_is_tolerated() -> None:
    calls: list[dict] = []
    fenced = f"```json\n{VALID_TEXT}\n```"
    client = build_client(_chat_handler([fenced], calls))

    response = client.post("/v1/generate", json=_payload())

    assert response.status_code == 200
    assert response.json()["output"]["json"] == {"name": "Widget", "price": 9.9}


# ---------------------------------------------------------------------------
# 校验失败 → 修复重试
# ---------------------------------------------------------------------------
def test_invalid_output_triggers_repair_retry() -> None:
    calls: list[dict] = []
    client = build_client(
        _chat_handler([MISSING_PRICE_TEXT, VALID_TEXT], calls),
        schema_repair_max_attempts=2,
    )

    response = client.post("/v1/generate", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["output"]["attempts"] == 2
    assert body["structured"]["attempts"] == 2
    assert len(calls) == 2

    # 第二次请求把“错误输出 + 违例”回灌给模型，且基于原始消息重建
    repair = calls[1]["messages"]
    assert repair[-2]["role"] == "assistant"
    assert repair[-2]["content"] == MISSING_PRICE_TEXT
    assert repair[-1]["role"] == "user"
    assert "$.price" in repair[-1]["content"]
    assert len(repair) == len(calls[0]["messages"]) + 2

    # 默认累加所有尝试的用量，便于计费口径一致
    assert body["usage"]["total_tokens"] == 6


def test_exhausted_retries_return_error_code() -> None:
    calls: list[dict] = []
    client = build_client(
        _chat_handler([MISSING_PRICE_TEXT], calls),
        schema_repair_max_attempts=1,
    )

    response = client.post("/v1/generate", json=_payload())

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "output_schema_validation_failed"
    assert error["attempts"] == 2
    assert error["schema_name"] == "output"
    assert error["violations"]
    assert len(calls) == 2


def test_request_level_attempt_override() -> None:
    calls: list[dict] = []
    client = build_client(_chat_handler([MISSING_PRICE_TEXT], calls))

    response = client.post("/v1/generate", json=_payload(max_repair_attempts=0))

    assert response.status_code == 502
    assert response.json()["error"]["attempts"] == 1
    assert len(calls) == 1


def test_truncated_output_does_not_retry() -> None:
    calls: list[dict] = []
    client = build_client(
        _chat_handler(['{"name": "Widget"'], calls, finish_reason="length"),
        schema_repair_max_attempts=3,
    )

    response = client.post("/v1/generate", json=_payload())

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["finish_reason"] == "length"
    assert len(calls) == 1  # 截断重试大概率还是截断，不浪费预算


def test_return_raw_keeps_last_output() -> None:
    calls: list[dict] = []
    client = build_client(
        _chat_handler([MISSING_PRICE_TEXT], calls),
        schema_repair_max_attempts=0,
    )

    response = client.post("/v1/generate", json=_payload(on_invalid="return_raw"))

    assert response.status_code == 200
    body = response.json()
    assert body["output"]["text"] == MISSING_PRICE_TEXT
    assert body["output"]["schema_valid"] is False
    assert body["structured"]["violations"]
    assert response.headers["X-Gateway-Schema-Valid"] == "false"


# ---------------------------------------------------------------------------
# 客户端错误：schema 本身不合法 / 模式不支持 / 与其它字段冲突
# ---------------------------------------------------------------------------
def test_invalid_schema_returns_400() -> None:
    client = build_client(lambda request: chat_ok(VALID_TEXT))
    response = client.post(
        "/v1/generate",
        json={"input": "hi", "structured_output": {"schema": {"type": "nope"}}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_output_schema"


def test_oversized_schema_returns_400() -> None:
    client = build_client(lambda request: chat_ok(VALID_TEXT), schema_max_bytes=32)
    response = client.post("/v1/generate", json=_payload())
    assert response.status_code == 400
    assert "过大" in response.json()["error"]["message"]


def test_strict_mode_requires_capability() -> None:
    client = build_client(lambda request: chat_ok(VALID_TEXT))
    response = client.post("/v1/generate", json=_payload(mode="strict"))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_structured_output"


def test_strict_mode_passes_native_json_schema() -> None:
    calls: list[dict] = []
    client = build_client(
        _chat_handler([VALID_TEXT], calls),
        env={"MODEL_DEEPSEEK_CHAT_CAPS": "tools,json_schema,strict_json"},
    )

    response = client.post("/v1/generate", json=_payload(mode="strict"))

    assert response.status_code == 200
    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == INVOICE_SCHEMA
    # strict 依赖上游约束解码，无需再把 schema 塞进提示词
    assert all("JSON Schema" not in (m.get("content") or "") for m in calls[0]["messages"])
    assert response.headers["X-Gateway-Schema-Mode"] == "strict"


def test_conflicting_response_format_returns_400() -> None:
    client = build_client(lambda request: chat_ok(VALID_TEXT))
    response = client.post(
        "/v1/generate",
        json={**_payload(), "response_format": {"type": "json_object"}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_streaming_with_structured_output_returns_400() -> None:
    client = build_client(lambda request: chat_ok(VALID_TEXT))
    response = client.post("/v1/generate", json={**_payload(), "stream": True})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# 具名 schema 注册表
# ---------------------------------------------------------------------------
def test_registered_schema_id_is_loaded(tmp_path: Path) -> None:
    (tmp_path / "invoice.json").write_text(
        json.dumps(INVOICE_SCHEMA), encoding="utf-8"
    )
    calls: list[dict] = []
    client = build_client(
        _chat_handler([VALID_TEXT], calls), schemas_dir=str(tmp_path)
    )

    response = client.post(
        "/v1/generate",
        json={
            "input": "hi",
            "structured_output": {"schema_id": "invoice", "schema_name": "invoice"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["structured"]["schema_id"] == "invoice"
    assert body["output"]["json"]["name"] == "Widget"


def test_unknown_schema_id_returns_400() -> None:
    client = build_client(lambda request: chat_ok(VALID_TEXT))
    response = client.post(
        "/v1/generate", json={"input": "hi", "structured_output": {"schema_id": "nope"}}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_output_schema"


# ---------------------------------------------------------------------------
# 三个入口保持一致
# ---------------------------------------------------------------------------
def test_chat_completions_endpoint_validates() -> None:
    calls: list[dict] = []
    client = build_client(_chat_handler([VALID_TEXT], calls))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "structured_output": {"schema": INVOICE_SCHEMA},
        },
    )

    assert response.status_code == 200
    body = response.json()
    # 兼容入口保持字符串 content，另在 x_gateway 里给出校验摘要
    assert json.loads(body["choices"][0]["message"]["content"]) == {
        "name": "Widget",
        "price": 9.9,
    }
    assert body["x_gateway"]["structured"]["valid"] is True
    assert response.headers["X-Gateway-Schema-Attempts"] == "1"


def test_chat_completions_repairs_then_errors() -> None:
    calls: list[dict] = []
    client = build_client(
        _chat_handler([MISSING_PRICE_TEXT], calls), schema_repair_max_attempts=1
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "structured_output": {"schema": INVOICE_SCHEMA},
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "output_schema_validation_failed"


def test_responses_endpoint_validates_and_maps_text_format() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
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
                        "content": [{"type": "output_text", "text": VALID_TEXT}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )

    client = build_client(handler, chain="gpt-resp", enable_responses=True)
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-resp",
            "input": "hi",
            "structured_output": {"schema": INVOICE_SCHEMA},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["x_gateway"]["structured"]["valid"] is True
    # chat 用 response_format，responses 协议对应 text.format
    assert calls[0]["text"]["format"] == {"type": "json_object"}
    assert "response_format" not in calls[0]


def test_responses_endpoint_maps_strict_json_schema() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": VALID_TEXT}],
                    }
                ],
            },
        )

    client = build_client(
        handler,
        chain="gpt-resp",
        enable_responses=True,
        env={"MODEL_GPT_RESP_CAPS": "json_schema,strict_json"},
    )
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-resp",
            "input": "hi",
            "structured_output": {"schema": INVOICE_SCHEMA, "mode": "strict"},
        },
    )

    assert response.status_code == 200
    fmt = calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == INVOICE_SCHEMA
