"""提示词模板：CRUD、渲染注入、隔离性与边界测试。

全部离线：上游用 ``httpx.MockTransport`` 模拟，存储用内存 SQLite。
"""

from __future__ import annotations

import json

import httpx

from test_api import build_client, chat_ok

DB = "sqlite:///:memory:"


def _create(client, **overrides):
    body = {
        "name": overrides.pop("name", "greet"),
        "content": overrides.pop("content", "你是一个助手。\n\n主题：{{topic}}"),
        **overrides,
    }
    return client.post("/v1/prompts", json=body)


def _seed(client, **overrides) -> str:
    response = _create(client, **overrides)
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def test_create_prompt_returns_id_and_parsed_variables() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    response = _create(client)
    assert response.status_code == 201
    body = response.json()
    assert body["id"].startswith("ptpl_")
    assert body["user_id"] == "test-user"
    assert body["version"] == 1
    assert body["variables"] == ["topic"]
    assert len(body["content_hash"]) == 64
    assert body["token_estimate"] > 0


def test_list_returns_summaries_without_content() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    detail = client.get(f"/v1/prompts/{prompt_id}").json()
    assert detail["content"].startswith("你是一个助手")

    listed = client.get("/v1/prompts").json()
    assert listed["total"] == 1
    assert listed["data"][0]["id"] == prompt_id
    assert "content" not in listed["data"][0]


def test_prompt_is_isolated_between_users() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    response = client.get(
        f"/v1/prompts/{prompt_id}", headers={"X-User-Id": "other-user"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "prompt_not_found"

    listed = client.get("/v1/prompts", headers={"X-User-Id": "other-user"}).json()
    assert listed["total"] == 0


def test_duplicate_name_conflicts() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    _seed(client)
    response = _create(client)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "prompt_name_conflict"


def test_prompt_quota_is_enforced() -> None:
    client = build_client(
        lambda request: chat_ok(), database_url=DB, prompt_max_per_user=1
    )
    _seed(client, name="one")
    response = _create(client, name="two")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "prompt_quota_exceeded"


def test_oversized_template_rejected() -> None:
    client = build_client(
        lambda request: chat_ok(), database_url=DB, prompt_max_bytes=32
    )
    response = _create(client, content="x" * 64)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "prompt_too_large"


def test_declared_variables_must_match_template() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    response = _create(client, variables=["other"])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prompt_variables_invalid"

    ok = _create(client, name="ok", variables=["topic"])
    assert ok.status_code == 201


def test_update_bumps_version_and_keeps_history() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    updated = client.patch(
        f"/v1/prompts/{prompt_id}", json={"content": "新版 {{topic}}"}
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert updated.json()["content"] == "新版 {{topic}}"

    old = client.get(f"/v1/prompts/{prompt_id}?version=1").json()
    assert old["content"].startswith("你是一个助手")
    assert old["version"] == 1


def test_update_rejects_stale_expected_version() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)
    client.patch(f"/v1/prompts/{prompt_id}", json={"content": "v2 {{topic}}"})

    response = client.patch(
        f"/v1/prompts/{prompt_id}",
        json={"content": "v3 {{topic}}", "expected_version": 1},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "prompt_version_conflict"


def test_delete_is_soft_and_frees_name() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    deleted = client.delete(f"/v1/prompts/{prompt_id}")
    assert deleted.status_code == 204
    assert client.get(f"/v1/prompts/{prompt_id}").status_code == 404
    assert client.get("/v1/prompts").json()["total"] == 0

    assert _create(client).status_code == 201


# ---------------------------------------------------------------------------
# 注入到三个入口
# ---------------------------------------------------------------------------
def test_generate_injects_rendered_prompt_as_system_message() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok("ok")

    client = build_client(handler, database_url=DB)
    prompt_id = _seed(client)

    response = client.post(
        "/v1/generate",
        json={
            "input": "hi",
            "prompt": {"id": prompt_id, "variables": {"topic": "量子计算"}},
        },
    )
    assert response.status_code == 200
    payload = seen[0]
    # 模板是唯一的 system 来源：上游只会收到一条 system 消息
    system_messages = [m for m in payload["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "你是一个助手。\n\n主题：量子计算"
    # 模板字段绝不能透传给上游
    assert "prompt" not in payload
    assert response.headers["X-Gateway-Prompt-Id"] == prompt_id
    assert response.headers["X-Gateway-Prompt-Version"] == "1"


def test_prompt_conflicts_with_instructions() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    response = client.post(
        "/v1/generate",
        json={
            "input": "hi",
            "instructions": "回答要简洁",
            "prompt": {"id": prompt_id, "variables": {"topic": "T"}},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "prompt_instructions_conflict"
    assert error["conflict_source"] == "instructions"


def test_prompt_conflicts_with_system_message() -> None:
    # 最常见的 OpenAI SDK 写法：system 放在 messages 里
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个翻译"},
                {"role": "user", "content": "hi"},
            ],
            "prompt": {"id": prompt_id, "variables": {"topic": "T"}},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "prompt_instructions_conflict"
    assert error["conflict_source"] == "messages"


def test_prompt_conflicts_with_system_input_item() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)

    response = client.post(
        "/v1/generate",
        json={
            "input": [
                {"role": "system", "content": "你是一个翻译"},
                {"role": "user", "content": "hi"},
            ],
            "prompt": {"id": prompt_id, "variables": {"topic": "T"}},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "prompt_instructions_conflict"
    assert error["conflict_source"] == "input"



def test_template_keeps_literal_json_braces() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok("ok")

    client = build_client(handler, database_url=DB)
    prompt_id = _seed(
        client,
        content='只输出 JSON，例如 {"ok": true}。主题：{{topic}}',
    )
    response = client.post(
        "/v1/generate",
        json={"input": "hi", "prompt": {"id": prompt_id, "variables": {"topic": "T"}}},
    )
    assert response.status_code == 200
    assert '{"ok": true}' in seen[0]["messages"][0]["content"]


def test_missing_variable_returns_400() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client)
    response = client.post(
        "/v1/generate", json={"input": "hi", "prompt": {"id": prompt_id}}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "prompt_variables_invalid"
    assert error["missing_variables"] == ["topic"]


def test_unknown_variable_returns_400() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    prompt_id = _seed(client, content="没有占位符")
    response = client.post(
        "/v1/generate",
        json={"input": "hi", "prompt": {"id": prompt_id, "variables": {"tpoic": "x"}}},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "prompt_variables_invalid"
    assert error["unknown_variables"] == ["tpoic"]


def test_on_missing_empty_renders_blank() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok("ok")

    client = build_client(handler, database_url=DB)
    prompt_id = _seed(client, content="前缀[{{topic}}]后缀")
    response = client.post(
        "/v1/generate",
        json={
            "input": "hi",
            "prompt": {"id": prompt_id, "on_missing": "empty"},
        },
    )
    assert response.status_code == 200
    assert "前缀[]后缀" in seen[0]["messages"][0]["content"]


def test_pinned_version_renders_old_content() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok("ok")

    client = build_client(handler, database_url=DB)
    prompt_id = _seed(client, content="v1 {{topic}}")
    client.patch(f"/v1/prompts/{prompt_id}", json={"content": "v2 {{topic}}"})

    client.post(
        "/v1/generate",
        json={
            "input": "hi",
            "prompt": {"id": prompt_id, "version": 1, "variables": {"topic": "T"}},
        },
    )
    assert seen[-1]["messages"][0]["content"] == "v1 T"


def test_target_user_appends_to_first_user_message() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok("ok")

    client = build_client(handler, database_url=DB)
    prompt_id = _seed(client, content="补充：{{topic}}")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "问题"}],
            "prompt": {
                "id": prompt_id,
                "variables": {"topic": "T"},
                "target": "user",
                "position": "append",
            },
        },
    )
    assert response.status_code == 200
    payload = seen[0]
    assert payload["messages"][0] == {
        "role": "user",
        "content": "问题\n\n补充：T",
    }
    assert "prompt" not in payload
    assert response.headers["X-Gateway-Prompt-Id"] == prompt_id


def test_chat_completions_requires_user_id_when_using_prompt() -> None:
    handler = lambda request: chat_ok("ok")  # noqa: E731
    client = build_client(handler, database_url=DB)
    prompt_id = _seed(client)

    anonymous = build_client(handler, database_url=DB, user_id=None)
    # 匿名客户端看不到别人的模板；这里用同一进程的另一个 app 验证鉴权先行
    response = anonymous.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": {"id": prompt_id, "variables": {"topic": "T"}},
        },
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_user_id"


def test_responses_endpoint_injects_into_instructions() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
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
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        )

    client = build_client(
        handler, chain="gpt-resp", enable_responses=True, database_url=DB
    )
    prompt_id = _seed(client, content="角色：{{role}}")

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-resp",
            "input": "hi",
            "prompt": {"id": prompt_id, "variables": {"role": "编辑"}},
        },
    )
    assert response.status_code == 200
    assert "角色：编辑" in seen[0]["instructions"]
    assert "prompt" not in seen[0]


def test_streaming_response_exposes_prompt_headers() -> None:
    sse = (
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"lo"},'
        '"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode()
        )

    client = build_client(handler, database_url=DB)
    prompt_id = _seed(client)

    with client.stream(
        "POST",
        "/v1/generate",
        json={
            "input": "hi",
            "stream": True,
            "prompt": {"id": prompt_id, "variables": {"topic": "T"}},
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
        assert response.headers["X-Gateway-Prompt-Id"] == prompt_id

    assert "event: delta" in body


# ---------------------------------------------------------------------------
# 存储未配置 / 就绪检查
# ---------------------------------------------------------------------------
def test_generate_with_prompt_without_database_returns_503() -> None:
    client = build_client(lambda request: chat_ok())
    response = client.post(
        "/v1/generate", json={"input": "hi", "prompt": {"id": "ptpl_x"}}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "prompt_store_unavailable"


def test_generate_without_prompt_still_works_without_database() -> None:
    client = build_client(lambda request: chat_ok("ok"))
    assert client.post("/v1/generate", json={"input": "hi"}).status_code == 200


def test_ready_reports_database_ok_when_configured() -> None:
    client = build_client(lambda request: chat_ok(), database_url=DB)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["database"] == "ok"
