"""LLM Gateway 对外接口的 Pydantic 契约定义。

这里只描述接口的输入与输出形状，不包含任何业务实现。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "llm-gateway"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: Literal["ok", "unavailable", "not_configured"] = "not_configured"
    providers: Literal["ok", "unavailable", "not_configured"] = "not_configured"
    models: int = 0
    detail: str | None = None


# ---------------------------------------------------------------------------
# 提示词模板注入
# ---------------------------------------------------------------------------
class PromptInjection(BaseModel):
    """在三个 LLM 入口上引用一个已上传的模板，并给出渲染参数。

    - ``target``：渲染结果写到哪里。``system`` / ``instructions`` 等价，都会注入
      system 消息（Responses 协议的 ``instructions`` 字段映射后同样是 system）；
      ``user`` 写进首条 user 消息。
    - ``position``：``prepend``（模板在前，推荐，利于上游 prompt cache）或
      ``append``。
    - ``on_missing``：变量缺失时 ``error``（默认，400）还是替换为空串。
    - ``variables`` 中出现模板未声明的 key 一律报错，避免拼写错误被静默忽略。

    **互斥**：与请求里已有的 system 内容（``instructions``、``messages`` /
    ``input`` 中的 ``role=system``）不能同时出现，否则 400
    ``prompt_instructions_conflict``。原因：不同厂商对「多条 system 消息」的处理
    不一致，网关要求 system 提示只有一个来源。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, description="模板 ID")
    version: int | None = Field(
        default=None, ge=1, description="固定使用某个版本；缺省用最新版本"
    )
    variables: dict[str, Any] = Field(
        default_factory=dict, description="模板占位符的取值"
    )
    target: Literal["system", "instructions", "user"] = "system"
    position: Literal["prepend", "append"] = "prepend"
    on_missing: Literal["error", "empty"] = "error"


# ---------------------------------------------------------------------------
# 结构化输出契约
# ---------------------------------------------------------------------------
class StructuredOutput(BaseModel):
    """非流式返回的目标 JSON 结构 + 校验/修复策略。

    两种声明方式（二选一）：

    - ``schema``：直接内联 JSON Schema（draft 2020-12）；
    - ``schema_id``：引用服务端具名 schema 注册表（``SCHEMAS_DIR/<id>.json``），
      避免每次请求携带大段 schema，也便于版本化与回滚。

    ``mode`` 决定下发方式：

    - ``strict``：只允许具备 ``strict_json`` 能力的部署，靠上游约束解码；
    - ``json_object``：只要求上游支持 ``response_format``；
    - ``prompt_only``：把 schema 写进提示词，完全不发 ``response_format``；
    - ``auto``（默认）：能 strict 就 strict，否则退到 json_object / prompt_only。
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    json_schema: dict[str, Any] | None = Field(
        default=None, alias="schema", description="内联 JSON Schema"
    )
    schema_id: str | None = Field(
        default=None,
        max_length=64,
        description="服务端注册表中的具名 schema ID（SCHEMAS_DIR/<id>.json）",
    )
    schema_name: str = Field(
        default="output", max_length=64, description="上游 strict 模式要求的 schema 名"
    )
    mode: Literal["auto", "strict", "json_object", "prompt_only"] = "auto"
    max_repair_attempts: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="除首次之外的重试次数；缺省使用 SCHEMA_REPAIR_MAX_ATTEMPTS",
    )
    on_invalid: Literal["error", "return_raw"] = Field(
        default="error",
        description="重试耗尽后：error 返回错误码；return_raw 原样返回最后一次输出",
    )

    @model_validator(mode="after")
    def _require_one_source(self) -> StructuredOutput:
        has_inline = self.json_schema is not None
        has_id = bool(self.schema_id)
        if has_inline == has_id:
            raise ValueError("schema 与 schema_id 必须且只能提供一个")
        return self


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="消息角色"
    )
    content: str | None = Field(default=None, description="消息内容")
    name: str | None = Field(default=None, description="可选的发言人名称")
    # OpenAI Chat Completions 的标准扩展字段，显式声明以获得静态类型支持
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="assistant 消息发起工具调用时的调用列表"
    )
    tool_call_id: str | None = Field(
        default=None, description="role=tool 时对应上游 tool_call 的 id"
    )
    function_call: dict[str, Any] | None = Field(
        default=None, description="已废弃的 function calling 形态，保留兼容"
    )


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(
        ..., min_length=1, description="模型标识，例如 deepseek-chat / gpt-4o / llama3"
    )
    messages: list[ChatMessage] = Field(
        ..., min_length=1, description="对话消息列表"
    )
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = Field(default=False, description="是否启用流式响应")
    structured_output: StructuredOutput | None = Field(
        default=None,
        description="非流式的 JSON Schema 校验（流式请求不支持）",
    )
    prompt: PromptInjection | None = Field(
        default=None,
        description="引用已上传的提示词模板并注入到 system 消息（与 instructions / system 消息互斥）",
    )
    # TODO: 需要补充 gateway 自有扩展字段，例如 provider 覆盖


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    x_gateway: dict[str, Any] | None = Field(
        default=None, description="网关扩展信息（结构化输出校验、降级等）"
    )


# ---------------------------------------------------------------------------
# 统一生成接口（gateway 主入口，调用方无需区分 chat / responses）
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(
        default=None, description="逻辑模型名；缺省时使用 MODEL_CHAIN 首位"
    )
    mode: Literal["auto", "chat", "responses"] = Field(
        default="auto",
        description="auto 沿用该模型的上游协议；也可显式要求 chat 或 responses",
    )
    messages: list[ChatMessage] | None = Field(
        default=None, description="与 chat completions 一致的消息列表"
    )
    input: str | list[dict[str, Any]] | None = Field(
        default=None, description="与 responses 一致的输入（文本或消息数组）"
    )
    instructions: str | None = Field(default=None, description="系统级指令")
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    stream: bool = Field(default=False, description="是否启用流式响应")
    structured_output: StructuredOutput | None = Field(
        default=None,
        description="非流式的 JSON Schema 校验（流式请求不支持）",
    )
    prompt: PromptInjection | None = Field(
        default=None,
        description="引用已上传的提示词模板并注入到 system 消息（与 instructions / system 消息互斥）",
    )

    @model_validator(mode="after")
    def _require_input(self) -> GenerateRequest:
        if not self.messages and self.input is None:
            raise ValueError("messages 与 input 至少需要提供一个")
        return self


class GenerateOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = ""
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    json_output: Any | None = Field(
        default=None, alias="json", description="校验通过后解析出的 JSON"
    )
    schema_valid: bool | None = Field(
        default=None, description="是否经过 schema 校验且通过"
    )
    attempts: int = Field(default=1, description="含首次在内的上游尝试次数")


class GenerateResponse(BaseModel):
    id: str
    object: Literal["generation"] = "generation"
    created: int
    model: str = Field(description="请求方指定的逻辑模型名")
    upstream_model: str = Field(description="实际调用的上游模型名")
    provider: str
    mode: Literal["chat", "responses"]
    fallback: bool = Field(default=False, description="是否因降级使用了备选模型")
    output: GenerateOutput
    usage: Usage = Field(default_factory=Usage)
    structured: dict[str, Any] | None = Field(
        default=None, description="结构化输出校验摘要（schema/mode/attempts/violations）"
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., min_length=1, description="模型标识")
    input: str | list[dict[str, Any]] = Field(
        ..., description="输入文本，或与 Responses API 一致的消息数组"
    )
    instructions: str | None = Field(default=None, description="系统级指令")
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = Field(default=False, description="是否启用流式响应")
    structured_output: StructuredOutput | None = Field(
        default=None,
        description="非流式的 JSON Schema 校验（本入口只支持非流式）",
    )
    prompt: PromptInjection | None = Field(
        default=None,
        description="引用已上传的提示词模板并注入到 system 消息（与 instructions / system 消息互斥）",
    )


class ResponseOutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    id: str | None = None
    status: str | None = None
    content: list[dict[str, Any]] | None = None


class ResponseObject(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: str
    output: list[ResponseOutputItem] = Field(default_factory=list)
    usage: Usage | None = None
    metadata: dict[str, Any] | None = Field(
        default=None, description="网关扩展元信息（provider / upstream_model / fallback）"
    )
    x_gateway: dict[str, Any] | None = Field(
        default=None, description="网关扩展信息（结构化输出校验等）"
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str
    # TODO: 标记来源厂商或本地 ollama
    provider: str | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts（模板 CRUD）
# ---------------------------------------------------------------------------
class PromptCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    content: str = Field(
        ...,
        min_length=1,
        # 字符级硬上限，挡住明显的超大请求；字节级上限由服务层按配置校验
        max_length=1_000_000,
        description="提示词正文，占位符写作 {{name}}",
    )
    description: str | None = Field(default=None, max_length=512)
    enabled: bool = Field(default=True, description="是否允许被请求引用")
    variables: list[str] | None = Field(
        default=None,
        max_length=64,
        description="可选的变量声明；提供时必须与模板中的占位符集合一致",
    )

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("模板名称不能为空")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
            raise ValueError("模板名称不能包含控制字符")
        return name


class PromptUpdateRequest(BaseModel):
    """部分更新；``content`` 变化时会生成新版本（旧版本仍可按 version 引用）。"""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    content: str | None = Field(default=None, min_length=1, max_length=1_000_000)
    description: str | None = Field(default=None, max_length=512)
    enabled: bool | None = None
    variables: list[str] | None = Field(default=None, max_length=64)
    expected_version: int | None = Field(
        default=None,
        ge=1,
        description="乐观锁：与当前 head_version 不一致时返回 409",
    )

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.strip()
        if not name:
            raise ValueError("模板名称不能为空")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
            raise ValueError("模板名称不能包含控制字符")
        return name


class Prompt(BaseModel):
    """模板详情（含正文）。"""

    id: str
    user_id: str
    name: str
    content: str
    description: str | None = None
    enabled: bool = True
    version: int = 1
    content_hash: str = ""
    variables: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    created_at: datetime
    updated_at: datetime


class PromptSummary(BaseModel):
    """列表项，**不含正文**，避免列表接口返回大量长文本。"""

    id: str
    user_id: str
    name: str
    description: str | None = None
    enabled: bool = True
    version: int = 1
    content_hash: str = ""
    variables: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    created_at: datetime
    updated_at: datetime


class PromptListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[PromptSummary] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0

