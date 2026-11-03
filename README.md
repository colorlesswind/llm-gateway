# llm-gateway

统一管理各厂商模型 API 与本地 ollama 模型调用的网关。

对外只暴露**一个统一入口** `/v1/generate`，调用方不需要区分 chat completions
与 responses ；内部按模型注册表选择厂商、按降级链调度，并把两条上游协议的差异
收敛在适配层里。

## 环境与启动

本项目使用 Conda 环境 `agent-training`：

```sh
conda activate agent-training
pip install -r requirements-dev.txt
cp .env.example .env      # 填入真实 API Key
uvicorn src.main:app --reload
```

接口文档：http://127.0.0.1:8000/docs

> 配置缺失（如未填 `DEEPSEEK_API_KEY`、`MODEL_CHAIN` 里写了未注册的模型）会在
> **启动时**直接抛 `ConfigurationError`，不会等到第一个请求才失败。

## 架构

```mermaid
flowchart LR
    Client -->|POST /v1/generate| Router
    Router --> Registry[模型注册表<br/>MODEL_CHAIN]
    Registry --> Failover[降级 / 重试 / 熔断]
    Failover --> Chat[OpenAI 兼容适配器<br/>DeepSeek / Kimi / Qwen / Ollama]
    Failover --> Resp[Responses 适配器<br/>OpenAI 原生]
    Chat --> Upstream1[(厂商 API)]
    Resp --> Upstream2[(厂商 API)]
```

- 适配器只实现 `complete()` 与 `stream()` 两个方法，协议差异由 `translator` 负责；
- 新增一个 OpenAI 兼容厂商通常只需要在 `.env` 里加几行配置；
- 不做 chat → responses 的事件模拟：模型不支持就直接报错（见下文开关）。

## 配置（.env）

| 键 | 说明 |
| --- | --- |
| `MODEL_CHAIN` | 降级链，逗号分隔，靠前者优先，例如 `deepseek-chat,gpt-4o-mini` |
| `MAIN_MODEL` / `BACKUP_MODEL` | `MODEL_CHAIN` 为空时的兼容写法 |
| `ENABLE_RESPONSES_MODE` | 是否允许 responses 方式调用，默认 `false` |
| `MODEL_<ID>_PROVIDER` | 必填，对应 `<PROVIDER>_API_KEY` / `<PROVIDER>_BASE_URL` |
| `MODEL_<ID>_UPSTREAM` | 上游真实模型名，默认等于 `<ID>` |
| `MODEL_<ID>_MODE` | `chat`（默认）或 `responses` |
| `MODEL_<ID>_CAPS` | 能力位：`tools,vision,json_schema,stream_usage,strict_json` 或 `none`；`strict_json` 表示支持 OpenAI 严格 `json_schema`（约束解码） |
| `MODEL_<ID>_TIMEOUT` | 覆盖该模型的读超时（秒） |
| `MODEL_<ID>_MAX_TOKENS_FIELD` | 输出长度字段名，OpenAI 推理模型用 `max_completion_tokens` |
| `RETRY_MAX_ATTEMPTS` | 429 限流的最大尝试次数（含首次），默认 3 |
| `FIRST_TOKEN_TIMEOUT` / `IDLE_TIMEOUT` | 流式首包超时 / 事件间隔超时（秒） |
| `CIRCUIT_FAILURE_THRESHOLD` / `CIRCUIT_COOLDOWN` | 熔断阈值与冷却时间 |
| `MAX_CONCURRENT_REQUESTS` | 入口同时处理中的请求数上限（含 SSE 流），默认 10；超出返回 429，`0` 表示关闭 |
| `OVERLOAD_RETRY_AFTER` | 429 响应里 `Retry-After` 的建议等待秒数，默认 1 |
| `STRUCTURED_OUTPUT_ENABLED` | 结构化输出总开关，默认 `true`；关闭后请求带 `structured_output` 会被拒绝 |
| `SCHEMA_REPAIR_MAX_ATTEMPTS` | schema 校验失败后**除首次之外**的重试次数，默认 2（与 `RETRY_MAX_ATTEMPTS` 独立） |
| `SCHEMA_REPAIR_MAX_ERROR_CHARS` / `SCHEMA_REPAIR_MAX_OUTPUT_CHARS` | 回灌给上游的错误摘要 / 错误输出上限（字符） |
| `SCHEMA_MAX_BYTES` / `SCHEMA_MAX_DEPTH` | 请求 schema 的大小 / 嵌套深度上限，超限直接 400 |
| `SCHEMA_REPAIR_ON_LENGTH` | `finish_reason=length` 时是否仍重试，默认 `false` |
| `SCHEMA_SUM_USAGE` | 修复重试是否累加所有尝试的 `usage`，默认 `true` |
| `SCHEMAS_DIR` | 具名 schema 注册表目录（`<schema_id>.json`），默认 `schemas` |

模型 ID 中的 `-` 与 `.` 在环境变量里写作 `_`，例如 `gpt-4o-mini` →
`MODEL_GPT_4O_MINI_PROVIDER`。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/{api_version}/generate` | **统一入口**：流式 / 非流式，chat / responses 均由 `mode` 决定 |
| POST | `/{api_version}/chat/completions` | OpenAI 兼容入口（保留给现成 SDK） |
| POST | `/{api_version}/responses` | responses 兼容入口，**仅非流式**，需开启 `ENABLE_RESPONSES_MODE` |
| GET | `/{api_version}/models` | 注册表中的可用模型 |
| POST/GET | `/{api_version}/prompts` | 提示词管理（待实现，501） |
| GET | `/health` | 存活检查 |
| GET | `/ready` | 就绪检查（数据库 + 模型注册表） |

三个会打到上游模型的入口（`generate` / `chat/completions` / `responses`）统一受并发限流保护：
同时处理中的请求数（**含整条 SSE 流**）超过 `MAX_CONCURRENT_REQUESTS` 时直接返回 **429**，
响应体 `error.code = gateway_concurrency_limited` 并带 `Retry-After`；`/health`、`/ready`、
`/{api_version}/models` 不受影响。

### 统一入口示例

所有请求必须携带 `X-User-Id` 请求头作为客户端唯一标识。缺失或非法时直接返回
**401**（`error.code` 为 `missing_user_id` / `invalid_user_id`），不会触到上游，也不会计入并发限流之外的任何调用。

```jsonc
// 非流式
POST /v1/generate
X-User-Id: user-123
{
  "model": "deepseek-chat",   // 可省略，省略时使用 MODEL_CHAIN 首位
  "mode": "auto",             // auto | chat | responses
  "messages": [{"role": "user", "content": "你好"}]
}
```

> 当前只做「形状」校验（非空、长度 ≤128、无控制字符）：header 由调用方自行声明，
> **并不是可信凭证**。接入真实鉴权后应改为从 API key / JWT 中解析身份。
> 兼容入口 `/chat/completions`、`/responses` 目前尚未强制校验该头。

也可以使用 responses 风格的输入：

```jsonc
{"input": "你好", "instructions": "你是一个简洁的助手", "stream": true}
```

响应中除正文外还带有真实调用信息，便于感知是否发生降级：

```jsonc
{
  "model": "deepseek-chat",          // 请求的逻辑模型
  "upstream_model": "gpt-4o-mini",   // 实际调用的上游模型
  "provider": "openai",
  "mode": "chat",
  "fallback": true,
  "output": {"text": "...", "finish_reason": "stop"},
  "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
}
```

同时返回响应头 `X-Gateway-Model` / `X-Gateway-Provider` / `X-Gateway-Mode` /
`X-Gateway-Fallback`。

## 结构化输出（非流式 JSON Schema 校验）

`generate` / `chat/completions` / `responses` 三个入口的非流式返回都支持
`structured_output`：先校验 JSON 是否符合给定 schema，不通过时把**错误输出 +
校验违例**回灌给模型重新生成，重试预算耗尽后返回明确的错误码。

```jsonc
POST /v1/generate
X-User-Id: user-123
{
  "input": "把这张发票转成结构化数据",
  "structured_output": {
    // 二选一：内联 schema，或引用服务端具名 schema（SCHEMAS_DIR/invoice.json）
    "schema": {
      "type": "object",
      "properties": {"name": {"type": "string"}, "price": {"type": "number"}},
      "required": ["name", "price"],
      "additionalProperties": false
    },
    // "schema_id": "invoice",
    "schema_name": "invoice",   // strict 模式下上游要求的名字
    "mode": "auto",             // auto | strict | json_object | prompt_only
    "max_repair_attempts": 2,    // 可覆盖 .env 的 SCHEMA_REPAIR_MAX_ATTEMPTS
    "on_invalid": "error"        // error（默认）| return_raw
  }
}
```

`mode` 决定下发方式：

| mode | 行为 | 前提 |
| --- | --- | --- |
| `auto`（默认） | 链内有 `strict_json` 能力则走严格模式，否则退化为 `json_object`，再退化为纯提示词 | — |
| `strict` | 只允许 `strict_json` 部署，`response_format.json_schema.strict=true`；不发提示词副本 | 至少一个候选模型声明 `strict_json` |
| `json_object` | 下发 `{"type":"json_object"}`（Responses 协议为 `text.format`） | 至少一个候选模型声明 `json_schema` |
| `prompt_only` | 不下发任何 `response_format`，只把 schema 写进 system 提示词 | 无 |

与 OpenAI 原生字段的关系：`response_format` 仍然可用，但和 `structured_output`
**不能同时出现**（400 `invalid_request`）；一旦提供了 `structured_output`，网关
以它为准，并在能力不足时明确降级，而不是像以前那样静默丢弃 `response_format`。

校验重试的三个约定：

- 修复循环位于**降级层之上**：格式问题不会污染熔断计数；重试优先固定在上一次
  成功的部署，只有上游真报错时才降级；
- 回灌消息**始终基于原始消息重建**，不会随着重试次数线性膨胀；
- 重试预算与 `RETRY_MAX_ATTEMPTS`（限流退避）互相独立，避免成本相乘。

响应中除正文外会给出校验结果（`/generate`）：

```jsonc
{
  "output": {
    "text": "{\"name\": \"Widget\", \"price\": 9.9}",
    "json": {"name": "Widget", "price": 9.9},
    "schema_valid": true,
    "attempts": 2
  },
  "structured": {"schema_name": "invoice", "mode": "json_object",
                 "valid": true, "attempts": 2},
  "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
}
```

兼容入口保持协议原样：`chat/completions` 的 `content` 仍是字符串，
`responses` 的 `output` 仍是 `output_text`，校验摘要放在 `x_gateway.structured`；
三个入口都会返回 `X-Gateway-Schema-Mode` / `X-Gateway-Schema-Valid` /
`X-Gateway-Schema-Attempts` 响应头。

错误码：

| 状态 | `error.code` | 含义 |
| --- | --- | --- |
| 400 | `invalid_output_schema` | schema 本身不合法 / 过大 / 过深 / `schema_id` 未注册 |
| 400 | `unsupported_structured_output` | 报告的 `mode` 超出候选模型能力，或结构化输出已关闭 |
| 400 | `invalid_request` | `structured_output` 与 `response_format` 冲突，或用在流式请求上 |
| 502 | `output_schema_validation_failed` | 预算内始终产不出合规输出，`error` 里带 `attempts` / `violations` / `finish_reason` |

> `finish_reason=length`（输出被截断）默认**不重试**：同样预算下重试大概率还是
> 截断，应先调大 `max_tokens`。需要时用 `SCHEMA_REPAIR_ON_LENGTH=true` 打开。
>
> 校验使用 JSON Schema **draft 2020-12** 的结构约束；`format` 关键字默认不强制
> （未启用 format checker），需要格式约束请改用 `pattern` / `enum` / `minimum` 等。

### 流式事件（SSE）

统一入口使用与协议无关的事件信封：

```
event: delta
data: {"type": "delta", "text": "你"}

event: done
data: {"type": "done", "model": "deepseek-chat", "provider": "deepseek",
       "upstream_model": "deepseek-chat", "fallback": false,
       "finish_reason": "stop", "usage": {...}}

event: error
data: {"type": "error", "code": "upstream_timeout", "message": "...", "retryable": true}
```

## 降级策略

| 上游错误 | 行为 |
| --- | --- |
| 连接失败 / 超时 / 5xx | 降级到下一个候选模型 |
| 429 | 同一模型指数退避重试，最多 3 次，仍失败再降级 |
| 400 / 401 / 403 | **不降级**，直接返回（换模型只会掩盖配置或参数问题） |
| 内容审核拒答 | **不降级**，避免重复计费 |

补充约定：

- 连续失败达到阈值会触发熔断，冷却期内直接跳过该模型；
- **流式响应一旦发出首个事件就不再降级**，改为下发 `event: error`；
  首包之前的失败仍会返回真实的 HTTP 错误码；
- 降级前会按目标模型的能力位裁剪 `tools` / `response_format` 等参数；
- 结构化输出的校验重试独立于上表：它在降级层之上运行，只对“格式不合规”重试，
  不会因为格式问题把模型熔断；
- 流式调用的首包与事件间隔分别受 `FIRST_TOKEN_TIMEOUT` / `IDLE_TIMEOUT` 约束。

## 目录结构

```
src/
  app.py                      # 应用装配（注入配置、注册错误处理）
  main.py                     # uvicorn 入口
  config.py                   # Settings + .env 原始键读取
  schemas.py                  # 对外契约
  routes/
    generate.py               # 统一入口
    chat.py / responses.py    # 兼容入口
    _sse.py                   # SSE 响应构造（先取首事件再决定状态码）
    deps.py / models.py / health.py / prompts.py
  services/llm_adaptor/
    types.py                  # 内部 IR
    base.py                   # 能力位、部署描述、适配器协议
    errors.py                 # 错误分类与 HTTP 映射
    registry.py               # MODEL_<ID>_* → 部署；候选链解析
    failover.py               # 降级 / 429 退避 / 熔断
    structured.py             # JSON 提取、schema 校验、修复消息构造
    translator.py             # 协议翻译 + SSE 编码
    gateway.py                # 统一网关
    providers/
      openai_compat.py        # 所有 OpenAI 兼容厂商
      openai_responses.py     # 原生 Responses API
```

## 测试

```sh
python -m pytest -q
```

测试全部离线（`httpx.MockTransport` 打桩上游），覆盖降级、429 退避、不可重试
错误、流式中断不降级、responses 开关与启动期配置校验，以及结构化输出的
校验 / 修复重试 / 能力降级 / 错误码。


