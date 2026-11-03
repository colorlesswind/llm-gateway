"""结构化输出：schema 解析、JSON 提取、校验与修复消息构造。

职责边界：

- 这里只做**纯函数**级别的工作（不碰网络、不碰 failover）；
- “校验不通过 → 有限次重试”的编排在 :mod:`src.services.llm_adaptor.gateway`；
- 能力不足时**不静默降级**：``mode=strict`` 找不到支持严格模式的部署直接 400。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator

from src.config import Settings
from src.schemas import StructuredOutput
from src.services.llm_adaptor.base import Cap, Deployment
from src.services.llm_adaptor.errors import (
    InvalidOutputSchema,
    UnsupportedStructuredOutput,
)
from src.services.llm_adaptor.types import Message

# 具名 schema 的 ID 规则：既能当文件名，又不允许路径穿越
SCHEMA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# 一次修复提示里最多回灌多少条违例（再多对模型也没帮助，只会烧 token）
MAX_VIOLATIONS = 8
_CODE_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*(?P<body>.*?)\s*```$", re.DOTALL)

StructuredMode = Literal["strict", "json_object", "prompt_only"]


@dataclass(frozen=True)
class StructuredSpec:
    """一次请求的、已解析并校验过的结构化输出契约。"""

    name: str
    schema: dict[str, Any]
    mode: StructuredMode  # strict | json_object | prompt_only
    max_attempts: int  # 含首次
    on_invalid: str = "error"
    schema_id: str | None = None
    source: str = "inline"
    inject_prompt: bool = True

    @property
    def response_format(self) -> dict[str, Any] | None:
        """要下发给上游的 ``response_format``；``prompt_only`` 时不发。"""
        if self.mode == "prompt_only":
            return None
        if self.mode == "strict":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.name,
                    "schema": self.schema,
                    "strict": True,
                },
            }
        return {"type": "json_object"}

    @property
    def prompt_text(self) -> str:
        schema_text = json.dumps(self.schema, ensure_ascii=False, separators=(",", ":"))
        return (
            "你必须只输出一个 JSON 值，且严格符合下面的 JSON Schema。\n"
            "不要输出 Markdown 代码块、注释或任何解释文字。\n\n"
            f"JSON Schema:\n{schema_text}"
        )


# ---------------------------------------------------------------------------
# 解析与校验入口
# ---------------------------------------------------------------------------
def prepare_structured(
    raw: StructuredOutput | None,
    *,
    settings: Settings,
    deployments: Sequence[Deployment],
) -> StructuredSpec | None:
    """把对外契约解析成可执行的 :class:`StructuredSpec`。

    所有“客户端可修复”的问题都在这里变成 400，避免把坏请求打到上游。
    """
    if raw is None:
        return None
    if not settings.structured_output_enabled:
        raise UnsupportedStructuredOutput(
            "结构化输出已被网关关闭（STRUCTURED_OUTPUT_ENABLED=false）"
        )

    schema, source = _resolve_schema(raw, settings)
    _guard_schema(schema, settings)

    mode = _resolve_mode(raw.mode, deployments)
    extra = (
        raw.max_repair_attempts
        if raw.max_repair_attempts is not None
        else settings.schema_repair_max_attempts
    )
    max_attempts = 1 + max(0, int(extra))

    return StructuredSpec(
        name=raw.schema_name,
        schema=schema,
        mode=mode,
        max_attempts=max_attempts,
        on_invalid=raw.on_invalid,
        schema_id=raw.schema_id,
        source=source,
        # strict 依赖上游约束解码，不需要再把 schema 塞进提示词（省 token）
        inject_prompt=raw.mode != "strict",
    )


def _resolve_schema(
    raw: StructuredOutput, settings: Settings
) -> tuple[dict[str, Any], str]:
    if raw.json_schema is not None:
        return raw.json_schema, "inline"
    assert raw.schema_id is not None  # 已由 StructuredOutput 校验
    return _load_registered_schema(raw.schema_id, settings), "registry"


def _load_registered_schema(schema_id: str, settings: Settings) -> dict[str, Any]:
    if not SCHEMA_ID_PATTERN.match(schema_id):
        raise InvalidOutputSchema(
            f"schema_id 非法：{schema_id!r}（只允许字母、数字、_ . -，且不以符号开头）"
        )
    base = Path(settings.schemas_dir)
    path = base / f"{schema_id}.json"
    if not path.is_file():
        raise InvalidOutputSchema(f"schema_id {schema_id!r} 未注册（期望文件 {path}）")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InvalidOutputSchema(
            f"schema_id {schema_id!r} 的文件无法解析：{exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise InvalidOutputSchema(f"schema_id {schema_id!r} 的顶层必须是 JSON 对象")
    return loaded


def _guard_schema(schema: dict[str, Any], settings: Settings) -> None:
    """大小/深度/合法性三道闸门，全部按“客户端问题”返回 400。"""
    raw = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    size = len(raw.encode("utf-8"))
    if size > settings.schema_max_bytes:
        raise InvalidOutputSchema(
            f"JSON Schema 过大（{size} 字节，上限 {settings.schema_max_bytes}）"
        )
    depth = _depth(schema)
    if depth > settings.schema_max_depth:
        raise InvalidOutputSchema(
            f"JSON Schema 嵌套过深（{depth} 层，上限 {settings.schema_max_depth}）"
        )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        location = "/".join(str(part) for part in exc.absolute_schema_path)
        raise InvalidOutputSchema(
            f"不是合法的 JSON Schema（{location or '#'}）：{exc.message}"
        ) from exc


def _depth(value: Any, level: int = 0) -> int:
    if isinstance(value, dict):
        return max((_depth(v, level + 1) for v in value.values()), default=level)
    if isinstance(value, list):
        return max((_depth(v, level + 1) for v in value), default=level)
    return level


def _resolve_mode(requested: str, deployments: Sequence[Deployment]) -> StructuredMode:
    strict_capable = any(d.caps & Cap.STRICT_JSON for d in deployments)
    json_capable = any(d.caps & Cap.JSON_SCHEMA for d in deployments)

    if requested == "strict":
        if not strict_capable:
            models = ", ".join(d.model for d in deployments) or "（空）"
            raise UnsupportedStructuredOutput(
                "mode=strict 需要上游支持严格 json_schema，但候选模型均未声明 "
                f"strict_json 能力：{models}"
            )
        return "strict"
    if requested == "json_object":
        if not json_capable:
            raise UnsupportedStructuredOutput(
                "mode=json_object 需要上游支持 response_format，但候选模型均未声明 "
                "json_schema 能力"
            )
        return "json_object"
    if requested == "prompt_only":
        return "prompt_only"

    # auto：按能力从强到弱退化
    if strict_capable:
        return "strict"
    if json_capable:
        return "json_object"
    return "prompt_only"


# ---------------------------------------------------------------------------
# 输出提取与校验
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any:
    """从模型输出里取出 JSON 值。

    容忍常见“脏输出”：```json 代码块、前后夹带解释文字、字符串里的裸换行。
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("输出为空")

    fenced = _CODE_FENCE.match(raw)
    candidate = fenced.group("body").strip() if fenced else raw

    try:
        return json.loads(candidate, strict=False)
    except ValueError:
        pass

    decoder = json.JSONDecoder(strict=False)
    for index, char in enumerate(candidate):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except ValueError:
            continue
        return value
    raise ValueError("输出不是合法的 JSON")


@lru_cache(maxsize=128)
def _validator(canonical_schema: str) -> Validator:
    return Draft202012Validator(json.loads(canonical_schema))


def validate_instance(value: Any, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """返回违例摘要列表；空列表表示通过。"""
    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    validator = _validator(canonical)
    violations: list[dict[str, Any]] = []
    for error in sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path)):
        violations.append({"path": _error_path(error), "reason": error.message})
        if len(violations) >= MAX_VIOLATIONS:
            break
    return violations


def _error_path(error: Any) -> str:
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    # required 校验的错误路径停在父节点，"缺了哪个字段"只能从 validator_value 得到；
    # 把它补进路径，模型才能直接定位该补什么。
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = [name for name in error.validator_value if name not in error.instance]
        for name in missing:
            path += f".{name}"
    return path


# ---------------------------------------------------------------------------
# 修复消息
# ---------------------------------------------------------------------------
def build_repair_messages(
    base_messages: Iterable[Message],
    invalid_text: str,
    violations: Sequence[dict[str, Any]],
    *,
    max_error_chars: int,
    max_output_chars: int,
) -> list[Message]:
    """把“上一次的错误输出 + 校验违例”回灌给模型。

    始终基于**原始消息**重建，避免多轮重试后上下文无限增长。
    """
    summary = "\n".join(
        f"- {item['path']}: {item['reason']}" for item in list(violations)[:MAX_VIOLATIONS]
    )
    summary = _truncate(summary or "- （无可用违例详情）", max_error_chars)
    echoed = _truncate(invalid_text or "", max_output_chars)

    return [
        *base_messages,
        Message(role="assistant", content=echoed),
        Message(
            role="user",
            content=(
                "上面的输出不符合要求的 JSON Schema，校验错误如下：\n"
                f"{summary}\n\n"
                "请重新输出完整的、符合该 schema 的 JSON。只输出 JSON 本身，"
                "不要输出 Markdown 代码块、注释或解释文字。"
            ),
        ),
    ]


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}...（已截断）"


__all__ = [
    "MAX_VIOLATIONS",
    "SCHEMA_ID_PATTERN",
    "StructuredSpec",
    "build_repair_messages",
    "extract_json",
    "prepare_structured",
    "validate_instance",
]
