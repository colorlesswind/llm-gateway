"""模板解析与渲染（纯函数，不做任何 IO）。

刻意**不使用** ``str.format`` / ``str.format_map``：

- 提示词里出现 JSON 示例（``{"a": 1}``）会直接 ``KeyError``；
- ``format_map`` 支持 ``{obj.__class__}`` 之类的属性访问，是明确的注入面。

这里只识别 ``{{ name }}`` 形式的占位符，其余花括号原样保留；变量取值通过
``re.sub`` 回调逐段替换，不存在任何求值过程。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from src.services.prompts.errors import (
    PromptTemplateInvalid,
    PromptVariablesInvalid,
)

# 变量名限定为合法 Python 标识符，避免 ``{{ a.b }}`` / ``{{ a[0] }}`` 之类的表达式
PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

# 单个模板允许的占位符数量上限（防止渲染阶段退化为 O(n^2)）
MAX_VARIABLES = 64

# 估算 token 时 ASCII 与非 ASCII 的平均字符数（中文约 1 字/token）
ASCII_CHARS_PER_TOKEN = 4.0


def extract_variables(template: str) -> list[str]:
    """按首次出现顺序返回模板声明（使用）的变量名，去重。"""
    seen: list[str] = []
    for match in PLACEHOLDER_RE.finditer(template or ""):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def validate_template(template: str) -> list[str]:
    """上传期校验：非空、变量数量受限；返回声明的变量列表。"""
    if not template or not template.strip():
        raise PromptTemplateInvalid("提示词模板正文不能为空")
    variables = extract_variables(template)
    if len(variables) > MAX_VARIABLES:
        raise PromptTemplateInvalid(
            f"占位符数量超限（{len(variables)} > {MAX_VARIABLES}）"
        )
    return variables


def content_hash(text: str) -> str:
    """模板正文的 sha256（hex）。日志/响应里只暴露它，不暴露原文。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def byte_length(text: str) -> int:
    return len((text or "").encode("utf-8"))


def estimate_tokens(text: str) -> int:
    """粗略 token 估算，仅用于配额提示与日志，不作为硬性预算。"""
    text = text or ""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / ASCII_CHARS_PER_TOKEN + other_chars))


def _stringify(name: str, value: Any) -> str:
    """把变量值转成字符串。只接受标量与 JSON 可序列化的容器。"""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        # 必须排在 int 之前：bool 是 int 的子类
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:  # pragma: no cover - 防御性
            raise PromptVariablesInvalid(
                f"变量 {name} 无法序列化为字符串：{exc}"
            ) from exc
    raise PromptVariablesInvalid(
        f"变量 {name} 的类型不受支持：{type(value).__name__}"
    )


def render_template(
    template: str,
    variables: dict[str, Any],
    *,
    on_missing: str = "error",
) -> str:
    """把 ``variables`` 渲染进模板。

    - 变量 key 不在模板中声明 → 400（拼写错误不应该被静默忽略）；
    - 模板声明的变量缺失 → 由 ``on_missing`` 决定报错还是替换为空串。
    """
    declared = extract_variables(template)
    declared_set = set(declared)
    unknown = sorted(set(variables) - declared_set)
    if unknown:
        raise PromptVariablesInvalid(
            "模板未声明这些变量：" + ", ".join(unknown),
            details={"unknown_variables": unknown},
        )

    missing = [name for name in declared if name not in variables]
    if missing and on_missing != "empty":
        raise PromptVariablesInvalid(
            "缺少模板所需变量：" + ", ".join(missing),
            details={"missing_variables": missing},
        )

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            return ""
        return _stringify(name, variables[name])

    return PLACEHOLDER_RE.sub(_replace, template)


__all__ = [
    "MAX_VARIABLES",
    "PLACEHOLDER_RE",
    "byte_length",
    "content_hash",
    "estimate_tokens",
    "extract_variables",
    "render_template",
    "validate_template",
]
