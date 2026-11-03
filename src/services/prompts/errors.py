"""提示词模板相关错误。

全部继承 :class:`GatewayError`，因此会被 ``app.py`` 里已注册的异常处理器统一
转换成 ``{"error": {"code": ..., "message": ...}}`` 形状，错误码风格与适配层一致。
"""

from __future__ import annotations

from src.services.llm_adaptor.errors import GatewayError


class PromptError(GatewayError):
    """提示词相关错误的基类。"""

    code = "prompt_error"
    status_code = 400
    retryable = False


class PromptTemplateInvalid(PromptError):
    """模板本身不合法（为空、占位符过多）。"""

    code = "prompt_template_invalid"


class PromptRenderError(PromptError):
    """渲染阶段失败（注入目标不存在、变量类型不支持等）。"""

    code = "prompt_render_failed"


class PromptVariablesInvalid(PromptError):
    """变量与模板契约不匹配（缺失、多余、类型非法）。"""

    code = "prompt_variables_invalid"


class PromptNotFound(PromptError):
    """模板不存在**或不属于当前用户**。

    两种情况统一返回 404，避免通过状态码差异探测其他用户的模板是否存在。
    """

    code = "prompt_not_found"
    status_code = 404


class PromptDisabled(PromptError):
    """模板存在但已被停用。"""

    code = "prompt_disabled"
    status_code = 409


class PromptTooLarge(PromptError):
    """模板正文或渲染结果超出大小上限。"""

    code = "prompt_too_large"
    status_code = 413


class PromptQuotaExceeded(PromptError):
    """该用户的模板数量已达上限。"""

    code = "prompt_quota_exceeded"
    status_code = 409


class PromptNameConflict(PromptError):
    """同一用户下已存在同名模板。"""

    code = "prompt_name_conflict"
    status_code = 409


class PromptVersionConflict(PromptError):
    """乐观锁冲突：``expected_version`` 与当前 head_version 不一致。"""

    code = "prompt_version_conflict"
    status_code = 409


class PromptStoreUnavailable(PromptError):
    """模板存储未配置或当前不可用。"""

    code = "prompt_store_unavailable"
    status_code = 503
    retryable = True


__all__ = [
    "PromptDisabled",
    "PromptError",
    "PromptNameConflict",
    "PromptNotFound",
    "PromptQuotaExceeded",
    "PromptRenderError",
    "PromptStoreUnavailable",
    "PromptTemplateInvalid",
    "PromptTooLarge",
    "PromptVariablesInvalid",
    "PromptVersionConflict",
]
