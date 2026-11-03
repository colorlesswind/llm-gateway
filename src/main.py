"""LLM Gateway 应用入口。

本地运行：
    uvicorn src.main:app --reload

配置缺失会在导入时立即抛出 ``ConfigurationError``（启动期 fail fast）。
"""

from __future__ import annotations

from src.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
