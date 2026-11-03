"""各协议的适配器实现。

- :mod:`openai_compat` 覆盖所有 OpenAI 兼容厂商（DeepSeek / Kimi / Qwen / Ollama …）
- :mod:`openai_responses` 覆盖原生 Responses API
"""

from src.services.llm_adaptor.providers.openai_compat import OpenAICompatAdapter
from src.services.llm_adaptor.providers.openai_responses import OpenAIResponsesAdapter

__all__ = ["OpenAICompatAdapter", "OpenAIResponsesAdapter"]
