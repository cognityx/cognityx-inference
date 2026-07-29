"""Commercial inference provider adapters with cycle-safe lazy exports."""

from typing import Any

__all__ = [
    "GroqProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "XAIProvider",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from cognityx_inference.providers import openai_compatible

        return getattr(openai_compatible, name)
    raise AttributeError(name)
