"""Local inference backend adapters.

Concrete backends are imported lazily so Hugging Face cache configuration can
be established before Transformers initializes its hub constants.
"""

from cognityx_inference.backends.base import InferenceBackend

__all__ = ["InferenceBackend", "TransformersBackend", "VLLMBackend"]


def __getattr__(name: str):
    if name in {"TransformersBackend", "VLLMBackend"}:
        from cognityx_inference.backends.legacy import (
            TransformersBackend,
            VLLMBackend,
        )

        return {
            "TransformersBackend": TransformersBackend,
            "VLLMBackend": VLLMBackend,
        }[name]
    raise AttributeError(name)
