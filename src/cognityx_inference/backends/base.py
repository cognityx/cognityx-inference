"""Local inference backend contract."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from cognityx_inference.contracts import (
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
)


class InferenceBackend(Protocol):
    """A loadable local or self-hosted inference runtime."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    def load(self) -> None: ...

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> InferenceResponse: ...

    def unload(self) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def model_context_limit(self) -> int | None: ...
