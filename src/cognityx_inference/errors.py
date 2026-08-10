"""Inference platform exceptions without heavyweight runtime imports."""

from __future__ import annotations

from typing import Any, Mapping


class InferenceContractError(RuntimeError):
    """Machine-readable validation or execution failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": str(self),
            "details": self.details,
        }


class AdapterError(InferenceContractError):
    """A Training adapter cannot be safely used for inference."""


class ResearchRunError(InferenceContractError):
    """A frozen research execution cannot be completed or published."""


class ModelMetadataUnavailableError(OSError):
    """Model configuration is absent from the selected local cache."""


class ContextWindowExceeded(ValueError):
    """The certified context has no valid room for the requested output."""


class TokenCountingUnavailable(RuntimeError):
    """No approved tokenizer or explicit estimate is available."""


class ProviderCredentialMissing(RuntimeError):
    """No credential was found in the configured provider sources."""

    def __init__(
        self,
        provider: str,
        environment_name: str,
        secret_name: str | None = None,
    ) -> None:
        sources = f"environment variable {environment_name}"
        if secret_name:
            sources += f" or configured secret entry {secret_name}"
        super().__init__(f"Provider '{provider}' requires {sources}.")
        self.provider = provider
        self.environment_name = environment_name
        self.secret_name = secret_name


class ProviderRequestError(RuntimeError):
    """A sanitized provider request failure."""

    def __init__(
        self, provider: str, category: str, *, status: int | None = None
    ) -> None:
        detail = f" ({status})" if status is not None else ""
        super().__init__(f"Provider '{provider}' request failed: {category}{detail}.")
        self.provider = provider
        self.category = category
        self.status = status


class UnsupportedProviderParameter(ValueError):
    """A caller explicitly supplied a parameter rejected by the provider."""

    def __init__(self, provider: str, parameters: list[str]) -> None:
        joined = ", ".join(sorted(parameters))
        super().__init__(
            f"Provider '{provider}' does not support explicit parameter(s): {joined}."
        )
        self.provider = provider
        self.parameters = tuple(sorted(parameters))
