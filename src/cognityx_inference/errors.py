"""Inference platform exceptions without heavyweight runtime imports."""


class ModelMetadataUnavailableError(OSError):
    """Model configuration is absent from the selected local cache."""


class ContextWindowExceeded(ValueError):
    """The certified context has no valid room for the requested output."""


class TokenCountingUnavailable(RuntimeError):
    """No approved tokenizer or explicit estimate is available."""


class ProviderCredentialMissing(RuntimeError):
    """A configured provider credential environment variable is absent."""

    def __init__(self, provider: str, environment_name: str) -> None:
        super().__init__(
            f"Provider '{provider}' requires environment variable "
            f"{environment_name}."
        )
        self.provider = provider
        self.environment_name = environment_name


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
