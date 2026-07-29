"""Explicit, secret-safe live diagnostics for configured providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.contracts import FinishReason, InferenceRequest
from cognityx_inference.errors import (
    ProviderCredentialMissing,
    ProviderRequestError,
)
from cognityx_inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from cognityx_inference.security import CredentialResolver


@dataclass(frozen=True, slots=True)
class ProviderDiagnosticResult:
    provider: str
    model: str | None
    status: str
    latency_seconds: float | None = None
    error_category: str | None = None
    model_list_authenticated: bool = False
    non_streaming: bool = False
    streaming: bool = False
    usage: dict[str, int | None] | None = None
    rate_limits: dict[str, str] | None = None
    account_balance: None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def test_provider(
    definition: ProviderDefinition,
    *,
    credential_resolver: CredentialResolver | None = None,
) -> ProviderDiagnosticResult:
    model = definition.smoke_test_model
    adapter = OpenAICompatibleProvider.from_definition(
        definition,
        credential_resolver=credential_resolver,
    )
    try:
        available = adapter.credential_available()
    except ValueError:
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=model,
            status="failed",
            error_category="credential_configuration",
        )
    if not available:
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=model,
            status="skipped",
            error_category="missing_credential",
        )
    if not model:
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=None,
            status="failed",
            error_category="smoke_test_model_not_configured",
        )
    started = time.monotonic()
    try:
        adapter.list_models()
        common = dict(
            model=model,
            messages=(
                {
                    "role": "user",
                    "content": "Reply with exactly OK.",
                },
            ),
            provider=definition.name,
            max_output_tokens=4,
            temperature=0,
        )
        plain = adapter.infer(InferenceRequest(**common))
        chunks: list[str] = []
        streamed = adapter.infer(
            InferenceRequest(**common, stream=True),
            on_text=chunks.append,
        )
        valid_plain = (
            plain.content.strip() == "OK"
            and plain.finish_reason is not FinishReason.UNKNOWN
            and _usage_valid(plain.usage)
        )
        valid_stream = (
            "".join(chunks).strip() == "OK"
            and streamed.finish_reason is not FinishReason.UNKNOWN
            and _usage_valid(streamed.usage)
        )
        if not valid_plain or not valid_stream:
            return ProviderDiagnosticResult(
                provider=definition.name,
                model=model,
                status="failed",
                latency_seconds=time.monotonic() - started,
                error_category="response_validation",
                model_list_authenticated=True,
                non_streaming=valid_plain,
                streaming=valid_stream,
            )
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=model,
            status="passed",
            latency_seconds=time.monotonic() - started,
            model_list_authenticated=True,
            non_streaming=True,
            streaming=True,
            usage={
                "prompt_tokens": plain.usage.prompt_tokens,
                "completion_tokens": plain.usage.completion_tokens,
                "total_tokens": plain.usage.total_tokens,
            },
            rate_limits=dict(plain.extensions.get("rate_limits") or {}),
        )
    except ProviderCredentialMissing:
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=model,
            status="skipped",
            error_category="missing_credential",
        )
    except ProviderRequestError as exc:
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=model,
            status="failed",
            latency_seconds=time.monotonic() - started,
            error_category=exc.category,
        )
    except (ValueError, RuntimeError):
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=model,
            status="failed",
            latency_seconds=time.monotonic() - started,
            error_category="configuration_or_validation",
        )


def _usage_valid(usage: Any) -> bool:
    """Accept unavailable usage, but validate every supplied count."""
    values = (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )
    return all(
        value is None or (isinstance(value, int) and value >= 0)
        for value in values
    )
