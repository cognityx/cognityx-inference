"""Explicit, secret-safe live diagnostics for configured providers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.contracts import FinishReason, InferenceRequest
from cognityx_inference.errors import (
    ProviderCredentialMissing,
    ProviderRequestError,
    UnsupportedProviderParameter,
)
from cognityx_inference.providers.factory import build_provider_adapters
from cognityx_inference.security import CredentialResolver
from cognityx_inference.token_budget import apply_token_budget


@dataclass(frozen=True, slots=True)
class ProviderDiagnosticResult:
    provider: str
    model: str | None
    status: str
    reason: str | None = None
    latency_ms: int | None = None
    error_category: str | None = None
    authentication: str = "not_tested"
    model_discovery: str = "not_tested"
    available_alternatives: tuple[str, ...] = ()
    non_streaming: str = "not_tested"
    streaming: str = "not_tested"
    structured_output: str = "not_tested"
    token_budget: str = "not_tested"
    usage: dict[str, int | None] | None = None
    rate_limits: dict[str, str] | None = None
    account_balance: None = None
    tested_at: str | None = None

    @property
    def latency_seconds(self) -> float | None:
        return self.latency_ms / 1000 if self.latency_ms is not None else None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["latency_seconds"] = self.latency_seconds
        return value


def test_provider(
    definition: ProviderDefinition,
    *,
    credential_resolver: CredentialResolver | None = None,
    model: str | None = None,
    structured_output: bool = False,
    timeout_seconds: float = 30,
) -> ProviderDiagnosticResult:
    selected_model = model or definition.default_model or definition.smoke_test_model
    if not definition.enabled:
        return _skipped(definition.name, selected_model, "not_configured")
    resolver = credential_resolver or CredentialResolver()
    try:
        adapters = build_provider_adapters({definition.name: definition}, resolver)
        adapter = adapters.get(definition.name)
        if adapter is None or not adapter.credential_available():
            return _skipped(definition.name, selected_model, "not_configured")
    except (ValueError, ProviderCredentialMissing):
        return _skipped(definition.name, selected_model, "not_configured")
    if not selected_model:
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=None,
            status="failed",
            reason="default_model_not_configured",
            error_category="not_configured",
        )

    started = time.monotonic()
    discovery = "not_supported"
    alternatives: tuple[str, ...] = ()
    reasoning = {"thinking_level": "low"} if definition.name == "gemini" else {}
    try:
        try:
            models = adapter.list_models(timeout_seconds=timeout_seconds)
        except LookupError:
            models = ()
        else:
            discovery = "passed"
            if selected_model not in models:
                alternatives = tuple(models[:5])
                return _failed(
                    definition.name,
                    selected_model,
                    started,
                    "model_unavailable",
                    authentication="passed",
                    model_discovery=discovery,
                    alternatives=alternatives,
                )
        request = InferenceRequest(
            model=selected_model,
            messages=({"role": "user", "content": "Reply with exactly OK."},),
            provider=definition.name,
            max_output_tokens=128,
            temperature=0,
            timeout_seconds=timeout_seconds,
            reasoning=reasoning,
        )
        profile = adapter.context_profile(selected_model)
        budget_state = "not_configured"
        if profile is not None:
            request, _ = apply_token_budget(
                request,
                profile,
                adapter.count_input_tokens,
            )
            budget_state = "passed"
        plain = adapter.infer(request)
        chunks: list[str] = []
        streamed = adapter.infer(
            InferenceRequest(
                model=selected_model,
                messages=({"role": "user", "content": "Reply with exactly OK."},),
                provider=definition.name,
                max_output_tokens=128,
                temperature=0,
                timeout_seconds=timeout_seconds,
                stream=True,
                reasoning=reasoning,
            ),
            on_text=chunks.append,
        )
        valid_plain = _valid_response(plain.content, plain.finish_reason, plain.usage)
        valid_stream = _valid_response(
            "".join(chunks), streamed.finish_reason, streamed.usage
        )
        structured_state = "not_tested"
        if structured_output:
            structured = adapter.infer(
                InferenceRequest(
                    model=selected_model,
                    messages=(
                        {
                            "role": "user",
                            "content": 'Return {"status":"OK"} and nothing else.',
                        },
                    ),
                    provider=definition.name,
                    max_output_tokens=128,
                    temperature=0,
                    timeout_seconds=timeout_seconds,
                    response_format={"type": "json_object"},
                    reasoning=reasoning,
                )
            )
            structured_state = (
                "passed" if structured.content.strip().startswith("{") else "failed"
            )
        passed = valid_plain and valid_stream and structured_state != "failed"
        return ProviderDiagnosticResult(
            provider=definition.name,
            model=selected_model,
            status="passed" if passed else "failed",
            latency_ms=round((time.monotonic() - started) * 1000),
            error_category=None if passed else "invalid_response",
            authentication="passed",
            model_discovery=discovery,
            non_streaming="passed" if valid_plain else "failed",
            streaming="passed" if valid_stream else "failed",
            structured_output=structured_state,
            token_budget=budget_state,
            usage={
                "prompt_tokens": plain.usage.prompt_tokens,
                "completion_tokens": plain.usage.completion_tokens,
                "total_tokens": plain.usage.total_tokens,
            },
            rate_limits=dict(plain.extensions.get("rate_limits") or {}),
            tested_at=datetime.now(timezone.utc).isoformat(),
        )
    except ProviderCredentialMissing:
        return _skipped(definition.name, selected_model, "not_configured")
    except UnsupportedProviderParameter:
        return _failed(
            definition.name,
            selected_model,
            started,
            "unsupported_parameter",
            authentication="passed",
            model_discovery=discovery,
            alternatives=alternatives,
        )
    except ProviderRequestError as exc:
        return _failed(
            definition.name,
            selected_model,
            started,
            exc.category,
            authentication=(
                "failed"
                if exc.category in {"authentication_failed", "permission_denied"}
                else "passed"
            ),
            model_discovery=discovery,
            alternatives=alternatives,
        )
    except (ValueError, RuntimeError):
        return _failed(
            definition.name,
            selected_model,
            started,
            "invalid_response",
            authentication="passed",
            model_discovery=discovery,
            alternatives=alternatives,
        )


def _valid_response(content: str, finish: FinishReason, usage: Any) -> bool:
    values = (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
    return (
        content.strip() == "OK"
        and finish is not FinishReason.UNKNOWN
        and all(
            value is None or (isinstance(value, int) and value >= 0) for value in values
        )
    )


def _skipped(provider: str, model: str | None, reason: str) -> ProviderDiagnosticResult:
    return ProviderDiagnosticResult(
        provider=provider,
        model=model,
        status="skipped",
        reason=reason,
        error_category=reason,
    )


def _failed(
    provider: str,
    model: str,
    started: float,
    category: str,
    *,
    authentication: str,
    model_discovery: str,
    alternatives: tuple[str, ...],
) -> ProviderDiagnosticResult:
    return ProviderDiagnosticResult(
        provider=provider,
        model=model,
        status="failed",
        latency_ms=round((time.monotonic() - started) * 1000),
        error_category=category,
        authentication=authentication,
        model_discovery=model_discovery,
        available_alternatives=alternatives,
        tested_at=datetime.now(timezone.utc).isoformat(),
    )
