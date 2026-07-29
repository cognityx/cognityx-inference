"""Provider definitions, profiles, policies, and normalized status values."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.contracts import InferenceRequest, ModelCapabilities


class ProfileState(StrEnum):
    UNVERIFIED = "unverified"
    DISCOVERED = "discovered"
    TESTED = "tested"
    CERTIFIED = "certified"
    DEPRECATED = "deprecated"


class ProviderErrorCategory(StrEnum):
    NOT_CONFIGURED = "not_configured"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    RATE_LIMITED = "rate_limited"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN_PROVIDER_ERROR = "unknown_provider_error"


@dataclass(frozen=True, slots=True)
class ParameterPolicy:
    """Declarative common-to-provider parameter contract."""

    supported: frozenset[str] = frozenset()
    translated: Mapping[str, str] = field(default_factory=dict)
    ignored_when_unset: frozenset[str] = frozenset()
    rejected_when_supplied: frozenset[str] = frozenset()
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def wire_name(self, common_name: str) -> str:
        return self.translated.get(common_name, common_name)


@dataclass(frozen=True, slots=True)
class ProviderAccountLimits:
    """Observed or configured account throughput limits, never model context."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderModelProfile:
    provider: str
    model_id: str
    context: CertifiedContextProfile | None = None
    tokenizer: str | None = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    parameter_policy: ParameterPolicy = field(default_factory=ParameterPolicy)
    limits: ProviderAccountLimits = field(default_factory=ProviderAccountLimits)
    state: ProfileState = ProfileState.UNVERIFIED
    verification_source: str = "manual"
    verified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        policy = value["parameter_policy"]
        for field_name in (
            "supported",
            "ignored_when_unset",
            "rejected_when_supplied",
        ):
            policy[field_name] = sorted(policy[field_name])
        return value


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    provider: str
    adapter: str
    enabled: bool
    credential_status: str
    configuration_status: str
    default_model: str | None
    default_profile: str | None
    last_successful_test: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiscoveredModels:
    provider: str
    models: tuple[str, ...]
    discovered_at: str
    cached: bool = False

    @classmethod
    def now(cls, provider: str, models: tuple[str, ...]) -> "DiscoveredModels":
        return cls(
            provider=provider,
            models=models,
            discovered_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def supplied_parameters(request: InferenceRequest) -> set[str]:
    """Return only common parameters explicitly carrying caller intent."""
    selected = {"messages", "model"}
    optional = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "max_output_tokens": request.max_output_tokens,
        "stop": request.stop or None,
        "seed": request.seed,
        "logprobs": True if request.log_probabilities else None,
        "top_logprobs": request.top_log_probabilities,
        "tools": request.tools or None,
        "tool_choice": request.tool_choice,
        "response_format": request.response_format,
        "reasoning": request.reasoning or None,
    }
    selected.update(name for name, value in optional.items() if value is not None)
    selected.update(str(name) for name in request.extensions)
    return selected
