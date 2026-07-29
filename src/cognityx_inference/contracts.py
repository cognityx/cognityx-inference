"""Provider-neutral inference value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

JSON = Mapping[str, Any]


class FinishReason(StrEnum):
    """Normalized reasons for generation termination."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


class LoadPolicy(StrEnum):
    """Control whether a local inference call may load model weights."""

    AUTO = "auto"
    REQUIRE_LOADED = "require_loaded"


class DiscoveryPolicy(StrEnum):
    """Control behavior when compatible hardware certification is absent."""

    ASK = "ask"
    AUTO = "auto"
    REQUIRE_EXISTING = "require_existing"


@dataclass(frozen=True, slots=True)
class TokenDetail:
    """One generated token and optional provider-reported measurements."""

    token: str
    token_id: int | None = None
    log_probability: float | None = None
    probability: float | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    top_log_probabilities: JSON = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider- or runtime-reported token counts."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Certified context calculation applied before inference dispatch."""

    input_tokens: int
    max_context_tokens: int
    reserved_tokens: int
    requested_max_output_tokens: int | None
    effective_max_output_tokens: int
    source: str
    counting: str = "tokenizer"


@dataclass(frozen=True, slots=True)
class InferenceTimings:
    """Measured request stages in seconds."""

    latency_seconds: float | None = None
    queue_seconds: float | None = None
    model_loading_seconds: float | None = None
    prompt_processing_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    token_generation_seconds: float | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Features a backend or provider can implement without emulation."""

    streaming: bool = False
    lifecycle: bool = False
    stop_conditions: bool = False
    top_k: bool = False
    min_p: bool = False
    seed: bool = False
    log_probabilities: bool = False
    token_offsets: bool = False
    reasoning: bool = False
    tools: bool = False
    structured_output: bool = False
    vision: bool = False
    local_telemetry: bool = False


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """A normalized chat or text generation request."""

    model: str
    messages: tuple[JSON, ...] = ()
    prompt: str | None = None
    tools: tuple[JSON, ...] = ()
    tool_choice: Any | None = None
    response_format: JSON | None = None
    client_type: str = "openai"
    provider: str = "local"
    backend: str = "vllm"
    profile: str = "bf16"
    load_policy: LoadPolicy = LoadPolicy.AUTO
    discovery_policy: DiscoveryPolicy = DiscoveryPolicy.REQUIRE_EXISTING
    required_context_length: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_output_tokens: int | None = None
    # Migration alias accepted from older Python callers. New code should use
    # max_output_tokens; OpenAI HTTP max_tokens is translated at the API edge.
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    log_probabilities: bool = False
    top_log_probabilities: int | None = None
    reasoning: JSON = field(default_factory=dict)
    stream: bool = False
    timeout_seconds: float | None = None
    first_token_timeout_seconds: float | None = None
    no_token_progress_timeout_seconds: float | None = None
    execution_context: JSON = field(default_factory=dict)
    request_metadata: JSON = field(default_factory=dict)
    extensions: JSON = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.load_policy, str):
            object.__setattr__(self, "load_policy", LoadPolicy(self.load_policy))
        if isinstance(self.discovery_policy, str):
            object.__setattr__(
                self,
                "discovery_policy",
                DiscoveryPolicy(self.discovery_policy),
            )
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not self.client_type.strip():
            raise ValueError("client_type cannot be empty")
        if not self.messages and self.prompt is None:
            raise ValueError("messages or prompt must be supplied")
        if self.messages and self.prompt is not None:
            raise ValueError("messages and prompt are mutually exclusive")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if (
            self.max_output_tokens is not None
            and self.max_tokens is not None
            and self.max_output_tokens != self.max_tokens
        ):
            raise ValueError("max_output_tokens and legacy max_tokens cannot disagree")
        if self.max_output_tokens is None and self.max_tokens is not None:
            object.__setattr__(self, "max_output_tokens", self.max_tokens)
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if (
            self.required_context_length is not None
            and self.required_context_length <= 0
        ):
            raise ValueError("required_context_length must be positive")
        for name in (
            "timeout_seconds",
            "first_token_timeout_seconds",
            "no_token_progress_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value.pop("max_tokens", None)
        return value


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """Normalized inference output with explicit unavailable values."""

    request_id: str
    content: str
    model: str
    provider: str
    backend: str
    finish_reason: FinishReason = FinishReason.UNKNOWN
    reasoning_content: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    timings: InferenceTimings = field(default_factory=InferenceTimings)
    token_budget: TokenBudget | None = None
    token_details: tuple[TokenDetail, ...] = ()
    retry_count: int = 0
    timed_out: bool = False
    requested_parameters: JSON = field(default_factory=dict)
    effective_parameters: JSON = field(default_factory=dict)
    telemetry: JSON = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    extensions: JSON = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["finish_reason"] = self.finish_reason.value
        return value
