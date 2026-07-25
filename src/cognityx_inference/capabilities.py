"""Certified local-runtime capacity profiles and context resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


class ContextLimitError(ValueError):
    """Base class for context-capacity validation failures."""


class ModelContextLimitExceeded(ContextLimitError):
    """The requested context exceeds model-declared capacity."""


class CertifiedContextLimitExceeded(ContextLimitError):
    """The requested context exceeds the tested hardware boundary."""


class HardwareDiscoveryRequired(RuntimeError):
    """No compatible certified hardware result exists."""

    def __init__(
        self,
        *,
        model: str,
        backend: str,
        profile: str,
        model_context_limit: int | None,
        discovery_request_id: str | None = None,
    ) -> None:
        super().__init__(
            f"No compatible hardware certification exists for "
            f"{model} using {backend}/{profile}."
        )
        self.model = model
        self.backend = backend
        self.profile = profile
        self.model_context_limit = model_context_limit
        self.discovery_request_id = discovery_request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "hardware_discovery_required",
            "message": str(self),
            "model": self.model,
            "backend": self.backend,
            "profile": self.profile,
            "model_context_limit": self.model_context_limit,
            "discovery_request_id": self.discovery_request_id,
        }


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    """Fields that must match before certification can be reused."""

    hardware_fingerprint: str
    model: str
    model_revision: str | None
    backend: str
    backend_version: str | None
    profile: str
    kv_cache_precision: str
    tensor_parallelism: int = 1


@dataclass(frozen=True, slots=True)
class CertifiedInferenceProfile:
    """A successful hardware-tested local inference configuration."""

    profile_id: str
    created_at: str
    compatibility: RuntimeCompatibility
    model_context_limit: int | None
    maximum_certified_context_length: int
    maximum_certified_batch_size: int | None = None
    gpu_memory_utilization: float | None = None
    minimum_observed_tokens_per_second: float | None = None
    maximum_time_to_first_token_seconds: float | None = None
    evidence_job_id: str | None = None
    evidence_summary_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CertifiedInferenceProfile":
        data = dict(value)
        data["compatibility"] = RuntimeCompatibility(**data["compatibility"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ContextResolution:
    """Resolved engine capacity and its provenance."""

    model_limit: int
    certified_limit: int
    effective_limit: int
    requested_context_length: int | None
    profile_id: str
    warning: str | None = None


def hardware_fingerprint(inventory: Mapping[str, Any]) -> str:
    """Hash stable hardware/runtime identity fields."""
    identity = {
        "gpu_name": inventory.get("gpu_name") or inventory.get("device"),
        "gpu_total_memory_bytes": inventory.get("gpu_total_memory_bytes"),
        "gpu_total_memory_gb": inventory.get("gpu_total_memory_gb"),
        "cuda_version": inventory.get("cuda_version"),
        "platform": inventory.get("platform"),
    }
    encoded = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def new_profile_id(compatibility: RuntimeCompatibility) -> str:
    value = {
        **asdict(compatibility),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def resolve_context(
    model_limit: int | None,
    profile: CertifiedInferenceProfile,
    requested: int | None,
) -> ContextResolution:
    """Resolve and validate model, certified, and requested context limits."""
    if model_limit is None or model_limit <= 0:
        raise ContextLimitError("Model-declared context capacity is unavailable.")
    if requested is not None and requested > model_limit:
        raise ModelContextLimitExceeded(
            f"Requested context {requested} exceeds the model limit {model_limit}."
        )
    certified = profile.maximum_certified_context_length
    if requested is not None and requested > certified:
        raise CertifiedContextLimitExceeded(
            f"Requested context {requested} exceeds the certified hardware "
            f"limit {certified} for profile {profile.profile_id}."
        )
    effective = min(model_limit, certified)
    return ContextResolution(
        model_limit=model_limit,
        certified_limit=certified,
        effective_limit=effective,
        requested_context_length=requested,
        profile_id=profile.profile_id,
    )


def certified_profile_from_benchmark_result(
    result: Mapping[str, Any],
) -> CertifiedInferenceProfile | None:
    """Convert one successful legacy benchmark into conservative evidence."""
    if result.get("run_type") != "benchmark":
        return None
    metrics = result.get("metrics") or {}
    prompt_tokens = metrics.get("prompt_tokens")
    generated_tokens = metrics.get("generated_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(generated_tokens, int):
        return None
    tested_context = prompt_tokens + generated_tokens
    if tested_context <= 0 or result.get("finish_reason") in {"error", "cancelled"}:
        return None
    metadata = result.get("metadata") or {}
    engine = str(result.get("engine") or "transformers")
    profile_name = str(
        result.get("effective_load_profile")
        or result.get("requested_load_profile")
        or "bf16"
    )
    backend_version = (
        metadata.get("vllm_version")
        if engine == "vllm"
        else metadata.get("transformers_version")
    )
    compatibility = RuntimeCompatibility(
        hardware_fingerprint=hardware_fingerprint(metadata),
        model=str(result.get("model")),
        model_revision=result.get("model_revision"),
        backend=engine,
        backend_version=backend_version,
        profile=profile_name,
        kv_cache_precision=str(result.get("kv_cache_precision") or "auto"),
        tensor_parallelism=int(result.get("tensor_parallelism") or 1),
    )
    model_runtime = result.get("model_runtime") or {}
    architecture = model_runtime.get("architecture") or {}
    model_limit = architecture.get("max_position_embeddings")
    return CertifiedInferenceProfile(
        profile_id=new_profile_id(compatibility),
        created_at=datetime.now(timezone.utc).isoformat(),
        compatibility=compatibility,
        model_context_limit=model_limit if isinstance(model_limit, int) else None,
        maximum_certified_context_length=tested_context,
        maximum_certified_batch_size=1,
        gpu_memory_utilization=result.get("gpu_memory_utilization"),
        minimum_observed_tokens_per_second=metrics.get("tokens_per_second"),
        maximum_time_to_first_token_seconds=metrics.get(
            "time_to_first_token_seconds"
        ),
        evidence_job_id=None,
        evidence_summary_key=None,
        metadata={
            "source": "legacy_llm_benchmark",
            "benchmark_name": result.get("benchmark_name"),
            "configured_context_capacity": metrics.get(
                "total_context_length_tokens"
            ),
        },
    )
