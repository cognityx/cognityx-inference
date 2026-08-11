"""Application service that routes normalized inference requests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from cognityx_inference.environment import configure_huggingface_cache

# Hugging Face Hub snapshots its cache constants during import. Configure the
# shared cache before importing the backend module (which imports transformers).
configure_huggingface_cache()

from cognityx_inference.adapters import AdapterRepository, VerifiedAdapter
from cognityx_inference.backends.legacy import discover_model_context_limit
from cognityx_inference.capabilities import (
    HardwareDiscoveryRequired,
    RuntimeCompatibility,
    hardware_fingerprint,
    resolve_context,
)
from cognityx_inference.contracts import (
    DiscoveryPolicy,
    InferenceRequest,
    InferenceResponse,
    LoadPolicy,
    ThinkingMode,
    ThinkingResolution,
)
from cognityx_inference.errors import AdapterError
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.model_references import resolve_local_model_reference
from cognityx_inference.storage import (
    CertifiedProfileRepository,
    InferenceArtifactRepository,
)
from cognityx_inference.token_budget import apply_token_budget


class Provider(Protocol):
    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> InferenceResponse: ...


class ModelNotLoadedError(LookupError):
    """A require-loaded request has no active resident model."""


class DiscoveryStarter(Protocol):
    def start(
        self,
        *,
        model: str,
        backend: str,
        profile: str,
        model_context_limit: int | None,
        runtime: Mapping[str, Any],
    ) -> str: ...


class InferenceService:
    """Route calls while keeping storage and lifecycle policy centralized."""

    def __init__(
        self,
        models: ModelManager,
        providers: Mapping[str, Provider] | None = None,
        artifacts: InferenceArtifactRepository | None = None,
        certified_profiles: CertifiedProfileRepository | None = None,
        inventory: Callable[[], Mapping[str, Any]] | None = None,
        discovery: DiscoveryStarter | None = None,
        provider_registry: Any | None = None,
        adapter_repository: AdapterRepository | None = None,
        research_runner: Any | None = None,
    ) -> None:
        self.models = models
        self.providers = dict(providers or {})
        self.artifacts = artifacts
        self.certified_profiles = certified_profiles
        self.inventory = inventory or (lambda: {})
        self.discovery = discovery
        self.provider_registry = provider_registry
        self.adapter_repository = adapter_repository
        self.research_runner = research_runner

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
        *,
        owner_id: str = "local",
    ) -> InferenceResponse:
        dispatched_request = request
        if request.provider != "local":
            if request.adapter_manifest_uri is not None:
                raise AdapterError(
                    "adapter_not_supported",
                    "Training adapters can only be selected by the local vLLM backend",
                )
            try:
                provider = self.providers[request.provider]
            except KeyError as exc:
                raise ValueError(f"Unknown provider: {request.provider}") from exc
            self._validate_capabilities(request, provider)
            thinking_resolution = self._resolve_thinking(request, provider)
            context_profile = getattr(provider, "context_profile", lambda model: None)(
                request.model
            )
            if context_profile is None:
                budget = None
            else:
                dispatched_request, budget = apply_token_budget(
                    request,
                    context_profile,
                    lambda selected: provider.count_input_tokens(selected),
                )
            response = provider.infer(dispatched_request, on_text=on_text)
            if budget is not None:
                response = replace(response, token_budget=budget)
            else:
                response = replace(
                    response,
                    warnings=(
                        *response.warnings,
                        "Provider model context profile is unverified; "
                        "automatic token budgeting was unavailable.",
                    ),
                )
            response = replace(
                response,
                extensions={
                    **response.extensions,
                    "thinking": thinking_resolution.to_dict(),
                },
            )
        else:
            selected_adapter = self._resolve_adapter(request)
            canonical_model = resolve_local_model_reference(request.model).resolved
            if canonical_model != request.model:
                request = replace(request, model=canonical_model)
            dispatched_request = request
            if request.load_policy is LoadPolicy.REQUIRE_LOADED:
                try:
                    lease = self.models.acquire_loaded(request.model)
                except LookupError as exc:
                    raise ModelNotLoadedError(str(exc)) from exc
                context_resolution = None
                certified = self._profile_for_loaded_lease(lease)
            else:
                requested_runtime = dict(request.extensions.get("runtime", {}))
                if request.model_revision is not None:
                    requested_runtime["model_revision"] = request.model_revision
                runtime, context_resolution, certified = self.prepare_runtime(
                    request.model,
                    request.backend,
                    request.profile,
                    requested_runtime,
                    request.required_context_length,
                    request.discovery_policy,
                    owner_id,
                )
                lease = self.models.acquire(request.model, request.backend, runtime)
            with lease as backend:
                self._validate_capabilities(request, backend)
                runtime_identity = self._runtime_identity(backend)
                thinking_resolution = self._resolve_thinking(request, backend)
                self._validate_requested_revision(request, runtime_identity)
                compatibility = (
                    self.adapter_repository.compatibility(
                        selected_adapter, runtime_identity
                    )
                    if selected_adapter is not None
                    and self.adapter_repository is not None
                    else None
                )
                counter = getattr(backend, "count_input_tokens", None)
                # Older profiles retain their stored context boundary and gain
                # the default reserve only when the backend can count exactly.
                if certified is not None and counter is not None:
                    dispatched_request, budget = apply_token_budget(
                        request,
                        certified.context_profile,
                        counter,
                    )
                else:
                    budget = None
                if selected_adapter is None:
                    response = backend.infer(dispatched_request, on_text=on_text)
                else:
                    response = backend.infer(
                        dispatched_request,
                        on_text=on_text,
                        adapter=selected_adapter,
                    )
                if budget is not None:
                    response = replace(response, token_budget=budget)
                response = replace(
                    response,
                    extensions={
                        **response.extensions,
                        "runtime_fingerprint": _runtime_fingerprint(
                            backend,
                            dispatched_request,
                            runtime_identity,
                            thinking_resolution,
                        ),
                        "thinking": thinking_resolution.to_dict(),
                        **(
                            {
                                "adapter": selected_adapter.public_identity(),
                                "adapter_compatibility": compatibility,
                            }
                            if selected_adapter is not None
                            else {}
                        ),
                    },
                )
            if context_resolution is not None:
                response = replace(
                    response,
                    extensions={
                        **response.extensions,
                        "context_resolution": asdict(context_resolution),
                    },
                )
        if self.artifacts is not None:
            self.artifacts.save_exchange(dispatched_request, response)
        return response

    def _resolve_adapter(self, request: InferenceRequest) -> VerifiedAdapter | None:
        if request.adapter_manifest_uri is None:
            return None
        if self.adapter_repository is None:
            raise AdapterError(
                "adapter_not_supported",
                "Adapter Storage access is not configured for this service",
            )
        return self.adapter_repository.verify_and_materialize(
            request.adapter_manifest_uri
        )

    @staticmethod
    def _runtime_identity(backend: Any) -> dict[str, Any]:
        resolver = getattr(backend, "runtime_identity", None)
        if resolver is None:
            return {"name": getattr(backend, "model_name", None)}
        return dict(resolver())

    @staticmethod
    def _resolve_thinking(
        request: InferenceRequest, backend: Any
    ) -> ThinkingResolution:
        resolver = getattr(backend, "resolve_thinking", None)
        if resolver is not None:
            return resolver(request.thinking)
        if request.thinking is ThinkingMode.ENABLED:
            raise ValueError(
                "Explicitly requested thinking is unsupported by "
                f"{request.client_type}/{request.backend}"
            )
        return ThinkingResolution(
            requested=request.thinking,
            effective=ThinkingMode.DISABLED,
            mechanism="ordinary_generation:no_thinking_capability",
        )

    @staticmethod
    def _validate_requested_revision(
        request: InferenceRequest, runtime_identity: Mapping[str, Any]
    ) -> None:
        requested = request.model_revision
        actual = runtime_identity.get("requested_revision") or runtime_identity.get(
            "resolved_revision"
        )
        if requested is not None and actual is not None and requested != actual:
            raise AdapterError(
                "base_model_revision_mismatch",
                "The loaded base model revision does not match the requested revision",
                details={"expected": requested, "actual": actual},
            )

    def load_model(
        self,
        model: str,
        backend: str = "vllm",
        profile: str = "bf16",
        runtime: Mapping[str, Any] | None = None,
        required_context_length: int | None = None,
        discovery_policy: DiscoveryPolicy = DiscoveryPolicy.REQUIRE_EXISTING,
        *,
        owner_id: str = "local",
    ) -> Any:
        """Resolve certified capacity, load the model, and keep it resident."""
        model = resolve_local_model_reference(model).resolved
        resolved, context, _ = self.prepare_runtime(
            model,
            backend,
            profile,
            dict(runtime or {}),
            required_context_length,
            discovery_policy,
            owner_id,
        )
        status = self.models.load(model, backend, resolved)
        return status, context

    def count_input_tokens(self, request: InferenceRequest) -> int | None:
        """Count a prompt against the active local backend without generating."""
        if request.provider != "local":
            return None
        canonical_model = resolve_local_model_reference(request.model).resolved
        try:
            lease = self.models.acquire_loaded(canonical_model)
        except LookupError as exc:
            raise ModelNotLoadedError(str(exc)) from exc
        with lease as backend:
            counter = getattr(backend, "count_input_tokens", None)
            return counter(replace(request, model=canonical_model)) if counter else None

    def prepare_runtime(
        self,
        model: str,
        backend: str,
        profile: str,
        runtime: dict[str, Any],
        required_context_length: int | None,
        discovery_policy: DiscoveryPolicy,
        owner_id: str = "local",
    ) -> tuple[dict[str, Any], Any | None, Any | None]:
        """Resolve context from model metadata and compatible certification."""
        kv_cache_was_requested = "kv_cache_precision" in runtime
        runtime["quantization"] = profile
        if self.certified_profiles is None:
            return runtime, None, None
        model_limit, revision = discover_model_context_limit(
            model,
            allow_download=bool(runtime.get("allow_download", False)),
            revision=runtime.get("model_revision"),
        )
        inventory = dict(self.inventory())
        compatibility = RuntimeCompatibility(
            hardware_fingerprint=hardware_fingerprint(inventory),
            model=model,
            model_revision=revision,
            backend=backend,
            backend_version=_backend_version(backend),
            profile=profile,
            kv_cache_precision=str(runtime.get("kv_cache_precision", "auto")),
            tensor_parallelism=int(runtime.get("tensor_parallelism", 1)),
        )
        requested_profile_id = runtime.get("certified_profile_id")
        certified = None
        if requested_profile_id:
            getter = getattr(self.certified_profiles, "get_profile", None)
            certified = getter(str(requested_profile_id)) if getter else None
            if certified is None:
                raise ValueError(f"Certified profile not found: {requested_profile_id}")
            if certified.compatibility != compatibility:
                raise ValueError(
                    "The selected certified profile is not compatible with "
                    "this model, backend, runtime, and hardware."
                )
        else:
            certified = self.certified_profiles.find_compatible(compatibility)
        if (
            certified is None
            and not kv_cache_was_requested
            and hasattr(self.certified_profiles, "find_compatible_any_kv_cache")
        ):
            certified = self.certified_profiles.find_compatible_any_kv_cache(
                compatibility
            )
        if certified is None:
            request_id = (
                self.discovery.start(
                    model=model,
                    backend=backend,
                    profile=profile,
                    owner_id=owner_id,
                    model_context_limit=model_limit,
                    runtime=runtime,
                )
                if (
                    self.discovery is not None
                    and discovery_policy is DiscoveryPolicy.AUTO
                )
                else None
            )
            raise HardwareDiscoveryRequired(
                model=model,
                backend=backend,
                profile=profile,
                model_context_limit=model_limit,
                discovery_request_id=request_id,
            )
        context = resolve_context(model_limit, certified, required_context_length)
        runtime["context_length"] = context.effective_limit
        runtime["kv_cache_precision"] = certified.compatibility.kv_cache_precision
        runtime.setdefault(
            "gpu_memory_utilization", certified.gpu_memory_utilization or 0.9
        )
        runtime["certified_profile_id"] = certified.profile_id
        return runtime, context, certified

    def _profile_for_loaded_lease(self, lease: Any) -> Any | None:
        if self.certified_profiles is None:
            return None
        runtime = dict(lease.identity.runtime)
        raw = runtime.get("certified_profile_id")
        if raw is None:
            return None
        profile_id = raw.strip("'\"")
        getter = getattr(self.certified_profiles, "get_profile", None)
        return getter(profile_id) if getter else None

    @staticmethod
    def _validate_capabilities(request: InferenceRequest, adapter: Any) -> None:
        capabilities = adapter.capabilities
        requested = {
            "streaming": request.stream,
            "stop_conditions": bool(request.stop),
            "top_k": request.top_k is not None,
            "min_p": request.min_p is not None,
            "seed": request.seed is not None,
            "log_probabilities": (
                request.log_probabilities or request.top_log_probabilities is not None
            ),
            "reasoning": bool(request.reasoning),
            "thinking": request.thinking is ThinkingMode.ENABLED,
            "adapters": request.adapter_manifest_uri is not None,
        }
        unsupported = [
            name
            for name, enabled in requested.items()
            if enabled and not getattr(capabilities, name)
        ]
        if unsupported:
            if "adapters" in unsupported:
                raise AdapterError(
                    "adapter_not_supported",
                    f"The {request.backend} backend does not support adapter selection",
                )
            raise ValueError(
                "Explicitly requested parameter(s) unsupported by "
                f"{request.client_type}/{request.backend}: {', '.join(unsupported)}"
            )


def _backend_version(backend: str) -> str | None:
    package = {"vllm": "vllm", "transformers": "transformers"}.get(backend)
    if package is None:
        return None
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _runtime_fingerprint(
    backend: Any,
    request: InferenceRequest,
    identity: Mapping[str, Any],
    thinking: ThinkingResolution,
) -> dict[str, Any]:
    """Describe the runtime inputs that must match across a paired run."""
    runtime = dict(getattr(backend, "runtime", {}) or {})
    engine = getattr(backend, "engine", None)
    engine_runtime = dict(getattr(engine, "model_runtime", {}) or {})
    settings = getattr(engine, "settings", None)
    effective_generation = (
        asdict(settings) if hasattr(settings, "__dataclass_fields__") else {}
    )
    document = {
        "base_model": dict(identity),
        "backend": request.backend,
        "backend_version": _backend_version(request.backend),
        "runtime": runtime,
        "engine_runtime": engine_runtime,
        "decoding": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_output_tokens": request.max_output_tokens,
            "stop": list(request.stop),
            "seed": request.seed,
            "reasoning": dict(request.reasoning),
            "thinking": thinking.to_dict(),
            "effective_generation": effective_generation,
        },
        "input_behavior": {
            "client_type": request.client_type,
            "input_shape": "messages" if request.messages else "prompt",
            "tools_supplied": bool(request.tools),
            "response_format_supplied": request.response_format is not None,
            "chat_template_checksum": identity.get("chat_template_checksum"),
        },
        "software": {
            "cognityx_inference_version": _package_version("cognityx-inference"),
            "cognityx_inference_git_revision": _git_revision(),
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "vllm_version": _package_version("vllm"),
        },
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        **document,
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _git_revision() -> str | None:
    configured = os.environ.get("COGNITYX_INFERENCE_GIT_REVISION")
    if configured:
        return configured
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
