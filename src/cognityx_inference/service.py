"""Application service that routes normalized inference requests."""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Mapping, Protocol

from cognityx_inference.environment import configure_huggingface_cache

# Hugging Face Hub snapshots its cache constants during import. Configure the
# shared cache before importing the backend module (which imports transformers).
configure_huggingface_cache()

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
)
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.model_references import resolve_local_model_reference
from cognityx_inference.storage import (
    CertifiedProfileRepository,
    InferenceArtifactRepository,
)


class Provider(Protocol):
    def infer(self, request: InferenceRequest) -> InferenceResponse: ...


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
    ) -> None:
        self.models = models
        self.providers = dict(providers or {})
        self.artifacts = artifacts
        self.certified_profiles = certified_profiles
        self.inventory = inventory or (lambda: {})
        self.discovery = discovery

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
        *,
        owner_id: str = "local",
    ) -> InferenceResponse:
        if request.provider != "local":
            try:
                provider = self.providers[request.provider]
            except KeyError as exc:
                raise ValueError(f"Unknown provider: {request.provider}") from exc
            self._validate_capabilities(request, provider)
            response = provider.infer(request)
        else:
            canonical_model = resolve_local_model_reference(request.model).resolved
            if canonical_model != request.model:
                request = replace(request, model=canonical_model)
            if request.load_policy is LoadPolicy.REQUIRE_LOADED:
                try:
                    lease = self.models.acquire_loaded(request.model)
                except LookupError as exc:
                    raise ModelNotLoadedError(str(exc)) from exc
                context_resolution = None
            else:
                runtime, context_resolution = self.prepare_runtime(
                    request.model,
                    request.backend,
                    request.profile,
                    dict(request.extensions.get("runtime", {})),
                    request.required_context_length,
                    request.discovery_policy,
                    owner_id,
                )
                lease = self.models.acquire(request.model, request.backend, runtime)
            with lease as backend:
                self._validate_capabilities(request, backend)
                response = backend.infer(request, on_text=on_text)
            if context_resolution is not None:
                response = replace(
                    response,
                    extensions={
                        **response.extensions,
                        "context_resolution": asdict(context_resolution),
                    },
                )
        if self.artifacts is not None:
            self.artifacts.save_exchange(request, response)
        return response

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
        resolved, context = self.prepare_runtime(
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

    def prepare_runtime(
        self,
        model: str,
        backend: str,
        profile: str,
        runtime: dict[str, Any],
        required_context_length: int | None,
        discovery_policy: DiscoveryPolicy,
        owner_id: str = "local",
    ) -> tuple[dict[str, Any], Any | None]:
        """Resolve context from model metadata and compatible certification."""
        runtime["quantization"] = profile
        if self.certified_profiles is None:
            return runtime, None
        model_limit, revision = discover_model_context_limit(
            model,
            allow_download=bool(runtime.get("allow_download", False)),
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
        certified = self.certified_profiles.find_compatible(compatibility)
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
        runtime.setdefault(
            "gpu_memory_utilization", certified.gpu_memory_utilization or 0.9
        )
        runtime["certified_profile_id"] = certified.profile_id
        return runtime, context

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
                request.log_probabilities
                or request.top_log_probabilities is not None
            ),
            "reasoning": bool(request.reasoning),
        }
        unsupported = [
            name
            for name, enabled in requested.items()
            if enabled and not getattr(capabilities, name)
        ]
        if unsupported:
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
