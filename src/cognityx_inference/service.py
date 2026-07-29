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
    ) -> None:
        self.models = models
        self.providers = dict(providers or {})
        self.artifacts = artifacts
        self.certified_profiles = certified_profiles
        self.inventory = inventory or (lambda: {})
        self.discovery = discovery
        self.provider_registry = provider_registry

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
        *,
        owner_id: str = "local",
    ) -> InferenceResponse:
        dispatched_request = request
        if request.provider != "local":
            try:
                provider = self.providers[request.provider]
            except KeyError as exc:
                raise ValueError(f"Unknown provider: {request.provider}") from exc
            self._validate_capabilities(request, provider)
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
                certified = self._profile_for_loaded_lease(lease)
            else:
                runtime, context_resolution, certified = self.prepare_runtime(
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
                response = backend.infer(dispatched_request, on_text=on_text)
                if budget is not None:
                    response = replace(response, token_budget=budget)
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
