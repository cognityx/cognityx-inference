from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from cognityx_inference.contracts import (
    DiscoveryPolicy,
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
    LoadPolicy,
)
from cognityx_inference.capabilities import (
    CertifiedInferenceProfile,
    HardwareDiscoveryRequired,
    RuntimeCompatibility,
    hardware_fingerprint,
)
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.service import InferenceService, ModelNotLoadedError


@dataclass
class FakeBackend:
    model: str
    runtime: dict
    capabilities = ModelCapabilities(lifecycle=True, top_k=True)

    def load(self):
        pass

    def unload(self):
        pass

    def infer(self, request, on_text=None):
        return InferenceResponse(
            request_id="request-1",
            content="answer",
            model=request.model,
            provider="local",
            backend="fake",
        )


def test_explicit_unsupported_parameter_fails_clearly() -> None:
    service = InferenceService(ModelManager({"fake": FakeBackend}))

    with pytest.raises(ValueError, match="seed"):
        service.infer(
            InferenceRequest(
                model="model-a",
                prompt="hello",
                backend="fake",
                seed=42,
            )
        )


def test_supported_parameter_uses_normal_path() -> None:
    service = InferenceService(ModelManager({"fake": FakeBackend}))

    response = service.infer(
        InferenceRequest(
            model="model-a",
            prompt="hello",
            backend="fake",
            top_k=40,
        )
    )

    assert response.content == "answer"


def test_require_loaded_never_autoloads() -> None:
    service = InferenceService(ModelManager({"fake": FakeBackend}))

    with pytest.raises(ModelNotLoadedError):
        service.infer(
            InferenceRequest(
                model="model-a",
                prompt="hello",
                backend="fake",
                load_policy=LoadPolicy.REQUIRE_LOADED,
            )
        )


def test_missing_certification_returns_discovery_requirement(monkeypatch) -> None:
    monkeypatch.setattr(
        "cognityx_inference.service.discover_model_context_limit",
        lambda *args, **kwargs: (32768, "revision-1"),
    )

    class Profiles:
        def find_compatible(self, compatibility):
            return None

    service = InferenceService(
        ModelManager({"fake": FakeBackend}),
        certified_profiles=Profiles(),
        inventory=lambda: {"gpu_name": "GPU"},
    )

    with pytest.raises(HardwareDiscoveryRequired) as captured:
        service.infer(
            InferenceRequest(
                model="model-a",
                prompt="hello",
                backend="fake",
                discovery_policy=DiscoveryPolicy.REQUIRE_EXISTING,
            )
        )

    assert captured.value.model_context_limit == 32768
    assert captured.value.discovery_request_id is None


def test_certified_context_is_injected_into_runtime(monkeypatch) -> None:
    inventory = {"gpu_name": "GPU"}
    monkeypatch.setattr(
        "cognityx_inference.service.discover_model_context_limit",
        lambda *args, **kwargs: (32768, "revision-1"),
    )
    monkeypatch.setattr(
        "cognityx_inference.service._backend_version",
        lambda backend: "1.0",
    )
    compatible = RuntimeCompatibility(
        hardware_fingerprint=hardware_fingerprint(inventory),
        model="model-a",
        model_revision="revision-1",
        backend="fake",
        backend_version="1.0",
        profile="bf16",
        kv_cache_precision="auto",
    )
    certified = CertifiedInferenceProfile(
        profile_id="profile-1",
        created_at=datetime.now(timezone.utc).isoformat(),
        compatibility=compatible,
        model_context_limit=32768,
        maximum_certified_context_length=8192,
    )

    class Profiles:
        def find_compatible(self, compatibility):
            return certified if compatibility == compatible else None

    service = InferenceService(
        ModelManager({"fake": FakeBackend}),
        certified_profiles=Profiles(),
        inventory=lambda: inventory,
    )

    response = service.infer(
        InferenceRequest(model="model-a", prompt="hello", backend="fake")
    )

    assert response.extensions["context_resolution"]["effective_limit"] == 8192
    status = service.models.statuses()[0]
    assert ("context_length", "8192") in status.identity.runtime
