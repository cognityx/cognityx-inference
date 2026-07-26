from __future__ import annotations

from datetime import datetime, timezone

from cognityx_inference.capabilities import (
    CertifiedInferenceProfile,
    HardwareDiscoveryRequired,
    RuntimeCompatibility,
)
from cognityx_inference.contracts import DiscoveryPolicy
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.service import InferenceService
from cognityx_inference.storage import CertifiedProfileRepository
from cognityx_storage import LocalStorageBackend, StorageClient


def test_certified_profile_round_trip_uses_logical_storage(tmp_path) -> None:
    storage = StorageClient(LocalStorageBackend(tmp_path)).for_shared_data()
    repository = CertifiedProfileRepository(storage)
    compatibility = RuntimeCompatibility(
        hardware_fingerprint="gpu-a",
        model="org/model-a",
        model_revision="revision-1",
        backend="vllm",
        backend_version="1.0",
        profile="int4",
        kv_cache_precision="auto",
    )
    profile = CertifiedInferenceProfile(
        profile_id="profile-1",
        created_at=datetime.now(timezone.utc).isoformat(),
        compatibility=compatibility,
        model_context_limit=32768,
        maximum_certified_context_length=8192,
    )

    stored = repository.save(profile)
    loaded = repository.find_compatible(compatibility)

    assert stored.key.startswith(
        "shared/evaluations/hardware-boundary/certified-profiles/"
    )
    assert loaded == profile
    assert repository.list_profiles(model="org/model-a") == [profile]
    assert repository.get_profile("profile-1") == profile


def test_missing_certification_prefix_means_discovery_required(
    tmp_path, monkeypatch
) -> None:
    storage = StorageClient(LocalStorageBackend(tmp_path)).for_shared_data()
    repository = CertifiedProfileRepository(storage)
    monkeypatch.setattr(
        "cognityx_inference.service.discover_model_context_limit",
        lambda *args, **kwargs: (40960, "revision-1"),
    )
    monkeypatch.setattr(
        "cognityx_inference.service._backend_version",
        lambda backend: "1.0",
    )

    class Discovery:
        def start(self, **kwargs):
            return "discovery-1"

    service = InferenceService(
        ModelManager({}),
        certified_profiles=repository,
        inventory=lambda: {"gpu_name": "GPU", "gpu_total_memory_gb": 32},
        discovery=Discovery(),
    )

    try:
        service.load_model(
            "org/model-a",
            "vllm",
            "int4",
            discovery_policy=DiscoveryPolicy.AUTO,
        )
    except HardwareDiscoveryRequired as exc:
        assert exc.discovery_request_id == "discovery-1"
        assert exc.model_context_limit == 40960
    else:
        raise AssertionError("missing certification did not require discovery")
