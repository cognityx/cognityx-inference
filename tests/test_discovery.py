from __future__ import annotations

import time

from cognityx_inference.contracts import (
    FinishReason,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
)
from cognityx_inference.discovery import (
    BoundaryDiscoveryCoordinator,
    DiscoveryConfig,
)
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    CertifiedProfileRepository,
)
from cognityx_storage import LocalStorageBackend, StorageClient


class FakeBackend:
    capabilities = ModelCapabilities()

    def __init__(self, model, runtime):
        self.model = model
        self.runtime = runtime

    def load(self):
        pass

    def unload(self):
        pass

    def infer(self, request, on_text=None):
        return InferenceResponse(
            request_id="request-1",
            content="answer",
            model=self.model,
            provider="local",
            backend="fake",
            finish_reason=FinishReason.STOP,
            timings=InferenceTimings(
                time_to_first_token_seconds=0.1,
                tokens_per_second=10,
            ),
        )


def test_discovery_saves_profile_and_keeps_winner_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cognityx_inference.discovery.discover_model_context_limit",
        lambda *args, **kwargs: (256, "revision-1"),
    )
    monkeypatch.setattr(
        "cognityx_inference.discovery._backend_version",
        lambda backend: "1.0",
    )
    storage = StorageClient(LocalStorageBackend(tmp_path)).for_shared_data()
    profiles = CertifiedProfileRepository(storage)
    models = ModelManager({"fake": FakeBackend})
    coordinator = BoundaryDiscoveryCoordinator(
        models,
        profiles,
        BoundaryArtifactRepository(storage),
        lambda: {"gpu_name": "GPU", "gpu_total_memory_gb": 32},
        DiscoveryConfig(
            context_candidates=(128, 256),
            max_tokens=4,
            trial_timeout_seconds=10,
        ),
    )

    job_id = coordinator.start(
        model="model-a",
        backend="fake",
        profile="bf16",
        model_context_limit=256,
        runtime={},
    )
    deadline = time.monotonic() + 2
    while coordinator.status(job_id)["state"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    status = coordinator.status(job_id)
    assert status["state"] == "completed"
    assert status["certified_profile_id"]
    assert models.statuses()[0].identity.model == "model-a"
    assert all(
        trial["metrics"]["model_loading_seconds"] is not None
        and trial["metrics"]["generation_seconds"] >= 0
        for trial in status["trials"]
    )


def test_discovery_events_and_cancellation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cognityx_inference.discovery.discover_model_context_limit",
        lambda *args, **kwargs: (256, "revision-1"),
    )
    monkeypatch.setattr("cognityx_inference.discovery._backend_version", lambda _: "1")
    storage = StorageClient(LocalStorageBackend(tmp_path)).for_shared_data()
    coordinator = BoundaryDiscoveryCoordinator(
        ModelManager({"fake": FakeBackend}),
        CertifiedProfileRepository(storage),
        BoundaryArtifactRepository(storage),
        lambda: {"gpu_name": "GPU"},
        DiscoveryConfig(context_candidates=(128, 256)),
    )
    job_id = coordinator.start(model="model-a", backend="fake", profile="bf16", model_context_limit=256, runtime={})
    assert coordinator.cancel(job_id) is True
    deadline = time.monotonic() + 2
    while coordinator.status(job_id)["state"] not in {"cancelled", "completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert coordinator.status(job_id)["state"] == "cancelled"
    assert any(event["event"] == "cancel_requested" for event in coordinator.events(job_id))
