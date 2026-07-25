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
from cognityx_jobs import JobRepository
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
    events = coordinator.events(job_id)
    assert events[-1]["event"] == "discovery_completed"
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
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


def test_discovery_jobs_are_scoped_to_their_owner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cognityx_inference.discovery.discover_model_context_limit",
        lambda *args, **kwargs: (128, "revision-1"),
    )
    monkeypatch.setattr("cognityx_inference.discovery._backend_version", lambda _: "1")
    storage = StorageClient(LocalStorageBackend(tmp_path)).for_shared_data()
    coordinator = BoundaryDiscoveryCoordinator(
        ModelManager({"fake": FakeBackend}),
        CertifiedProfileRepository(storage),
        BoundaryArtifactRepository(storage),
        lambda: {"gpu_name": "GPU"},
        DiscoveryConfig(context_candidates=(128,)),
        jobs=JobRepository(str(tmp_path / "jobs.sqlite3")),
    )
    job_id = coordinator.start(
        model="model-a",
        backend="fake",
        profile="bf16",
        owner_id="alice",
        model_context_limit=128,
        runtime={},
    )

    assert coordinator.status(job_id, owner_id="bob") is None
    assert coordinator.list("bob") == []
    assert coordinator.cancel(job_id, owner_id="bob") is False
    assert coordinator.status(job_id, owner_id="alice") is not None
    assert [job["job_id"] for job in coordinator.list("alice")] == [job_id]


def test_job_repository_marks_orphaned_workers_interrupted(tmp_path) -> None:
    jobs = JobRepository(str(tmp_path / "jobs.sqlite3"))
    jobs.create("job-1", "hardware_boundary_discovery", {}, owner_id="alice")
    jobs.set_state("job-1", "running")

    assert jobs.mark_interrupted("hardware_boundary_discovery") == 1
    assert jobs.get("job-1").state == "interrupted"
