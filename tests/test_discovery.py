from __future__ import annotations

import time

from cognityx_inference.contracts import (
    FinishReason,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
    TokenUsage,
)
from cognityx_inference.capabilities import RuntimeCompatibility, hardware_fingerprint
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
                prompt_processing_seconds=0.02,
                token_generation_seconds=0.1,
            ),
            usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
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


def test_discovery_preserves_winning_runtime_and_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cognityx_inference.discovery.discover_model_context_limit",
        lambda *args, **kwargs: (256, "revision-1"),
    )
    monkeypatch.setattr("cognityx_inference.discovery._backend_version", lambda _: "1")

    class Monitor:
        def start(self): pass
        def mark_phase(self, phase): pass
        def stop(self):
            return {
                "host_cpu_average_percent": 10,
                "host_cpu_peak_percent": 20,
                "gpu_usage": {
                    "dedicated_memory_used_bytes_peak": 100,
                    "shared_memory_used_bytes_peak": 50,
                    "temperature_celsius_peak": 70,
                    "power_watts_peak": 250,
                },
                "phases": {
                    "model_loading": {
                        "host_cpu_peak_percent": 12,
                        "gpu_usage": {
                            "dedicated_memory_used_bytes_average": 90,
                            "dedicated_memory_used_bytes_peak": 100,
                            "power_watts_average": 200,
                            "power_watts_peak": 250,
                            "power_limit_watts": 575,
                        },
                    },
                },
            }

    monkeypatch.setattr("cognityx_inference.discovery.ResourceMonitor", lambda **_: Monitor())
    class FailingAutoBackend(FakeBackend):
        def infer(self, request, on_text=None):
            if self.runtime.get("kv_cache_precision") == "auto" and self.runtime.get("context_length") == 256:
                raise RuntimeError("out of memory")
            return super().infer(request, on_text)

    storage = StorageClient(LocalStorageBackend(tmp_path)).for_shared_data()
    profiles = CertifiedProfileRepository(storage)
    models = ModelManager({"fake": FailingAutoBackend})
    coordinator = BoundaryDiscoveryCoordinator(
        models, profiles, BoundaryArtifactRepository(storage),
        lambda: {"gpu_name": "GPU", "gpu_total_memory_gb": 32},
        DiscoveryConfig(
            context_candidates=(128, 256), kv_cache_precisions=("auto", "fp8"),
            generation_lengths=(4, 8), system_prompt="system", prompt="user",
        ),
    )
    job_id = coordinator.start(model="model-a", backend="fake", profile="int4", model_context_limit=256, runtime={})
    deadline = time.monotonic() + 2
    while coordinator.status(job_id)["state"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    status = coordinator.status(job_id)
    assert status["state"] == "completed"
    loaded = models.statuses()[0]
    runtime = dict(loaded.identity.runtime)
    assert runtime["kv_cache_precision"] == "'fp8'"
    profile = profiles.find_compatible(
        RuntimeCompatibility(
            hardware_fingerprint=hardware_fingerprint({"gpu_name": "GPU", "gpu_total_memory_gb": 32}),
            model="model-a", model_revision="revision-1", backend="fake", backend_version="1",
            profile="int4", kv_cache_precision="fp8",
        )
    )
    assert profile is not None
    assert profile.certified_configuration["kv_cache_precision"] == "fp8"
    assert profile.workload["system_prompt"] == "system"
    assert profile.token_breakdown["prompt_tokens"] == 4
    assert profile.resource_summary["gpu_usage"]["power_watts_peak"] == 250
    assert profile.certified_trial["configuration"]["kv_cache_precision"] == "fp8"
    assert (
        profile.certified_trial["metrics"]["resource_summary"]["phases"]
        ["model_loading"]["gpu_usage"]["power_limit_watts"]
        == 575
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
