"""Background hardware discovery for one local model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from cognityx_inference.backends.legacy import discover_model_context_limit
from cognityx_inference.capabilities import (
    CertifiedInferenceProfile,
    RuntimeCompatibility,
    hardware_fingerprint,
    new_profile_id,
)
from cognityx_inference.contracts import InferenceRequest, LoadPolicy
from cognityx_inference.lifecycle import ModelIdentity, ModelManager
from cognityx_inference.service import _backend_version
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    CertifiedProfileRepository,
)
from cognityx_jobs import JobRepository


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Safe finite candidates and acceptance thresholds."""

    context_candidates: tuple[int, ...] = (
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    )
    prompt: str = "Briefly explain why model context capacity matters."
    max_tokens: int = 64
    trial_timeout_seconds: float = 300
    minimum_tokens_per_second: float = 1


@dataclass(slots=True)
class DiscoveryJob:
    """Mutable in-memory status backed by append-only stored evidence."""

    job_id: str
    model: str
    backend: str
    profile: str
    state: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    certified_profile_id: str | None = None
    error: str | None = None
    trials: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def emit(self, event: str, **data: Any) -> None:
        self.events.append({"sequence": len(self.events) + 1, "event": event, **data})


class BoundaryDiscoveryCoordinator:
    """Run an approved context-boundary search and keep its winner loaded."""

    def __init__(
        self,
        models: ModelManager,
        profiles: CertifiedProfileRepository,
        artifacts: BoundaryArtifactRepository,
        inventory: Callable[[], Mapping[str, Any]],
        config: DiscoveryConfig | None = None,
        jobs: JobRepository | None = None,
    ) -> None:
        self.models = models
        self.profiles = profiles
        self.artifacts = artifacts
        self.inventory = inventory
        self.config = config or DiscoveryConfig()
        self.jobs = jobs or JobRepository()
        self._jobs: dict[str, DiscoveryJob] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
        model: str,
        backend: str,
        profile: str,
        model_context_limit: int | None,
        runtime: Mapping[str, Any],
    ) -> str:
        job = DiscoveryJob(str(uuid.uuid4()), model, backend, profile)
        with self._lock:
            self._jobs[job.job_id] = job
        self.jobs.create(job.job_id, "hardware_boundary_discovery", job.to_dict())
        thread = threading.Thread(
            target=self._run,
            args=(job, model_context_limit, dict(runtime)),
            daemon=True,
        )
        thread.start()
        return job.job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job is not None else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in {"completed", "failed", "cancelled"}:
                return False
            job.cancel_requested = True
            job.emit("cancel_requested")
            self.jobs.request_cancel(job_id)
            return True

    def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        persisted = self.jobs.events(job_id, after)
        if persisted:
            return persisted
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return []
            return [item for item in job.events if item["sequence"] > after]

    def _run(
        self,
        job: DiscoveryJob,
        supplied_model_limit: int | None,
        base_runtime: dict[str, Any],
    ) -> None:
        job.state = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.emit("discovery_started", model=job.model, backend=job.backend, profile=job.profile)
        self.jobs.set_state(job.job_id, "running")
        try:
            discovered_limit, revision = discover_model_context_limit(
                job.model,
                allow_download=bool(base_runtime.get("allow_download", False)),
            )
            model_limit = supplied_model_limit or discovered_limit
            if model_limit is None:
                raise RuntimeError("Model context limit is unavailable.")
            candidates = tuple(
                value
                for value in self.config.context_candidates
                if value <= model_limit
            )
            if model_limit not in candidates:
                candidates = (*candidates, model_limit)
            inventory = dict(self.inventory())
            compatibility = RuntimeCompatibility(
                hardware_fingerprint=hardware_fingerprint(inventory),
                model=job.model,
                model_revision=revision,
                backend=job.backend,
                backend_version=_backend_version(job.backend),
                profile=job.profile,
                kv_cache_precision=str(
                    base_runtime.get("kv_cache_precision", "auto")
                ),
                tensor_parallelism=int(
                    base_runtime.get("tensor_parallelism", 1)
                ),
            )
            self.artifacts.save_manifest(
                job.job_id,
                {
                    "job_id": job.job_id,
                    "type": "automatic_context_discovery",
                    "compatibility": asdict(compatibility),
                    "model_context_limit": model_limit,
                    "context_candidates": list(candidates),
                },
            )
            successful: list[dict[str, Any]] = []
            for number, context_length in enumerate(candidates, 1):
                if job.cancel_requested:
                    job.state = "cancelled"
                    job.emit("discovery_cancelled")
                    self.jobs.set_state(job.job_id, "cancelled")
                    return
                job.emit("trial_started", trial_id=f"trial-{number:03d}", trial_index=number, total_trials=len(candidates), context_length=context_length)
                runtime = {
                    **base_runtime,
                    "quantization": job.profile,
                    "context_length": context_length,
                }
                identity = ModelIdentity.create(job.model, job.backend, runtime)
                started = time.monotonic()
                generation_started: float | None = None
                load_seconds: float | None = None
                try:
                    with self.models.acquire(
                        job.model, job.backend, runtime
                    ) as backend:
                        load_seconds = time.monotonic() - started
                        generation_started = time.monotonic()
                        response = backend.infer(
                            InferenceRequest(
                                model=job.model,
                                prompt=self.config.prompt,
                                backend=job.backend,
                                profile=job.profile,
                                load_policy=LoadPolicy.REQUIRE_LOADED,
                                max_tokens=self.config.max_tokens,
                            )
                        )
                    generation_seconds = time.monotonic() - (
                        generation_started or started
                    )
                    elapsed = time.monotonic() - started
                    speed = response.timings.tokens_per_second
                    acceptable = (
                        generation_seconds <= self.config.trial_timeout_seconds
                        and (
                            speed is None
                            or speed >= self.config.minimum_tokens_per_second
                        )
                    )
                    status = "completed" if acceptable else "slow"
                    trial = {
                        "trial_id": f"trial-{number:03d}",
                        "status": status,
                        "configuration": {
                            "model": job.model,
                            "backend": job.backend,
                            "profile": job.profile,
                            "context_length": context_length,
                            **base_runtime,
                        },
                        "runtime_seconds": elapsed,
                        "metrics": {
                            "model_loading_seconds": load_seconds,
                            "generation_seconds": generation_seconds,
                            "tokens_per_second": speed,
                            "time_to_first_token_seconds": (
                                response.timings.time_to_first_token_seconds
                            ),
                        },
                    }
                except Exception as exc:
                    trial = {
                        "trial_id": f"trial-{number:03d}",
                        "status": (
                            "out_of_memory"
                            if "out of memory" in str(exc).lower()
                            else "failed"
                        ),
                        "configuration": {
                            "model": job.model,
                            "backend": job.backend,
                            "profile": job.profile,
                            "context_length": context_length,
                            **base_runtime,
                        },
                        "runtime_seconds": time.monotonic() - started,
                        "metrics": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                finally:
                    self.models.unload(identity, wait=True)
                job.trials.append(trial)
                job.emit("trial_completed", trial=trial, completed_trials=len(job.trials), total_trials=len(candidates))
                self.jobs.append_event(job.job_id, "trial_completed", {"trial": trial, "completed_trials": len(job.trials), "total_trials": len(candidates)})
                self.artifacts.save_trial(
                    job.job_id, trial["trial_id"], trial
                )
                if trial["status"] != "completed":
                    break
                successful.append(trial)
            if not successful:
                raise RuntimeError("No discovery candidate completed successfully.")
            best = successful[-1]
            metrics = best["metrics"]
            certified = CertifiedInferenceProfile(
                profile_id=new_profile_id(compatibility),
                created_at=datetime.now(timezone.utc).isoformat(),
                compatibility=compatibility,
                model_context_limit=model_limit,
                maximum_certified_context_length=best["configuration"][
                    "context_length"
                ],
                maximum_certified_batch_size=1,
                gpu_memory_utilization=base_runtime.get(
                    "gpu_memory_utilization", 0.9
                ),
                minimum_observed_tokens_per_second=metrics.get(
                    "tokens_per_second"
                ),
                maximum_time_to_first_token_seconds=metrics.get(
                    "time_to_first_token_seconds"
                ),
                evidence_job_id=job.job_id,
            )
            stored = self.profiles.save(certified)
            summary = {
                "job_id": job.job_id,
                "outcomes": job.trials,
                "certified_profile": certified.to_dict(),
                "certified_profile_key": getattr(stored, "key", None),
            }
            self.artifacts.save_summary(job.job_id, summary)
            ready_runtime = {
                **base_runtime,
                "quantization": job.profile,
                "context_length": certified.maximum_certified_context_length,
                "certified_profile_id": certified.profile_id,
            }
            ready_runtime.setdefault(
                "gpu_memory_utilization",
                certified.gpu_memory_utilization or 0.9,
            )
            self.models.load(job.model, job.backend, ready_runtime)
            job.certified_profile_id = certified.profile_id
            job.state = "completed"
            job.emit("discovery_completed", certified_profile_id=certified.profile_id)
            self.jobs.set_state(job.job_id, "completed")
        except Exception as exc:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.emit("discovery_failed", error=job.error)
            self.jobs.set_state(job.job_id, "failed")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()
