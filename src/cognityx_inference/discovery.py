"""Background hardware discovery for one local model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import copy
import threading
import time
import tomllib
from typing import Any, Callable, Mapping
import uuid

from cognityx_inference.backends.legacy import discover_model_context_limit
from cognityx_inference.capabilities import (
    CertifiedContextProfile,
    CertifiedInferenceProfile,
    RuntimeCompatibility,
    hardware_fingerprint,
    new_profile_id,
)
from cognityx_inference.contracts import InferenceRequest, LoadPolicy, ThinkingMode
from cognityx_inference.lifecycle import ModelIdentity, ModelManager
from cognityx_inference.service import _backend_version
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    CertifiedProfileRepository,
)
from cognityx_inference.telemetry import ResourceMonitor, read_windows_bridge
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
    system_prompt: str | None = None
    enable_thinking: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    max_tokens: int = 64
    trial_timeout_seconds: float = 300
    minimum_tokens_per_second: float = 1
    kv_cache_precisions: tuple[str, ...] = ("auto",)
    generation_lengths: tuple[int, ...] = (64,)
    telemetry_interval_seconds: float = 0.25
    windows_bridge_path: str | None = None
    windows_bridge_max_age_seconds: float = 5.0

    @classmethod
    def from_toml(cls, path: Path) -> "DiscoveryConfig":
        with path.open("rb") as source:
            document = tomllib.load(source)
        axes = document["search"]["axes"]
        limits = document.get("limits") or {}
        workload = document.get("workload") or {}
        telemetry = document.get("telemetry") or {}
        return cls(
            context_candidates=tuple(axes.get("context_length", cls.context_candidates)),
            kv_cache_precisions=tuple(axes.get("kv_cache_precision", ("auto",))),
            generation_lengths=tuple(axes.get("generation_length", (64,))),
            trial_timeout_seconds=float(limits.get("request_timeout_seconds", 300)),
            minimum_tokens_per_second=float(limits.get("minimum_tokens_per_second", 1)),
            prompt=str(workload.get("user_prompt", cls.prompt)),
            system_prompt=workload.get("system_prompt"),
            enable_thinking=bool(workload.get("enable_thinking", False)),
            temperature=workload.get("temperature"),
            top_p=workload.get("top_p"),
            top_k=workload.get("top_k"),
            min_p=workload.get("min_p"),
            seed=workload.get("seed"),
            telemetry_interval_seconds=float(telemetry.get("interval_seconds", 0.25)),
            windows_bridge_path=telemetry.get("windows_bridge_path"),
            windows_bridge_max_age_seconds=float(telemetry.get("windows_bridge_max_age_seconds", 5)),
        )


@dataclass(slots=True)
class DiscoveryJob:
    """Mutable in-memory status backed by append-only stored evidence."""

    job_id: str
    model: str
    backend: str
    profile: str
    owner_id: str = "local"
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
        self.jobs.mark_interrupted("hardware_boundary_discovery")
        self._jobs: dict[str, DiscoveryJob] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
        model: str,
        backend: str,
        profile: str,
        owner_id: str = "local",
        model_context_limit: int | None,
        runtime: Mapping[str, Any],
    ) -> str:
        job = DiscoveryJob(
            str(uuid.uuid4()), model, backend, profile, owner_id=owner_id
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self.jobs.create(
            job.job_id,
            "hardware_boundary_discovery",
            job.to_dict(),
            owner_id=owner_id,
        )
        thread = threading.Thread(
            target=self._run,
            args=(job, model_context_limit, dict(runtime)),
            daemon=True,
        )
        thread.start()
        return job.job_id

    def status(self, job_id: str, *, owner_id: str = "local") -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job.to_dict() if job.owner_id == owner_id else None
        try:
            record = self.jobs.get_for_owner(job_id, owner_id)
        except KeyError:
            return None
        return self._stored_status(record)

    def list(self, owner_id: str, *, include_history: bool = False) -> list[dict[str, Any]]:
        states = None if include_history else (
            "queued",
            "running",
            "cancellation_requested",
        )
        records = self.jobs.list_for_owner(owner_id, states=states)
        return [self.status(record.job_id, owner_id=owner_id) for record in records]

    def cancel(self, job_id: str, *, owner_id: str = "local") -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if (
                job is None
                or job.owner_id != owner_id
                or job.state in {"completed", "failed", "cancelled"}
            ):
                return False
            job.cancel_requested = True
            job.emit("cancel_requested")
            self.jobs.request_cancel(job_id)
            return True

    def events(
        self, job_id: str, after: int = 0, *, owner_id: str = "local"
    ) -> list[dict[str, Any]]:
        if self.status(job_id, owner_id=owner_id) is None:
            return []
        persisted = self.jobs.events(job_id, after)
        return persisted

    def _emit(self, job: DiscoveryJob, event: str, **data: Any) -> None:
        """Publish one event to both the live view and durable replay log."""
        sequence = self.jobs.append_event(job.job_id, event, data)
        job.events.append({"sequence": sequence, "event": event, **data})

    @staticmethod
    def _stored_status(record: Any) -> dict[str, Any]:
        status = dict(record.payload)
        status["job_id"] = record.job_id
        status["state"] = record.state
        status["owner_id"] = record.owner_id
        return status

    def _run(
        self,
        job: DiscoveryJob,
        supplied_model_limit: int | None,
        base_runtime: dict[str, Any],
    ) -> None:
        loaded_identity: ModelIdentity | None = None
        keep_loaded = False
        job.state = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._emit(
            job,
            "discovery_started",
            model=job.model,
            backend=job.backend,
            profile=job.profile,
        )
        self.jobs.set_state(job.job_id, "running")
        try:
            discovered_limit, revision = discover_model_context_limit(
                job.model,
                allow_download=bool(base_runtime.get("allow_download", False)),
            )
            model_limit = supplied_model_limit or discovered_limit
            if model_limit is None:
                raise RuntimeError("Model context limit is unavailable.")
            contexts = tuple(value for value in self.config.context_candidates if value <= model_limit)
            if model_limit not in contexts:
                contexts = (*contexts, model_limit)
            # Keep all request variants for an engine identity together.  This
            # avoids reloading a 100+ second model merely to change max_tokens.
            candidates = tuple(
                {
                    "context_length": context_length,
                    "kv_cache_precision": kv_cache_precision,
                    "generation_length": generation_length,
                    "batch_size": int(base_runtime.get("batch_size", 1)),
                }
                for kv_cache_precision, context_length, generation_length in product(
                    self.config.kv_cache_precisions, contexts, self.config.generation_lengths
                )
            )
            inventory = dict(self.inventory())
            self.artifacts.save_manifest(
                job.job_id,
                {
                    "job_id": job.job_id,
                    "type": "automatic_context_discovery",
                    "base_runtime": base_runtime,
                    "model_context_limit": model_limit,
                    "candidates": list(candidates),
                    "workload": self._workload_evidence(),
                },
            )
            successful: list[dict[str, Any]] = []
            context_boundaries: set[tuple[str, int]] = set()
            for number, candidate in enumerate(candidates, 1):
                context_length = candidate["context_length"]
                boundary_key = (
                    candidate["kv_cache_precision"],
                    candidate["generation_length"],
                )
                if boundary_key in context_boundaries:
                    continue
                if job.cancel_requested:
                    job.state = "cancelled"
                    self._emit(job, "discovery_cancelled")
                    self.jobs.set_state(job.job_id, "cancelled")
                    return
                self._emit(
                    job,
                    "trial_started",
                    trial_id=f"trial-{number:03d}",
                    trial_index=number,
                    total_trials=len(candidates),
                    configuration=candidate,
                )
                runtime = {
                    **base_runtime,
                    "quantization": job.profile,
                    "context_length": context_length,
                    "kv_cache_precision": candidate["kv_cache_precision"],
                }
                identity = ModelIdentity.create(job.model, job.backend, runtime)
                # An engine cannot satisfy this request; avoid a costly load.
                # Prompt tokenization is backend-specific, so the backend still
                # remains authoritative for the final validation.
                if candidate["generation_length"] >= context_length:
                    trial = self._invalid_trial(job, number, candidate, base_runtime)
                    job.trials.append(trial)
                    self._emit(job, "trial_completed", trial=trial,
                               completed_trials=len(job.trials), total_trials=len(candidates))
                    self.artifacts.save_trial(job.job_id, trial["trial_id"], trial)
                    continue
                if loaded_identity is not None and loaded_identity != identity:
                    self.models.unload(loaded_identity, wait=True)
                    loaded_identity = None
                started = time.monotonic()
                monitor = self._new_monitor()
                monitor.start()
                load_seconds: float | None = None
                try:
                    monitor.mark_phase("model_loading")
                    status = self.models.load(job.model, job.backend, runtime)
                    load_seconds = status.load_seconds if loaded_identity is None else 0.0
                    loaded_identity = identity
                    monitor.mark_phase("inference")
                    with self.models.acquire(job.model, job.backend, runtime) as backend:
                        response = backend.infer(
                            self._request(job, candidate)
                        )
                    monitor.mark_phase("completed")
                    resource_summary = monitor.stop()
                    elapsed = time.monotonic() - started
                    speed = response.timings.tokens_per_second
                    generation_seconds = (
                        response.timings.token_generation_seconds
                        or response.timings.latency_seconds
                        or elapsed - (load_seconds or 0.0)
                    )
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
                            **base_runtime,
                            **candidate,
                        },
                        "runtime_seconds": elapsed,
                        "metrics": {
                            "model_loading_seconds": load_seconds,
                            "prompt_processing_seconds": response.timings.prompt_processing_seconds,
                            "generation_seconds": generation_seconds,
                            "tokens_per_second": speed,
                            "time_to_first_token_seconds": (
                                response.timings.time_to_first_token_seconds
                            ),
                            "usage": asdict(response.usage),
                            "token_breakdown": self._token_breakdown(response),
                            "resource_summary": resource_summary,
                        },
                    }
                except Exception as exc:
                    resource_summary = monitor.stop()
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
                            **base_runtime,
                            **candidate,
                        },
                        "runtime_seconds": time.monotonic() - started,
                        "metrics": {"resource_summary": resource_summary},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                job.trials.append(trial)
                self._emit(
                    job,
                    "trial_completed",
                    trial=trial,
                    completed_trials=len(job.trials),
                    total_trials=len(candidates),
                )
                self.artifacts.save_trial(
                    job.job_id, trial["trial_id"], trial
                )
                if trial["status"] != "completed":
                    context_boundaries.add(boundary_key)
                    continue
                successful.append(trial)
            if not successful:
                raise RuntimeError("No discovery candidate completed successfully.")
            # Prefer capacity first, then the largest proven completion budget.
            best = max(
                successful,
                key=lambda item: (
                    item["configuration"]["context_length"],
                    item["configuration"]["generation_length"],
                ),
            )
            metrics = best["metrics"]
            configuration = best["configuration"]
            compatibility = RuntimeCompatibility(
                hardware_fingerprint=hardware_fingerprint(inventory),
                model=job.model,
                model_revision=revision,
                backend=job.backend,
                backend_version=_backend_version(job.backend),
                profile=job.profile,
                kv_cache_precision=str(configuration["kv_cache_precision"]),
                tensor_parallelism=int(configuration.get("tensor_parallelism", 1)),
            )
            certified = CertifiedInferenceProfile(
                profile_id=new_profile_id(compatibility),
                created_at=datetime.now(timezone.utc).isoformat(),
                compatibility=compatibility,
                model_context_limit=model_limit,
                maximum_certified_context_length=best["configuration"][
                    "context_length"
                ],
                maximum_certified_batch_size=1,
                gpu_memory_utilization=configuration.get("gpu_memory_utilization", 0.9),
                minimum_observed_tokens_per_second=metrics.get(
                    "tokens_per_second"
                ),
                maximum_time_to_first_token_seconds=metrics.get(
                    "time_to_first_token_seconds"
                ),
                evidence_job_id=job.job_id,
                context=CertifiedContextProfile(
                    max_context_tokens=int(configuration["context_length"]),
                    max_output_tokens_limit=int(
                        configuration["generation_length"]
                    ),
                    tokenizer=job.model,
                ),
                certified_configuration=configuration,
                workload=self._workload_evidence(),
                token_breakdown=metrics.get("token_breakdown", {}),
                performance={
                    key: metrics.get(key)
                    for key in ("model_loading_seconds", "prompt_processing_seconds", "generation_seconds", "time_to_first_token_seconds", "tokens_per_second")
                },
                resource_summary=metrics.get("resource_summary", {}),
                certified_trial=copy.deepcopy(best),
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
                "kv_cache_precision": certified.compatibility.kv_cache_precision,
                "certified_profile_id": certified.profile_id,
            }
            ready_runtime.setdefault(
                "gpu_memory_utilization",
                certified.gpu_memory_utilization or 0.9,
            )
            ready_identity = ModelIdentity.create(job.model, job.backend, ready_runtime)
            if loaded_identity is not None and loaded_identity != ready_identity:
                self.models.unload(loaded_identity, wait=True)
            self.models.load(job.model, job.backend, ready_runtime)
            keep_loaded = True
            job.certified_profile_id = certified.profile_id
            job.state = "completed"
            self._emit(
                job,
                "discovery_completed",
                certified_profile_id=certified.profile_id,
            )
            self.jobs.set_state(job.job_id, "completed")
        except Exception as exc:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self._emit(job, "discovery_failed", error=job.error)
            self.jobs.set_state(job.job_id, "failed")
        finally:
            if loaded_identity is not None and not keep_loaded:
                self.models.unload(loaded_identity, wait=True)
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _request(self, job: DiscoveryJob, candidate: Mapping[str, Any]) -> InferenceRequest:
        messages: tuple[dict[str, str], ...] = (
            (({"role": "system", "content": self.config.system_prompt},) if self.config.system_prompt else ())
            + ({"role": "user", "content": self.config.prompt},)
        )
        return InferenceRequest(
            model=job.model, messages=messages, backend=job.backend, profile=job.profile,
            load_policy=LoadPolicy.REQUIRE_LOADED, max_tokens=int(candidate["generation_length"]),
            temperature=self.config.temperature, top_p=self.config.top_p,
            top_k=self.config.top_k, min_p=self.config.min_p, seed=self.config.seed,
            thinking=(
                ThinkingMode.ENABLED
                if self.config.enable_thinking
                else ThinkingMode.DISABLED
            ),
        )

    def _workload_evidence(self) -> dict[str, Any]:
        return {
            "system_prompt": self.config.system_prompt,
            "user_prompt": self.config.prompt,
            "reasoning": {"enabled": self.config.enable_thinking},
            "parameters": {"temperature": self.config.temperature, "top_p": self.config.top_p,
                           "top_k": self.config.top_k, "min_p": self.config.min_p, "seed": self.config.seed},
        }

    def _new_monitor(self) -> ResourceMonitor:
        bridge = self.config.windows_bridge_path
        return ResourceMonitor(
            interval_seconds=self.config.telemetry_interval_seconds,
            windows_sampler=(lambda: read_windows_bridge(bridge, max_age_seconds=self.config.windows_bridge_max_age_seconds)) if bridge else None,
        )

    def _token_breakdown(self, response: Any) -> dict[str, Any]:
        legacy = response.extensions.get("legacy_result", {}) if response.extensions else {}
        quality = legacy.get("quality_indicators", {}) if isinstance(legacy, Mapping) else {}
        return {
            "system_prompt_tokens": None,
            "user_prompt_tokens": None,
            "chat_template_overhead_tokens": None,
            "prompt_tokens": response.usage.prompt_tokens,
            "reasoning_tokens": quality.get("thinking_tokens"),
            "answer_tokens": quality.get("answer_tokens"),
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    def _invalid_trial(self, job: DiscoveryJob, number: int, candidate: Mapping[str, Any], base_runtime: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "trial_id": f"trial-{number:03d}", "status": "failed",
            "configuration": {"model": job.model, "backend": job.backend, "profile": job.profile, **base_runtime, **candidate},
            "runtime_seconds": 0.0, "metrics": {},
            "error": "Invalid configuration: generation_length must be smaller than context_length.",
        }
