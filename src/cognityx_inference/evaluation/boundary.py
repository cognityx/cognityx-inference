"""Caller-bounded Cartesian planning with contextual monotonic pruning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
import time
from typing import Any, Callable, Iterable, Mapping
import uuid

from cognityx_inference.capabilities import (
    CertifiedInferenceProfile,
    RuntimeCompatibility,
    new_profile_id,
)
from cognityx_inference.storage import BoundaryArtifactRepository
from cognityx_inference.storage import CertifiedProfileRepository

DEFAULT_LOAD_AXES = frozenset(
    {
        "model",
        "backend",
        "quantization",
        "context_length",
        "kv_cache_precision",
        "gpu_memory_utilization",
        "tensor_parallelism",
        "attention_implementation",
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    """A finite, explicitly supplied boundary search."""

    axes: Mapping[str, tuple[Any, ...]]
    ordered_axes: tuple[str, ...] = ()
    monotonic_axes: tuple[str, ...] = ()
    load_axes: frozenset[str] = DEFAULT_LOAD_AXES
    request_timeout_seconds: float | None = None
    first_token_timeout_seconds: float | None = None
    no_token_progress_timeout_seconds: float | None = None
    minimum_tokens_per_second: float | None = None
    slow_window_seconds: float = 10

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("at least one search axis is required")
        for name, values in self.axes.items():
            if not values:
                raise ValueError(f"axis {name} has no candidates")
            if len(set(map(repr, values))) != len(values):
                raise ValueError(f"axis {name} contains duplicate candidates")
        ordered = self.ordered_axes or tuple(self.axes)
        if set(ordered) != set(self.axes):
            raise ValueError("ordered_axes must contain every configured axis once")
        unknown = set(self.monotonic_axes) - set(self.axes)
        if unknown:
            raise ValueError(f"unknown monotonic axes: {sorted(unknown)}")
        for name in (
            "request_timeout_seconds",
            "first_token_timeout_seconds",
            "no_token_progress_timeout_seconds",
            "minimum_tokens_per_second",
            "slow_window_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def axis_order(self) -> tuple[str, ...]:
        return self.ordered_axes or tuple(self.axes)


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One immutable combination in a boundary search."""

    trial_id: str
    configuration: Mapping[str, Any]
    load_identity: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """One completed, failed, timed-out, or slow trial."""

    trial_id: str
    status: str
    configuration: Mapping[str, Any]
    runtime_seconds: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    boundary_axis: str | None = None
    boundary_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundaryPlanner:
    """Enumerate configured combinations model/load-identity major."""

    def __init__(self, config: BoundaryConfig) -> None:
        self.config = config

    def trials(self) -> tuple[TrialSpec, ...]:
        order = self.config.axis_order
        raw = []
        for index, values in enumerate(product(*(self.config.axes[name] for name in order)), 1):
            configuration = dict(zip(order, values))
            load_identity = tuple(
                (name, repr(configuration[name]))
                for name in order
                if name in self.config.load_axes
            )
            raw.append(
                TrialSpec(
                    trial_id=f"trial-{index:05d}",
                    configuration=configuration,
                    load_identity=load_identity,
                )
            )
        raw.sort(key=lambda item: item.load_identity)
        return tuple(raw)


@dataclass(frozen=True, slots=True)
class _Boundary:
    axis: str
    candidate_index: int
    value: Any
    context: tuple[tuple[str, str], ...]


class BoundaryEvaluator:
    """Run a finite plan and prune only dominated contextual candidates."""

    def __init__(
        self,
        config: BoundaryConfig,
        execute: Callable[[TrialSpec], TrialOutcome],
        artifacts: BoundaryArtifactRepository | None = None,
        certified_profiles: CertifiedProfileRepository | None = None,
        compatibility: RuntimeCompatibility | None = None,
        model_context_limit: int | None = None,
    ) -> None:
        self.config = config
        self.execute = execute
        self.artifacts = artifacts
        self.certified_profiles = certified_profiles
        self.compatibility = compatibility
        self.model_context_limit = model_context_limit

    def run(self, job_id: str | None = None) -> dict[str, Any]:
        job_id = job_id or str(uuid.uuid4())
        planner = BoundaryPlanner(self.config)
        trials = planner.trials()
        if self.artifacts is not None:
            self.artifacts.save_manifest(
                job_id,
                {
                    "job_id": job_id,
                    "axes": {key: list(value) for key, value in self.config.axes.items()},
                    "ordered_axes": list(self.config.axis_order),
                    "trial_count_before_pruning": len(trials),
                },
            )
        outcomes: list[TrialOutcome] = []
        pruned: list[dict[str, Any]] = []
        boundaries: list[_Boundary] = []
        for trial in trials:
            boundary = self._matching_boundary(trial.configuration, boundaries)
            if boundary is not None:
                pruned.append(
                    {
                        "trial_id": trial.trial_id,
                        "configuration": dict(trial.configuration),
                        "axis": boundary.axis,
                        "boundary_value": boundary.value,
                        "reason": "dominated_by_unacceptable_boundary",
                    }
                )
                continue
            started = time.monotonic()
            try:
                outcome = self.execute(trial)
            except TimeoutError as exc:
                outcome = TrialOutcome(
                    trial_id=trial.trial_id,
                    status="timeout",
                    configuration=trial.configuration,
                    runtime_seconds=time.monotonic() - started,
                    error=str(exc),
                )
            except Exception as exc:
                outcome = TrialOutcome(
                    trial_id=trial.trial_id,
                    status="failed",
                    configuration=trial.configuration,
                    runtime_seconds=time.monotonic() - started,
                    error=str(exc),
                )
            outcome = self._classify_limits(outcome)
            boundary_axis = self._boundary_axis(outcome)
            if boundary_axis is not None:
                outcome = TrialOutcome(
                    **{
                        **outcome.to_dict(),
                        "boundary_axis": boundary_axis,
                        "boundary_value": trial.configuration[boundary_axis],
                    }
                )
                boundaries.append(self._make_boundary(trial.configuration, boundary_axis))
            outcomes.append(outcome)
            if self.artifacts is not None:
                self.artifacts.save_trial(job_id, trial.trial_id, outcome.to_dict())
        summary = {
            "schema_version": "1.0",
            "job_id": job_id,
            "trial_count_before_pruning": len(trials),
            "executed_trial_count": len(outcomes),
            "outcomes": [item.to_dict() for item in outcomes],
            "pruned_candidates": pruned,
            "boundaries": [
                {
                    "axis": item.axis,
                    "boundary_value": item.value,
                    "context": dict(item.context),
                }
                for item in boundaries
            ],
        }
        if self.artifacts is not None:
            self.artifacts.save_summary(job_id, summary)
        if self.certified_profiles is not None and self.compatibility is not None:
            certified = certified_profile_from_summary(
                summary,
                self.compatibility,
                self.model_context_limit,
            )
            if certified is not None:
                stored = self.certified_profiles.save(certified)
                summary["certified_profile"] = certified.to_dict()
                summary["certified_profile_key"] = getattr(stored, "key", None)
        return summary

    def _classify_limits(self, outcome: TrialOutcome) -> TrialOutcome:
        if (
            outcome.status == "completed"
            and self.config.request_timeout_seconds is not None
            and outcome.runtime_seconds >= self.config.request_timeout_seconds
        ):
            return TrialOutcome(
                **{
                    **outcome.to_dict(),
                    "status": "timeout",
                    "error": "request_timeout_seconds exceeded",
                }
            )
        speed = outcome.metrics.get("tokens_per_second")
        if (
            outcome.status == "completed"
            and self.config.minimum_tokens_per_second is not None
            and outcome.runtime_seconds >= self.config.slow_window_seconds
            and isinstance(speed, (int, float))
            and speed < self.config.minimum_tokens_per_second
        ):
            return TrialOutcome(
                **{
                    **outcome.to_dict(),
                    "status": "slow",
                    "error": "minimum_tokens_per_second not sustained",
                }
            )
        return outcome

    def _boundary_axis(self, outcome: TrialOutcome) -> str | None:
        if outcome.status not in {"timeout", "slow"}:
            return None
        return next(
            (
                axis
                for axis in reversed(self.config.axis_order)
                if axis in self.config.monotonic_axes
            ),
            None,
        )

    def _make_boundary(
        self, configuration: Mapping[str, Any], axis: str
    ) -> _Boundary:
        candidates = self.config.axes[axis]
        return _Boundary(
            axis=axis,
            candidate_index=candidates.index(configuration[axis]),
            value=configuration[axis],
            context=tuple(
                sorted(
                    (name, repr(value))
                    for name, value in configuration.items()
                    if name != axis
                )
            ),
        )

    def _matching_boundary(
        self,
        configuration: Mapping[str, Any],
        boundaries: Iterable[_Boundary],
    ) -> _Boundary | None:
        for boundary in boundaries:
            context = tuple(
                sorted(
                    (name, repr(value))
                    for name, value in configuration.items()
                    if name != boundary.axis
                )
            )
            if context != boundary.context:
                continue
            candidate_index = self.config.axes[boundary.axis].index(
                configuration[boundary.axis]
            )
            if candidate_index >= boundary.candidate_index:
                return boundary
        return None


def certified_profile_from_summary(
    summary: Mapping[str, Any],
    compatibility: RuntimeCompatibility,
    model_context_limit: int | None,
) -> CertifiedInferenceProfile | None:
    """Create an operational profile from successful boundary evidence."""
    successful = [
        item
        for item in summary.get("outcomes", ())
        if item.get("status") == "completed"
        and isinstance(item.get("configuration", {}).get("context_length"), int)
    ]
    if not successful:
        return None
    maximum_context = max(
        item["configuration"]["context_length"] for item in successful
    )
    at_maximum = [
        item
        for item in successful
        if item["configuration"]["context_length"] == maximum_context
    ]
    batches = [
        item["configuration"].get("batch_size")
        for item in at_maximum
        if isinstance(item["configuration"].get("batch_size"), int)
    ]
    speeds = [
        item.get("metrics", {}).get("tokens_per_second")
        for item in at_maximum
        if isinstance(item.get("metrics", {}).get("tokens_per_second"), (int, float))
    ]
    first_tokens = [
        item.get("metrics", {}).get("time_to_first_token_seconds")
        for item in at_maximum
        if isinstance(
            item.get("metrics", {}).get("time_to_first_token_seconds"),
            (int, float),
        )
    ]
    memory = next(
        (
            item["configuration"].get("gpu_memory_utilization")
            for item in reversed(at_maximum)
            if isinstance(
                item["configuration"].get("gpu_memory_utilization"),
                (int, float),
            )
        ),
        None,
    )
    return CertifiedInferenceProfile(
        profile_id=new_profile_id(compatibility),
        created_at=datetime.now(timezone.utc).isoformat(),
        compatibility=compatibility,
        model_context_limit=model_context_limit,
        maximum_certified_context_length=maximum_context,
        maximum_certified_batch_size=max(batches, default=None),
        gpu_memory_utilization=memory,
        minimum_observed_tokens_per_second=min(speeds, default=None),
        maximum_time_to_first_token_seconds=max(first_tokens, default=None),
        evidence_job_id=summary.get("job_id"),
        evidence_summary_key=summary.get("summary_key"),
    )
