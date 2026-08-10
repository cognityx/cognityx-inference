"""Optional tracker-neutral indexing for authoritative Storage publications."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from cognityx_inference.errors import ResearchRunError


@dataclass(frozen=True, slots=True)
class TrackingConfiguration:
    """Live observability settings; Storage remains the source of truth."""

    backend: str = "none"
    tracking_uri: str | None = None
    experiment_name: str = "cognityx-research"
    parent_run_id: str | None = None
    failure_policy: str = "warn"

    def __post_init__(self) -> None:
        if self.backend not in {"none", "mlflow"}:
            raise ValueError("tracking.backend must be 'none' or 'mlflow'")
        if self.failure_policy not in {"warn", "error"}:
            raise ValueError("tracking.failure_policy must be 'warn' or 'error'")


class PublicationTracker(Protocol):
    def record(
        self,
        *,
        name: str,
        tags: Mapping[str, Any],
        metrics: Mapping[str, float | int],
        references: Mapping[str, str],
        parent_run_id: str | None = None,
    ) -> None: ...


class NoOpTracker:
    """Default tracker that performs no external writes."""

    def record(
        self,
        *,
        name: str,
        tags: Mapping[str, Any],
        metrics: Mapping[str, float | int],
        references: Mapping[str, str],
        parent_run_id: str | None = None,
    ) -> None:
        del name, tags, metrics, references, parent_run_id


class MLflowTracker:
    """Index Storage references in MLflow without duplicating artifacts."""

    def __init__(self, configuration: TrackingConfiguration) -> None:
        self.configuration = configuration

    def record(
        self,
        *,
        name: str,
        tags: Mapping[str, Any],
        metrics: Mapping[str, float | int],
        references: Mapping[str, str],
        parent_run_id: str | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking requires cognityx-inference[tracking]"
            ) from exc
        if self.configuration.tracking_uri:
            mlflow.set_tracking_uri(self.configuration.tracking_uri)
        mlflow.set_experiment(self.configuration.experiment_name)
        selected_parent = parent_run_id or self.configuration.parent_run_id
        normalized_tags = {
            str(key): str(value) for key, value in tags.items() if value is not None
        }
        if selected_parent:
            normalized_tags["mlflow.parentRunId"] = selected_parent
        # Storage URIs and checksums are searchable tags. Prediction files and
        # adapters are intentionally not uploaded as duplicate MLflow artifacts.
        normalized_tags.update(
            {
                f"cognityx.storage.{key}": str(value)
                for key, value in references.items()
                if value
            }
        )
        with mlflow.start_run(run_name=name, tags=normalized_tags):
            if metrics:
                mlflow.log_metrics(
                    {str(key): float(value) for key, value in metrics.items()}
                )


class SafeTracker:
    """Apply warn/error policy after the Storage publication has succeeded."""

    def __init__(
        self,
        tracker: PublicationTracker,
        *,
        failure_policy: str = "warn",
    ) -> None:
        self.tracker = tracker
        self.failure_policy = failure_policy

    def record(self, **payload: Any) -> dict[str, str] | None:
        try:
            self.tracker.record(**payload)
        except Exception as exc:
            if self.failure_policy == "error":
                raise ResearchRunError(
                    "tracking_failed",
                    "Storage publication succeeded but tracking failed",
                    details={"reason": str(exc)},
                ) from exc
            warnings.warn(f"Inference tracking failed: {exc}", stacklevel=2)
            return {"status": "failed", "message": str(exc)}
        return None


def build_tracker(configuration: TrackingConfiguration) -> SafeTracker:
    tracker: PublicationTracker
    if configuration.backend == "mlflow":
        tracker = MLflowTracker(configuration)
    else:
        tracker = NoOpTracker()
    return SafeTracker(tracker, failure_policy=configuration.failure_policy)
