"""Optional tracker-neutral indexing for authoritative Storage publications."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from cognityx_observability import (
    ArtifactReference,
    MLflowExporter,
    ObservationContext,
    ObservationSession,
)

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
    """Compatibility tracker delegating MLflow mechanics to Observability."""

    def __init__(
        self,
        configuration: TrackingConfiguration,
        *,
        mlflow_module: Any | None = None,
    ) -> None:
        self.configuration = configuration
        self.mlflow_module = mlflow_module

    def record(
        self,
        *,
        name: str,
        tags: Mapping[str, Any],
        metrics: Mapping[str, float | int],
        references: Mapping[str, str],
        parent_run_id: str | None = None,
    ) -> None:
        selected_parent = parent_run_id or self.configuration.parent_run_id
        run_id = tags.get("cognityx.inference_run_id") or tags.get(
            "cognityx.inference_pair_id"
        )
        idempotency_key = next(
            (
                str(references[key])
                for key in ("pair_manifest_uri", "run_manifest_uri")
                if references.get(key)
            ),
            None,
        )
        session = ObservationSession(
            ObservationContext(
                component=str(tags.get("cognityx.component") or "inference"),
                operation="publish_research_result",
                run_id=str(run_id) if run_id else None,
                parent_run_id=selected_parent,
                idempotency_key=idempotency_key,
                attributes={
                    str(key): value for key, value in tags.items() if value is not None
                },
            ),
            MLflowExporter(
                tracking_uri=self.configuration.tracking_uri,
                experiment_name=self.configuration.experiment_name,
                run_name=name,
                mlflow_module=self.mlflow_module,
            ),
            failure_policy="error",
        )
        session.start()
        session.metrics(metrics)
        session.artifacts(
            ArtifactReference(
                name=str(key),
                uri=str(value),
                checksum=(
                    str(references.get(str(key).removesuffix("_uri") + "_checksum"))
                    if references.get(str(key).removesuffix("_uri") + "_checksum")
                    else None
                ),
            )
            for key, value in references.items()
            if key.endswith("_uri") and value
        )
        session.finish(attributes={"storage_publication_authoritative": True})


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
