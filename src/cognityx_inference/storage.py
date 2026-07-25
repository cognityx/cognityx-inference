"""Logical-key persistence for inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Protocol

from cognityx_inference.contracts import InferenceRequest, InferenceResponse


class StorageClientLike(Protocol):
    """Subset of ``cognityx_storage.StorageClient`` used by inference."""

    def put_json(self, key: str, value: Any) -> Any: ...

    def materialize(self, key: str) -> Any: ...

    def exists(self, key: str) -> bool: ...

    def open(self, key: str) -> Any: ...

    def list(self, prefix: str = "") -> tuple[Any, ...]: ...


def _segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "unknown"


@dataclass(slots=True)
class InferenceArtifactRepository:
    """Persist records without exposing a physical storage layout."""

    storage: StorageClientLike

    def save_exchange(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> Any:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"inference/requests/{date}/{_segment(response.request_id)}.json"
        return self.storage.put_json(
            key,
            {"request": request.to_dict(), "response": response.to_dict()},
        )

    def save_configuration(self, configuration_id: str, value: Any) -> Any:
        return self.storage.put_json(
            f"inference/configurations/{_segment(configuration_id)}.json", value
        )

    def materialize_model(self, logical_key: str) -> Any:
        """Ask storage to make a model available locally to a local runtime."""
        return self.storage.materialize(logical_key)


@dataclass(slots=True)
class BoundaryArtifactRepository:
    """Persist append-only boundary-evaluation manifests and trials."""

    storage: StorageClientLike

    def save_manifest(self, job_id: str, value: Any) -> Any:
        return self.storage.put_json(
            f"evaluations/hardware-boundary/{_segment(job_id)}/manifest.json",
            value,
        )

    def save_trial(self, job_id: str, trial_id: str, value: Any) -> Any:
        return self.storage.put_json(
            "evaluations/hardware-boundary/"
            f"{_segment(job_id)}/trials/{_segment(trial_id)}.json",
            value,
        )

    def save_summary(self, job_id: str, value: Any) -> Any:
        return self.storage.put_json(
            f"evaluations/hardware-boundary/{_segment(job_id)}/summary.json",
            value,
        )


@dataclass(slots=True)
class CertifiedProfileRepository:
    """Store and find append-only certified inference profiles."""

    storage: StorageClientLike

    @staticmethod
    def prefix(
        hardware_fingerprint: str,
        backend: str,
        model: str,
        profile: str,
    ) -> str:
        return (
            "evaluations/hardware-boundary/certified-profiles/"
            f"{_segment(hardware_fingerprint)}/{_segment(backend)}/"
            f"{_segment(model)}/{_segment(profile)}"
        )

    def save(self, profile: Any) -> Any:
        compatibility = profile.compatibility
        key = (
            self.prefix(
                compatibility.hardware_fingerprint,
                compatibility.backend,
                compatibility.model,
                compatibility.profile,
            )
            + f"/{_segment(profile.profile_id)}.json"
        )
        return self.storage.put_json(key, profile.to_dict())

    def find_compatible(self, compatibility: Any) -> Any | None:
        from cognityx_inference.capabilities import CertifiedInferenceProfile

        prefix = self.prefix(
            compatibility.hardware_fingerprint,
            compatibility.backend,
            compatibility.model,
            compatibility.profile,
        )
        matches = []
        try:
            stored_objects = self.storage.list(prefix)
        except FileNotFoundError:
            # A first-use model/profile has no certification directory yet.
            return None
        for stored in stored_objects:
            if getattr(stored, "is_directory", False):
                continue
            key = getattr(stored, "key", "")
            marker = key.find(prefix)
            relative = key[marker:] if marker >= 0 else key
            try:
                with self.storage.open(relative) as source:
                    value = json.loads(source.read())
                candidate = CertifiedInferenceProfile.from_dict(value)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if candidate.compatibility == compatibility:
                matches.append(candidate)
        return max(matches, key=lambda item: item.created_at, default=None)
