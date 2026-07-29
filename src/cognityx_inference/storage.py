"""Logical-key persistence for inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import uuid
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
class ManagerStateRepository:
    """Persist one secret-free local worker state through logical storage."""

    storage: StorageClientLike
    key: str = "inference/manager/server-state.json"

    def save(self, value: Any) -> Any:
        return self.storage.put_json(self.key, value)

    def load(self) -> dict[str, Any] | None:
        if not self.storage.exists(self.key):
            return None
        with self.storage.open(self.key) as source:
            return json.loads(source.read())


@dataclass(slots=True)
class ChatSessionRepository:
    """Append-only saved chat states within an already user-scoped client."""

    storage: StorageClientLike

    def save(self, chat_id: str, revision: int, value: Any) -> Any:
        return self.storage.put_json(
            f"inference/chats/{_segment(chat_id)}/{revision:08d}.json", value
        )

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def load_latest(self, chat_id: str) -> Any:
        prefix = f"inference/chats/{_segment(chat_id)}"
        try:
            objects = self.storage.list(prefix)
        except FileNotFoundError as exc:
            raise KeyError(f"Chat session not found: {chat_id}") from exc
        files = [item for item in objects if not getattr(item, "is_directory", False)]
        if not files:
            raise KeyError(f"Chat session has no saved revisions: {chat_id}")
        latest = max(files, key=lambda item: getattr(item, "key", ""))
        key = getattr(latest, "key", "")
        relative = key[key.find(prefix):] if prefix in key else key
        with self.storage.open(relative) as source:
            return json.loads(source.read())


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

    def list_profiles(
        self,
        *,
        model: str | None = None,
        backend: str | None = None,
        profile: str | None = None,
        kv_cache_precision: str | None = None,
    ) -> list[Any]:
        """List saved certified profiles without exposing storage paths to callers."""
        from cognityx_inference.capabilities import CertifiedInferenceProfile

        prefix = "evaluations/hardware-boundary/certified-profiles"
        profiles = []
        for stored in self._walk(prefix):
            if getattr(stored, "is_directory", False) or not getattr(stored, "key", "").endswith(".json"):
                continue
            key = str(stored.key)
            relative = key[key.find(prefix):] if prefix in key else key
            try:
                with self.storage.open(relative) as source:
                    candidate = CertifiedInferenceProfile.from_dict(json.loads(source.read()))
            except (OSError, ValueError, TypeError, KeyError):
                continue
            identity = candidate.compatibility
            if model is not None and identity.model != model:
                continue
            if backend is not None and identity.backend != backend:
                continue
            if profile is not None and identity.profile != profile:
                continue
            if kv_cache_precision is not None and identity.kv_cache_precision != kv_cache_precision:
                continue
            profiles.append(candidate)
        return sorted(profiles, key=lambda item: item.created_at, reverse=True)

    def get_profile(self, profile_id: str) -> Any | None:
        return next(
            (item for item in self.list_profiles() if item.profile_id == profile_id),
            None,
        )

    def _walk(self, prefix: str) -> tuple[Any, ...]:
        try:
            children = self.storage.list(prefix)
        except FileNotFoundError:
            return ()
        result = []
        for child in children:
            result.append(child)
            if getattr(child, "is_directory", False):
                key = str(getattr(child, "key", ""))
                relative = key[key.find(prefix):] if prefix in key else key
                result.extend(self._walk(relative))
        return tuple(result)

    def find_compatible_any_kv_cache(self, compatibility: Any) -> Any | None:
        """Find the best profile when callers did not pin a KV-cache dtype."""
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
            return None
        for stored in stored_objects:
            if getattr(stored, "is_directory", False):
                continue
            key = getattr(stored, "key", "")
            relative = key[key.find(prefix):] if prefix in key else key
            try:
                with self.storage.open(relative) as source:
                    candidate = CertifiedInferenceProfile.from_dict(json.loads(source.read()))
            except (OSError, ValueError, TypeError, KeyError):
                continue
            actual = candidate.compatibility
            if (
                actual.hardware_fingerprint == compatibility.hardware_fingerprint
                and actual.model == compatibility.model
                and actual.model_revision == compatibility.model_revision
                and actual.backend == compatibility.backend
                and actual.backend_version == compatibility.backend_version
                and actual.profile == compatibility.profile
                and actual.tensor_parallelism == compatibility.tensor_parallelism
            ):
                matches.append(candidate)
        return max(
            matches,
            key=lambda item: (
                item.maximum_certified_context_length,
                int(item.certified_configuration.get("generation_length", 0)),
                item.created_at,
            ),
            default=None,
        )
