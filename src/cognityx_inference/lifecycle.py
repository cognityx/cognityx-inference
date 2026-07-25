"""Concurrency-safe lifecycle control for resident local models."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
import threading
import time
from types import TracebackType
from typing import Any, Callable, Mapping

from cognityx_inference.backends.base import InferenceBackend


class ModelState(StrEnum):
    """Resident model lifecycle states."""

    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    UNLOADING = "unloading"
    FAILED = "failed"


def _freeze(values: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in values.items()))


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Settings that determine whether loaded model weights can be reused."""

    model: str
    backend: str
    runtime: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls, model: str, backend: str, runtime: Mapping[str, Any] | None = None
    ) -> "ModelIdentity":
        return cls(model=model, backend=backend, runtime=_freeze(runtime or {}))


@dataclass(slots=True)
class _Resident:
    identity: ModelIdentity
    backend: InferenceBackend
    state: ModelState = ModelState.LOADING
    active_leases: int = 0
    loaded_at: float | None = None
    load_seconds: float | None = None
    last_used_at: float | None = None
    error: str | None = None
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )


@dataclass(frozen=True, slots=True)
class ResidentModelStatus:
    """Public status for one resident model."""

    identity: ModelIdentity
    state: ModelState
    active_leases: int
    loaded_at: float | None
    load_seconds: float | None
    last_used_at: float | None
    error: str | None


class ModelLease(AbstractContextManager[InferenceBackend]):
    """Keep a resident model protected from unload while in use."""

    def __init__(self, manager: "ModelManager", resident: _Resident) -> None:
        self._manager = manager
        self._resident = resident
        self._released = False

    def __enter__(self) -> InferenceBackend:
        return self._resident.backend

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release(self._resident)


class ModelManager:
    """Load, lease, inspect, drain, and unload local models safely."""

    def __init__(
        self,
        factories: Mapping[
            str, Callable[[str, dict[str, Any]], InferenceBackend]
        ],
    ) -> None:
        self._factories = dict(factories)
        self._models: dict[ModelIdentity, _Resident] = {}
        self._active: dict[str, ModelIdentity] = {}
        self._lock = threading.RLock()

    def load(
        self,
        model: str,
        backend: str,
        runtime: Mapping[str, Any] | None = None,
    ) -> ResidentModelStatus:
        identity = ModelIdentity.create(model, backend, runtime)
        waiting: _Resident | None = None
        with self._lock:
            existing = self._models.get(identity)
            if existing is not None:
                if existing.state is ModelState.LOADING:
                    waiting = existing
                else:
                    if existing.state is ModelState.FAILED:
                        raise RuntimeError(existing.error or "model load failed")
                    return self._status(existing)
            else:
                try:
                    factory = self._factories[backend]
                except KeyError as exc:
                    raise ValueError(f"Unknown local backend: {backend}") from exc
                resident = _Resident(identity, factory(model, dict(runtime or {})))
                self._models[identity] = resident
        if waiting is not None:
            with waiting.condition:
                while waiting.state is ModelState.LOADING:
                    waiting.condition.wait()
            with self._lock:
                if waiting.state is ModelState.FAILED:
                    raise RuntimeError(waiting.error or "model load failed")
                return self._status(waiting)
        started = time.monotonic()
        try:
            resident.backend.load()
        except Exception as exc:
            with self._lock:
                resident.state = ModelState.FAILED
                resident.error = str(exc)
            with resident.condition:
                resident.condition.notify_all()
            raise
        with self._lock:
            resident.load_seconds = time.monotonic() - started
            resident.loaded_at = time.time()
            resident.state = ModelState.READY
            self._active[model] = identity
            status = self._status(resident)
        with resident.condition:
            resident.condition.notify_all()
        return status

    def acquire(
        self,
        model: str,
        backend: str,
        runtime: Mapping[str, Any] | None = None,
    ) -> ModelLease:
        identity = ModelIdentity.create(model, backend, runtime)
        self.load(model, backend, runtime)
        with self._lock:
            resident = self._models[identity]
            if resident.state is ModelState.DRAINING:
                raise RuntimeError("model is draining and cannot accept new work")
            if resident.state not in {ModelState.READY, ModelState.BUSY}:
                raise RuntimeError(f"model is not ready: {resident.state}")
            resident.active_leases += 1
            resident.state = ModelState.BUSY
            resident.last_used_at = time.time()
            return ModelLease(self, resident)

    def acquire_loaded(self, model: str) -> ModelLease:
        """Acquire the active resident configuration without loading."""
        with self._lock:
            identity = self._active.get(model)
            if identity is None:
                raise LookupError(f"Model is not loaded: {model}")
            resident = self._models.get(identity)
            if resident is None:
                self._active.pop(model, None)
                raise LookupError(f"Model is not loaded: {model}")
            if resident.state is ModelState.DRAINING:
                raise RuntimeError("model is draining and cannot accept new work")
            if resident.state not in {ModelState.READY, ModelState.BUSY}:
                raise RuntimeError(f"model is not ready: {resident.state}")
            resident.active_leases += 1
            resident.state = ModelState.BUSY
            resident.last_used_at = time.time()
            return ModelLease(self, resident)

    def _release(self, resident: _Resident) -> None:
        unload = False
        with self._lock:
            resident.active_leases -= 1
            resident.last_used_at = time.time()
            if resident.active_leases == 0:
                if resident.state is ModelState.DRAINING:
                    resident.state = ModelState.UNLOADING
                    unload = True
                else:
                    resident.state = ModelState.READY
        if unload:
            self._unload_resident(resident)

    def unload(
        self,
        identity: ModelIdentity,
        *,
        wait: bool = False,
        timeout_seconds: float | None = None,
    ) -> bool:
        with self._lock:
            resident = self._models.get(identity)
            if resident is None:
                return False
            if resident.active_leases:
                resident.state = ModelState.DRAINING
                if not wait:
                    return False
            else:
                resident.state = ModelState.UNLOADING
        if resident.active_leases and wait:
            deadline = (
                None if timeout_seconds is None else time.monotonic() + timeout_seconds
            )
            while True:
                with self._lock:
                    if resident.active_leases == 0:
                        if identity not in self._models:
                            return True
                        resident.state = ModelState.UNLOADING
                        break
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)
        self._unload_resident(resident)
        return True

    def unload_all(self, *, wait: bool = False) -> dict[ModelIdentity, bool]:
        with self._lock:
            identities = tuple(self._models)
        return {identity: self.unload(identity, wait=wait) for identity in identities}

    def unload_active(self, model: str, *, wait: bool = False) -> bool:
        with self._lock:
            identity = self._active.get(model)
        return self.unload(identity, wait=wait) if identity is not None else False

    def statuses(self) -> tuple[ResidentModelStatus, ...]:
        with self._lock:
            return tuple(self._status(item) for item in self._models.values())

    def _unload_resident(self, resident: _Resident) -> None:
        resident.backend.unload()
        with self._lock:
            self._models.pop(resident.identity, None)
            if self._active.get(resident.identity.model) == resident.identity:
                self._active.pop(resident.identity.model, None)

    @staticmethod
    def _status(resident: _Resident) -> ResidentModelStatus:
        return ResidentModelStatus(
            identity=resident.identity,
            state=resident.state,
            active_leases=resident.active_leases,
            loaded_at=resident.loaded_at,
            load_seconds=resident.load_seconds,
            last_used_at=resident.last_used_at,
            error=resident.error,
        )
