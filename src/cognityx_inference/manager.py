"""Lightweight lifecycle manager for one local inference worker process."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib import request as urllib_request
from uuid import uuid4

from cognityx_inference.configuration import (
    LocalServerProfile,
    ManagerConfiguration,
)


class ServerState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ServerStatus:
    server_id: str | None
    profile: str | None
    model: str | None
    backend: str | None
    state: ServerState
    job_id: str | None = None
    pid: int | None = None
    process_group_id: int | None = None
    worker_url: str | None = None
    started_at: float | None = None
    updated_at: float | None = None
    error_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ServerStatus":
        data = dict(value)
        data["state"] = ServerState(data["state"])
        return cls(**data)


class StateRepository(Protocol):
    def save(self, value: Any) -> Any: ...

    def load(self) -> dict[str, Any] | None: ...


class JobRepositoryLike(Protocol):
    def create(
        self,
        job_id: str,
        job_type: str,
        payload: dict[str, Any],
        *,
        owner_id: str = "local",
    ) -> Any: ...

    def set_state(self, job_id: str, state: str) -> None: ...

    def append_event(
        self, job_id: str, event: str, data: dict[str, Any]
    ) -> int: ...

    def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]: ...


class WorkerProcessController:
    """Spawn and stop a worker without exposing its command or environment."""

    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    def start(self, profile: LocalServerProfile) -> tuple[int, int]:
        command = [
            sys.executable,
            "-m",
            "cognityx_inference.cli",
            "serve",
            "--host",
            profile.host,
            "--port",
            str(profile.port),
        ]
        environment = dict(os.environ)
        for secret_name in ("OPENAI_API_KEY", "GROQ_API_KEY", "XAI_API_KEY"):
            environment.pop(secret_name, None)
        options: dict[str, Any] = {
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(command, **options)
        self._processes[process.pid] = process
        group_id = process.pid if os.name == "nt" else os.getpgid(process.pid)
        return process.pid, group_id

    def is_alive(self, pid: int) -> bool:
        process = self._processes.get(pid)
        if process is not None:
            return process.poll() is None
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def stop(
        self,
        pid: int,
        process_group_id: int | None,
        *,
        timeout_seconds: float,
    ) -> bool:
        if not self.is_alive(pid):
            self._processes.pop(pid, None)
            return False
        process = self._processes.get(pid)
        if os.name == "nt":
            graceful_signal = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)
            if process is not None:
                process.send_signal(graceful_signal)
            else:
                os.kill(pid, graceful_signal)
        else:
            os.killpg(process_group_id or pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_seconds
        while self.is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        forced = self.is_alive(pid)
        if forced:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.killpg(process_group_id or pid, signal.SIGKILL)
        if process is not None:
            try:
                process.wait(timeout=max(1.0, timeout_seconds))
            except subprocess.TimeoutExpired:
                pass
        self._processes.pop(pid, None)
        return forced


class WorkerHTTPClient:
    """Health and load calls used by the manager during startup."""

    def ready(self, worker_url: str, *, timeout_seconds: float = 2) -> bool:
        request = urllib_request.Request(f"{worker_url}/v1/models", method="GET")
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds):
                return True
        except (HTTPError, URLError, OSError, TimeoutError):
            return False

    def load(
        self, worker_url: str, profile: LocalServerProfile, *, timeout_seconds: float
    ) -> dict[str, Any]:
        payload = {
            "model": profile.model,
            "backend": profile.backend,
            "profile": profile.load_profile,
            "runtime": {
                **dict(profile.runtime),
                **(
                    {"certified_profile_id": profile.certified_profile_id}
                    if profile.certified_profile_id
                    else {}
                ),
            },
            "discovery_policy": "require_existing",
        }
        request = urllib_request.Request(
            f"{worker_url}/v1/cognityx/models/load",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            raise RuntimeError(f"worker_load_http_{exc.code}") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise RuntimeError("worker_load_connection_error") from exc


class InferenceManager:
    """Manage exactly one local worker with durable, replayable state."""

    JOB_TYPE = "inference_server_start"

    def __init__(
        self,
        profiles: Mapping[str, LocalServerProfile],
        configuration: ManagerConfiguration,
        *,
        jobs: JobRepositoryLike,
        state_repository: StateRepository,
        process_controller: WorkerProcessController | None = None,
        worker_client: WorkerHTTPClient | None = None,
    ) -> None:
        self.profiles = dict(profiles)
        self.configuration = configuration
        self.jobs = jobs
        self.state_repository = state_repository
        self.processes = process_controller or WorkerProcessController()
        self.worker = worker_client or WorkerHTTPClient()
        self._lock = threading.RLock()
        self._status = self._recover()

    def start(self, profile_name: str, *, owner_id: str = "local") -> ServerStatus:
        try:
            profile = self.profiles[profile_name]
        except KeyError as exc:
            raise KeyError(f"Unknown local server profile: {profile_name}") from exc
        with self._lock:
            self._reconcile()
            if self._status.state in {
                ServerState.STARTING,
                ServerState.READY,
                ServerState.DEGRADED,
            }:
                if self._status.profile != profile_name:
                    raise RuntimeError(
                        "Another local server profile is already active."
                    )
                return self._status
            if self._status.state is ServerState.STOPPING:
                raise RuntimeError("The local server is still stopping.")
            server_id = f"server-{uuid4().hex}"
            job_id = f"job-{uuid4().hex}"
            now = time.time()
            self.jobs.create(
                job_id,
                self.JOB_TYPE,
                {
                    "server_id": server_id,
                    "profile": profile_name,
                    "model": profile.model,
                    "backend": profile.backend,
                },
                owner_id=owner_id,
            )
            self._status = ServerStatus(
                server_id=server_id,
                profile=profile_name,
                model=profile.model,
                backend=profile.backend,
                state=ServerState.STARTING,
                job_id=job_id,
                worker_url=f"http://{profile.host}:{profile.port}",
                started_at=now,
                updated_at=now,
            )
            self._persist()
            self._event("server_starting")
            threading.Thread(
                target=self._start_worker,
                args=(profile, server_id),
                name=f"cognityx-worker-start-{server_id}",
                daemon=True,
            ).start()
            return self._status

    def stop(self) -> ServerStatus:
        with self._lock:
            self._reconcile()
            if self._status.state is ServerState.STOPPED:
                return self._status
            if self._status.state is ServerState.STOPPING:
                return self._status
            self._transition(ServerState.STOPPING, "server_stopping")
            pid = self._status.pid
            group = self._status.process_group_id
        forced = False
        if pid is not None:
            forced = self.processes.stop(
                pid,
                group,
                timeout_seconds=self.configuration.stop_timeout_seconds,
            )
        with self._lock:
            self._transition(
                ServerState.STOPPED,
                "server_stopped",
                forced=forced,
            )
            if self._status.job_id:
                self.jobs.set_state(self._status.job_id, "completed")
            self._status = replace(
                self._status,
                pid=None,
                process_group_id=None,
                error_category=None,
                updated_at=time.time(),
            )
            self._persist()
            return self._status

    def status(self) -> ServerStatus:
        with self._lock:
            self._reconcile()
            return self._status

    def events(self, *, after: int = 0) -> list[dict[str, Any]]:
        status = self.status()
        if not status.job_id:
            return [
                {
                    "sequence": 0,
                    "event": "server_status",
                    **status.to_dict(),
                }
            ]
        return self.jobs.events(status.job_id, after=after)

    def _start_worker(
        self, profile: LocalServerProfile, server_id: str
    ) -> None:
        try:
            pid, group = self.processes.start(profile)
            with self._lock:
                if (
                    self._status.server_id != server_id
                    or self._status.state is not ServerState.STARTING
                ):
                    self.processes.stop(
                        pid,
                        group,
                        timeout_seconds=self.configuration.stop_timeout_seconds,
                    )
                    return
                self._status = replace(
                    self._status,
                    pid=pid,
                    process_group_id=group,
                    updated_at=time.time(),
                )
                self._persist()
                self._event("worker_process_started")
            deadline = (
                time.monotonic() + self.configuration.startup_timeout_seconds
            )
            while time.monotonic() < deadline:
                if not self.processes.is_alive(pid):
                    raise RuntimeError("worker_process_exited")
                if self.worker.ready(self._status.worker_url or ""):
                    break
                time.sleep(self.configuration.health_poll_seconds)
            else:
                raise RuntimeError("worker_startup_timeout")
            with self._lock:
                if (
                    self._status.server_id != server_id
                    or self._status.state is not ServerState.STARTING
                ):
                    return
                self._event("worker_api_ready")
            self.worker.load(
                self._status.worker_url or "",
                profile,
                timeout_seconds=max(
                    1.0, deadline - time.monotonic()
                ),
            )
            with self._lock:
                if (
                    self._status.server_id != server_id
                    or self._status.state is not ServerState.STARTING
                ):
                    return
                self._transition(ServerState.READY, "server_ready")
                if self._status.job_id:
                    self.jobs.set_state(self._status.job_id, "completed")
        except Exception as exc:
            with self._lock:
                if self._status.server_id != server_id:
                    return
                pid = self._status.pid
                group = self._status.process_group_id
            if pid is not None:
                self.processes.stop(
                    pid,
                    group,
                    timeout_seconds=self.configuration.stop_timeout_seconds,
                )
            with self._lock:
                if self._status.server_id != server_id:
                    return
                self._status = replace(
                    self._status,
                    state=ServerState.FAILED,
                    pid=None,
                    process_group_id=None,
                    error_category=_safe_error_category(exc),
                    updated_at=time.time(),
                )
                self._persist()
                self._event("server_failed")
                if self._status.job_id:
                    self.jobs.set_state(self._status.job_id, "failed")

    def _recover(self) -> ServerStatus:
        saved = self.state_repository.load()
        if not saved:
            return ServerStatus(
                server_id=None,
                profile=None,
                model=None,
                backend=None,
                state=ServerState.STOPPED,
                updated_at=time.time(),
            )
        status = ServerStatus.from_dict(saved)
        if status.pid is None or not self.processes.is_alive(status.pid):
            recovered = replace(
                status,
                state=ServerState.STOPPED,
                pid=None,
                process_group_id=None,
                error_category="stale_process_state",
                updated_at=time.time(),
            )
            self.state_repository.save(recovered.to_dict())
            if status.job_id and status.state in {
                ServerState.STARTING,
                ServerState.READY,
                ServerState.DEGRADED,
            }:
                self.jobs.set_state(status.job_id, "interrupted")
            return recovered
        state = (
            ServerState.READY
            if status.worker_url and self.worker.ready(status.worker_url)
            else ServerState.DEGRADED
        )
        recovered = replace(status, state=state, updated_at=time.time())
        self.state_repository.save(recovered.to_dict())
        return recovered

    def _reconcile(self) -> None:
        if (
            self._status.pid is not None
            and self._status.state
            not in {ServerState.STOPPED, ServerState.FAILED}
            and not self.processes.is_alive(self._status.pid)
        ):
            self._status = replace(
                self._status,
                state=ServerState.FAILED,
                pid=None,
                process_group_id=None,
                error_category="worker_process_exited",
                updated_at=time.time(),
            )
            self._persist()
            self._event("server_failed")
            if self._status.job_id:
                self.jobs.set_state(self._status.job_id, "failed")
            return
        if (
            self._status.pid is not None
            and self._status.worker_url
            and self._status.state
            in {ServerState.READY, ServerState.DEGRADED}
        ):
            healthy = self.worker.ready(self._status.worker_url)
            if healthy and self._status.state is ServerState.DEGRADED:
                self._transition(ServerState.READY, "server_recovered")
            elif not healthy and self._status.state is ServerState.READY:
                self._transition(ServerState.DEGRADED, "server_degraded")

    def _transition(
        self, state: ServerState, event: str, **extra: Any
    ) -> None:
        self._status = replace(
            self._status,
            state=state,
            updated_at=time.time(),
            error_category=None if state is not ServerState.FAILED else self._status.error_category,
        )
        self._persist()
        self._event(event, **extra)

    def _event(self, event: str, **extra: Any) -> None:
        if not self._status.job_id:
            return
        self.jobs.append_event(
            self._status.job_id,
            event,
            {
                "server_id": self._status.server_id,
                "profile": self._status.profile,
                "model": self._status.model,
                "backend": self._status.backend,
                "state": self._status.state.value,
                **extra,
            },
        )

    def _persist(self) -> None:
        self.state_repository.save(self._status.to_dict())


def _safe_error_category(exc: Exception) -> str:
    value = str(exc)
    if value in {
        "worker_process_exited",
        "worker_startup_timeout",
        "worker_load_connection_error",
    } or value.startswith("worker_load_http_"):
        return value
    return type(exc).__name__
