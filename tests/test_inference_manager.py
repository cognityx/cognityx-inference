from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

from cognityx_inference.configuration import (
    LocalServerProfile,
    ManagerConfiguration,
)
from cognityx_inference.manager import (
    InferenceManager,
    ServerState,
    ServerStatus,
)


class MemoryState:
    def __init__(self, value: dict[str, Any] | None = None) -> None:
        self.value = value

    def save(self, value: Any) -> None:
        self.value = dict(value)

    def load(self) -> dict[str, Any] | None:
        return self.value


class MemoryJobs:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.states: dict[str, str] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}

    def create(
        self,
        job_id: str,
        job_type: str,
        payload: dict[str, Any],
        *,
        owner_id: str = "local",
    ) -> None:
        self.created.append(job_id)
        self.states[job_id] = "queued"
        self.records[job_id] = []

    def set_state(self, job_id: str, state: str) -> None:
        self.states[job_id] = state

    def append_event(
        self, job_id: str, event: str, data: dict[str, Any]
    ) -> int:
        sequence = len(self.records[job_id]) + 1
        self.records[job_id].append(
            {"sequence": sequence, "event": event, **data}
        )
        return sequence

    def events(
        self, job_id: str, after: int = 0
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in self.records[job_id]
            if item["sequence"] > after
        ]


class FakeProcesses:
    def __init__(self, *, alive: bool = True) -> None:
        self.alive = alive
        self.starts = 0
        self.stops = 0

    def start(self, profile: LocalServerProfile) -> tuple[int, int]:
        self.starts += 1
        self.alive = True
        return 1234, 1234

    def is_alive(self, pid: int) -> bool:
        return self.alive and pid == 1234

    def stop(
        self, pid: int, group: int | None, *, timeout_seconds: float
    ) -> bool:
        self.stops += 1
        self.alive = False
        return False


class FakeWorker:
    def __init__(self, *, ready: bool = True, fail_load: bool = False) -> None:
        self.is_ready = ready
        self.fail_load = fail_load
        self.loads = 0

    def ready(self, worker_url: str, *, timeout_seconds: float = 2) -> bool:
        return self.is_ready

    def load(
        self,
        worker_url: str,
        profile: LocalServerProfile,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.loads += 1
        if self.fail_load:
            raise RuntimeError("worker_load_http_400")
        return {"state": "ready"}


def _manager(
    *,
    state: MemoryState | None = None,
    processes: FakeProcesses | None = None,
    worker: FakeWorker | None = None,
) -> tuple[InferenceManager, MemoryJobs, FakeProcesses]:
    jobs = MemoryJobs()
    controller = processes or FakeProcesses()
    manager = InferenceManager(
        {
            "test": LocalServerProfile(
                name="test",
                model="org/model",
                backend="vllm",
                port=8129,
            )
        },
        ManagerConfiguration(
            startup_timeout_seconds=0.03,
            stop_timeout_seconds=0.01,
            health_poll_seconds=0.001,
        ),
        jobs=jobs,
        state_repository=state or MemoryState(),
        process_controller=controller,
        worker_client=worker or FakeWorker(),
    )
    return manager, jobs, controller


def _wait(manager: InferenceManager, state: ServerState) -> None:
    deadline = time.monotonic() + 1
    while manager.status().state is not state:
        assert time.monotonic() < deadline
        time.sleep(0.002)


def test_start_stop_are_idempotent_and_events_are_replayable() -> None:
    manager, jobs, processes = _manager()

    first = manager.start("test")
    second = manager.start("test")
    assert first.server_id == second.server_id
    _wait(manager, ServerState.READY)
    assert len(jobs.created) == 1
    assert processes.starts == 1
    assert [event["event"] for event in manager.events()] == [
        "server_starting",
        "worker_process_started",
        "worker_api_ready",
        "server_ready",
    ]

    stopped = manager.stop()
    again = manager.stop()
    assert stopped.state is ServerState.STOPPED
    assert again.state is ServerState.STOPPED
    assert processes.stops == 1


def test_concurrent_starts_share_one_startup_job() -> None:
    manager, jobs, processes = _manager()
    statuses: list[ServerStatus] = []
    threads = [
        threading.Thread(target=lambda: statuses.append(manager.start("test")))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _wait(manager, ServerState.READY)

    assert len({status.server_id for status in statuses}) == 1
    assert len(jobs.created) == 1
    assert processes.starts == 1


def test_stale_pid_is_recovered_as_stopped() -> None:
    saved = ServerStatus(
        server_id="server-old",
        profile="test",
        model="org/model",
        backend="vllm",
        state=ServerState.READY,
        job_id="job-old",
        pid=999,
        process_group_id=999,
        worker_url="http://127.0.0.1:8129",
    )
    state = MemoryState(saved.to_dict())
    manager, jobs, _ = _manager(
        state=state, processes=FakeProcesses(alive=False)
    )
    jobs.records["job-old"] = []
    jobs.states["job-old"] = "running"

    status = manager.status()
    assert status.state is ServerState.STOPPED
    assert status.pid is None
    assert status.error_category == "stale_process_state"


def test_startup_failure_is_sanitized_and_terminal() -> None:
    manager, jobs, processes = _manager(worker=FakeWorker(fail_load=True))
    status = manager.start("test")
    _wait(manager, ServerState.FAILED)

    failed = manager.status()
    assert failed.server_id == status.server_id
    assert failed.error_category == "worker_load_http_400"
    assert processes.stops == 1
    assert jobs.states[status.job_id] == "failed"
    assert manager.events()[-1]["event"] == "server_failed"


def test_ready_worker_degrades_and_recovers_from_health_state() -> None:
    worker = FakeWorker()
    manager, _, _ = _manager(worker=worker)
    manager.start("test")
    _wait(manager, ServerState.READY)

    worker.is_ready = False
    assert manager.status().state is ServerState.DEGRADED
    worker.is_ready = True
    assert manager.status().state is ServerState.READY
    assert [event["event"] for event in manager.events()][-2:] == [
        "server_degraded",
        "server_recovered",
    ]
