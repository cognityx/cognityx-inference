from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from typing import Any

import pytest

from cognityx_inference.client import CognityxInferenceClient
from cognityx_inference.configuration import (
    LocalServerProfile,
    ManagerConfiguration,
)
from cognityx_inference.contracts import InferenceRequest
from cognityx_inference.manager import InferenceManager
from cognityx_inference.manager_api import create_manager_app
from test_inference_manager import MemoryJobs, MemoryState


def _free_port() -> int:
    with socket.socket() as selected:
        selected.bind(("127.0.0.1", 0))
        return int(selected.getsockname()[1])


class MockWorkerProcess:
    def __init__(self, port: int) -> None:
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.starts = 0
        self.stops = 0

    def start(self, profile: LocalServerProfile) -> tuple[int, int]:
        self.starts += 1

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._json({"object": "list", "data": []})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/v1/cognityx/models/load":
                    self._json(
                        {
                            "state": "ready",
                            "requested_model": payload["model"],
                        }
                    )
                    return
                if self.path == "/v1/chat/completions":
                    self._json(
                        {
                            "id": "mock-completion",
                            "object": "chat.completion",
                            "model": payload["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "OK",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 1,
                                "total_tokens": 4,
                            },
                            "cognityx": {
                                "token_budget": {
                                    "input_tokens": 3,
                                    "max_context_tokens": 64,
                                    "reserved_tokens": 4,
                                    "requested_max_output_tokens": None,
                                    "effective_max_output_tokens": 16,
                                    "source": "automatic",
                                }
                            },
                        }
                    )
                    return
                self.send_error(404)

            def _json(self, value: dict[str, Any]) -> None:
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        return 4321, 4321

    def is_alive(self, pid: int) -> bool:
        return self.server is not None

    def stop(
        self,
        pid: int,
        process_group_id: int | None,
        *,
        timeout_seconds: float,
    ) -> bool:
        self.stops += 1
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.server = None
        return False


def test_auto_start_watch_infer_budget_and_stop_end_to_end() -> None:
    uvicorn = pytest.importorskip("uvicorn")
    manager_port = _free_port()
    worker_port = _free_port()
    process = MockWorkerProcess(worker_port)
    jobs = MemoryJobs()
    manager = InferenceManager(
        {
            "mock": LocalServerProfile(
                name="mock",
                model="mock/model",
                host="127.0.0.1",
                port=worker_port,
            )
        },
        ManagerConfiguration(
            startup_timeout_seconds=2,
            stop_timeout_seconds=1,
            health_poll_seconds=0.01,
        ),
        jobs=jobs,
        state_repository=MemoryState(),
        process_controller=process,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_manager_app(manager),
            host="127.0.0.1",
            port=manager_port,
            log_level="critical",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while not server.started:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    events: list[dict[str, Any]] = []
    client = CognityxInferenceClient(
        backend="local",
        profile="mock",
        auto_start=True,
        manager_url=f"http://127.0.0.1:{manager_port}",
        startup_timeout_seconds=3,
        discovery_poll_seconds=0.01,
        on_server_event=events.append,
    )
    try:
        assert client.server_status()["state"] == "STOPPED"
        responses: list[dict[str, Any]] = []
        callers = [
            threading.Thread(
                target=lambda: responses.append(
                    client.infer(
                        InferenceRequest(
                            model="mock/model", prompt="Reply OK"
                        )
                    )
                )
            )
            for _ in range(2)
        ]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=4)
        assert len(responses) == 2
        response = responses[0]
        assert response["choices"][0]["message"]["content"] == "OK"
        assert (
            response["cognityx"]["token_budget"][
                "effective_max_output_tokens"
            ]
            == 16
        )
        assert client.server_status()["state"] == "READY"
        assert process.starts == 1
        assert any(event["event"] == "server_ready" for event in events)
        assert client.server_stop()["state"] == "STOPPED"
        assert process.stops == 1
    finally:
        manager.stop()
        server.should_exit = True
        thread.join(timeout=3)
