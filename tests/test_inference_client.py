from __future__ import annotations

import json
from urllib.error import URLError

from cognityx_inference.client import (
    CognityxInferenceClient,
    InferenceAPIError,
)
from cognityx_inference.contracts import DiscoveryPolicy, InferenceRequest


class FakeClient(CognityxInferenceClient):
    def __init__(self):
        super().__init__(
            "http://example",
            discovery_policy=DiscoveryPolicy.ASK,
            discovery_poll_seconds=0.001,
            input_fn=lambda prompt: "yes",
        )
        self.calls = []
        self.posts = 0

    def _request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            return {"state": "completed"}
        self.posts += 1
        if self.posts == 1:
            raise InferenceAPIError(
                428,
                {
                    "detail": {
                        "error": "hardware_discovery_required",
                        "discovery_request_id": None,
                    }
                },
            )
        if self.posts == 2:
            raise InferenceAPIError(
                428,
                {
                    "detail": {
                        "error": "hardware_discovery_required",
                        "discovery_request_id": "job-1",
                    }
                },
            )
        return {"choices": [{"message": {"content": "answer"}}]}


def test_ask_policy_approves_waits_and_retries_loaded_model() -> None:
    client = FakeClient()

    response = client.infer(
        InferenceRequest(model="model-a", prompt="hello")
    )

    assert response["choices"][0]["message"]["content"] == "answer"
    assert any(path.endswith("/discoveries/job-1") for _, path, _ in client.calls)
    final_payload = client.calls[-1][2]
    assert final_payload["cognityx"]["load_policy"] == "require_loaded"
    assert final_payload["cognityx"]["client_type"] == "openai"


def test_inference_payload_carries_top_log_probabilities() -> None:
    payload = CognityxInferenceClient._inference_payload(
        InferenceRequest(model="model-a", prompt="hello", top_log_probabilities=5),
        DiscoveryPolicy.REQUIRE_EXISTING,
    )
    assert payload["top_logprobs"] == 5


def test_empty_base_url_uses_environment_then_local_default(monkeypatch) -> None:
    monkeypatch.setenv("COGNITYX_INFERENCE_URL", "http://127.0.0.1:8013/")
    assert CognityxInferenceClient("").base_url == "http://127.0.0.1:8013"

    monkeypatch.delenv("COGNITYX_INFERENCE_URL")
    assert CognityxInferenceClient(None).base_url == "http://127.0.0.1:8000"


def test_certified_profile_client_paths() -> None:
    class Client(CognityxInferenceClient):
        def __init__(self):
            super().__init__("http://example")
            self.paths = []

        def _request(self, method, path, payload=None):
            self.paths.append((method, path))
            return [] if path.startswith("/v1/cognityx/certified-profiles?") else {"profile_id": "p-1"}

    client = Client()
    assert client.list_certified_profiles(model="org/model", profile="int4") == []
    assert client.get_certified_profile("p-1") == {"profile_id": "p-1"}
    assert client.paths == [
        ("GET", "/v1/cognityx/certified-profiles?model=org%2Fmodel&profile=int4"),
        ("GET", "/v1/cognityx/certified-profiles/p-1"),
    ]


def test_auto_discovery_publishes_its_job_id_before_waiting() -> None:
    published = []
    events = []
    client = FakeClient()
    client.on_discovery_started = published.append
    client.on_discovery_event = events.append
    client.stream_discovery = lambda job_id, after=0: iter(
        [{"event": "discovery_completed", "job_id": job_id}]
    )

    client.infer(InferenceRequest(model="model-a", prompt="hello"))

    assert published == [
        {
            "event": "hardware_discovery_started",
            "job_id": "job-1",
            "model": None,
            "backend": None,
            "profile": None,
        }
    ]
    assert events == [{"event": "discovery_completed", "job_id": "job-1"}]


def test_ask_policy_preserves_discovery_error_without_a_terminal() -> None:
    client = FakeClient()
    client.input_fn = lambda prompt: (_ for _ in ()).throw(EOFError())

    try:
        client.infer(InferenceRequest(model="model-a", prompt="hello"))
    except InferenceAPIError as exc:
        assert exc.status == 428
        assert exc.payload["detail"]["error"] == "hardware_discovery_required"
    else:
        raise AssertionError("Expected the original discovery requirement")


def test_sse_close_uses_durable_completed_status() -> None:
    events = []
    client = FakeClient()
    client.on_discovery_event = events.append
    client.stream_discovery = lambda job_id, after=0: iter(())

    response = client.infer(InferenceRequest(model="model-a", prompt="hello"))

    assert response["choices"][0]["message"]["content"] == "answer"
    assert events == []


def test_sse_disconnect_reconnects_from_last_sequence() -> None:
    events = []
    attempts = []
    client = FakeClient()
    client.on_discovery_event = events.append
    states = iter(({"state": "running"}, {"state": "completed"}))
    original_request = client._request

    def request(method, path, payload=None):
        if method == "GET":
            return next(states)
        return original_request(method, path, payload)

    def stream(job_id, after=0):
        attempts.append(after)
        if len(attempts) == 1:
            raise URLError("connection closed")
        return iter(
            [
                {
                    "sequence": 1,
                    "event": "trial_completed",
                    "job_id": job_id,
                },
                {
                    "sequence": 2,
                    "event": "discovery_completed",
                    "job_id": job_id,
                },
            ]
        )

    client._request = request
    client.stream_discovery = stream

    client.infer(InferenceRequest(model="model-a", prompt="hello"))

    assert attempts == [0, 0]
    assert [event["sequence"] for event in events] == [1, 2]


def test_interrupted_discovery_has_actionable_error() -> None:
    client = FakeClient()
    client.on_discovery_event = lambda event: None
    client.stream_discovery = lambda job_id, after=0: iter(())
    original_request = client._request

    def request(method, path, payload=None):
        if method == "GET":
            return {"state": "interrupted"}
        return original_request(method, path, payload)

    client._request = request

    try:
        client.infer(InferenceRequest(model="model-a", prompt="hello"))
    except RuntimeError as exc:
        assert "server restart" in str(exc)
    else:
        raise AssertionError("Expected interrupted discovery to fail")


def test_transient_sse_and_status_disconnects_are_retried() -> None:
    client = FakeClient()
    client.on_discovery_event = lambda event: None
    client.stream_discovery = lambda job_id, after=0: (
        _ for _ in ()
    )
    original_request = client._request
    status_attempts = 0

    def request(method, path, payload=None):
        nonlocal status_attempts
        if method == "GET":
            status_attempts += 1
            if status_attempts == 1:
                raise URLError("temporarily unavailable")
            return {"state": "completed"}
        return original_request(method, path, payload)

    client._request = request

    response = client.infer(InferenceRequest(model="model-a", prompt="hello"))

    assert response["choices"][0]["message"]["content"] == "answer"
    assert status_attempts == 2


def test_stream_chat_preloads_model_and_yields_sse_chunks(monkeypatch) -> None:
    client = CognityxInferenceClient("http://example")
    loads = []
    requests = []
    client.load_model = lambda *args, **kwargs: loads.append((args, kwargs))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(
                [
                    b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                    b"\n",
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":3}}\n',
                    b"\n",
                    b"data: [DONE]\n",
                ]
            )

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(
        "cognityx_inference.client.urllib_request.urlopen", urlopen
    )

    chunks = list(
        client.stream_chat(
            model="model-a",
            prompt="hello",
            backend="vllm",
            profile="int4",
            max_tokens=8,
        )
    )

    assert len(loads) == 1
    assert loads[0][0][:3] == ("model-a", "vllm", "int4")
    assert requests[0]["stream"] is True
    assert requests[0]["cognityx"]["load_policy"] == "require_loaded"
    assert requests[0]["cognityx"]["client_type"] == "openai"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[-1]["usage"]["total_tokens"] == 3


def test_stream_chat_surfaces_server_error_event(monkeypatch) -> None:
    client = CognityxInferenceClient("http://example")
    client.load_model = lambda *args, **kwargs: {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(
                [b'data: {"error":{"message":"generation failed"}}\n']
            )

    monkeypatch.setattr(
        "cognityx_inference.client.urllib_request.urlopen",
        lambda *args, **kwargs: Response(),
    )

    try:
        list(client.stream_chat(model="model-a", prompt="hello"))
    except RuntimeError as exc:
        assert str(exc) == "generation failed"
    else:
        raise AssertionError("Expected the streamed server error")
