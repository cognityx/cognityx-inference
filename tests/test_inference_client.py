from __future__ import annotations

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
