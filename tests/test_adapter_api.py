from __future__ import annotations

from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cognityx_inference.api import create_app
from cognityx_inference.contracts import FinishReason, InferenceResponse, TokenUsage


class FakeResearchRunner:
    def __init__(self) -> None:
        self.request = None
        self.owner_id = None

    def run(self, request, *, owner_id):
        self.request = request
        self.owner_id = owner_id
        return {"schema": "cognityx.inference.pair/v1", "pair_validation": "passed"}


class FakeService:
    def __init__(self) -> None:
        self.models = SimpleNamespace(statuses=lambda: ())
        self.provider_registry = None
        self.research_runner = FakeResearchRunner()
        self.request = None

    def infer(self, request, on_text=None, *, owner_id="local"):
        del on_text, owner_id
        self.request = request
        return InferenceResponse(
            request_id="request-1",
            content="OK",
            model=request.model,
            provider="local",
            backend=request.backend,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_existing_no_adapter_http_request_remains_compatible() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "OK"
    assert service.request.adapter_manifest_uri is None
    assert service.request.thinking.value == "disabled"


def test_http_propagates_explicit_thinking() -> None:
    service = FakeService()
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "hello"}],
            "cognityx": {"thinking": "enabled"},
        },
    )

    assert response.status_code == 200
    assert service.request.thinking.value == "enabled"


def test_http_adapter_fields_and_research_pair_are_additive() -> None:
    service = FakeService()
    client = TestClient(create_app(service))
    adapter_uri = "storage://local-main/models/adapter/manifest.json"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "model-a",
            "messages": [{"role": "user", "content": "hello"}],
            "cognityx": {
                "model_revision": "commit-1",
                "adapter_manifest_uri": adapter_uri,
                "adapter_purpose": "evaluation",
            },
        },
    )
    assert response.status_code == 200
    assert service.request.model_revision == "commit-1"
    assert service.request.adapter_manifest_uri == adapter_uri

    pair = client.post(
        "/v1/cognityx/research/pairs",
        json={
            "evaluation_manifest_uri": "storage://local-main/datasets/eval/manifest.json",
            "adapter_manifest_uri": adapter_uri,
            "model": "model-a",
            "experiment_id": "exp-1",
        },
    )
    assert pair.status_code == 200
    assert pair.json()["pair_validation"] == "passed"
    assert service.research_runner.request.context.experiment_id == "exp-1"
    assert service.research_runner.owner_id == "local"
