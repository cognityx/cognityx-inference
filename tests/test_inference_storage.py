from __future__ import annotations

from cognityx_inference.contracts import InferenceRequest, InferenceResponse
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    InferenceArtifactRepository,
)


class FakeStorage:
    def __init__(self) -> None:
        self.values = {}
        self.materialized = []

    def put_json(self, key, value):
        self.values[key] = value
        return {"key": key}

    def materialize(self, key):
        self.materialized.append(key)
        return "/opaque/materialized/value"


def test_inference_records_use_logical_keys() -> None:
    storage = FakeStorage()
    repository = InferenceArtifactRepository(storage)
    request = InferenceRequest(model="model-a", prompt="hello")
    response = InferenceResponse(
        request_id="request/unsafe",
        content="answer",
        model="model-a",
        provider="local",
        backend="transformers",
    )

    stored = repository.save_exchange(request, response)

    assert stored["key"].startswith("inference/requests/")
    assert stored["key"].endswith("/request-unsafe.json")
    assert not stored["key"].startswith("/")


def test_model_paths_only_come_from_storage_materialization() -> None:
    storage = FakeStorage()
    repository = InferenceArtifactRepository(storage)

    result = repository.materialize_model("models/qwen/version-1")

    assert result == "/opaque/materialized/value"
    assert storage.materialized == ["models/qwen/version-1"]


def test_boundary_records_are_append_only_per_trial() -> None:
    storage = FakeStorage()
    repository = BoundaryArtifactRepository(storage)

    repository.save_trial("job-1", "trial-1", {"status": "completed"})

    assert (
        "evaluations/hardware-boundary/job-1/trials/trial-1.json"
        in storage.values
    )
