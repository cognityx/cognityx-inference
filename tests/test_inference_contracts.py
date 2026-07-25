from __future__ import annotations

import pytest

from cognityx_inference.contracts import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
)


def test_request_accepts_common_and_extension_parameters() -> None:
    request = InferenceRequest(
        model="model-a",
        prompt="hello",
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        max_tokens=100,
        reasoning={"effort": "high"},
        extensions={"runtime": {"quantization": "int4"}},
    )

    assert request.to_dict()["top_k"] == 40
    assert request.extensions["runtime"]["quantization"] == "int4"


def test_request_requires_exactly_one_input_shape() -> None:
    with pytest.raises(ValueError, match="messages or prompt"):
        InferenceRequest(model="model-a")
    with pytest.raises(ValueError, match="mutually exclusive"):
        InferenceRequest(
            model="model-a",
            prompt="hello",
            messages=({"role": "user", "content": "hello"},),
        )


def test_unavailable_response_fields_remain_null() -> None:
    response = InferenceResponse(
        request_id="request-1",
        content="answer",
        model="model-a",
        provider="openai",
        backend="commercial-api",
        finish_reason=FinishReason.STOP,
    )

    value = response.to_dict()
    assert value["usage"]["prompt_tokens"] is None
    assert value["timings"]["time_to_first_token_seconds"] is None
    assert value["finish_reason"] == "stop"
