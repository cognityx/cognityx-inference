from __future__ import annotations

from types import SimpleNamespace

import pytest

from cognityx_inference.backends.legacy import VLLMBackend, _settings
from cognityx_inference.api import _request_from_chat_payload
from cognityx_inference.client import CognityxInferenceClient
from cognityx_inference.contracts import (
    DiscoveryPolicy,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
    ThinkingMode,
    ThinkingResolution,
)
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.research import (
    InferencePairRequest,
    InferencePairRunner,
    ResearchContext,
)
from cognityx_inference.service import InferenceService


def test_thinking_defaults_disabled_and_accepts_explicit_modes() -> None:
    default = InferenceRequest(model="model", prompt="hello")
    enabled = InferenceRequest(model="model", prompt="hello", thinking="enabled")
    disabled = InferenceRequest(model="model", prompt="hello", thinking="disabled")

    assert default.thinking is ThinkingMode.DISABLED
    assert enabled.thinking is ThinkingMode.ENABLED
    assert disabled.thinking is ThinkingMode.DISABLED
    assert _settings(default).enable_thinking is False
    assert _settings(enabled).enable_thinking is True


def test_qwen_native_template_mechanism_is_resolved_and_counted() -> None:
    calls: list[dict[str, object]] = []

    class Tokenizer:
        chat_template = "{% if enable_thinking %}<think>{% endif %}"

        def apply_chat_template(self, messages, **kwargs):
            del messages
            calls.append(kwargs)
            return [1, 2, 3]

    backend = VLLMBackend("Qwen/Qwen3-8B", {"context_length": 40960})
    backend.engine = SimpleNamespace(tokenizer=Tokenizer(), processor=None)

    resolution = backend.resolve_thinking(ThinkingMode.DISABLED)
    count = backend.count_input_tokens(
        InferenceRequest(model="Qwen/Qwen3-8B", prompt="hello")
    )

    assert resolution == ThinkingResolution(
        requested=ThinkingMode.DISABLED,
        effective=ThinkingMode.DISABLED,
        mechanism="chat_template_kwarg:enable_thinking",
    )
    assert count == 3
    assert calls[-1]["enable_thinking"] is False


def test_enabled_thinking_fails_when_model_has_no_native_toggle() -> None:
    backend = VLLMBackend("plain-model", {"context_length": 1024})
    backend.engine = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="ordinary template"),
        processor=None,
    )

    with pytest.raises(ValueError, match="does not expose a native thinking toggle"):
        backend.resolve_thinking(ThinkingMode.ENABLED)

    assert (
        backend.resolve_thinking(ThinkingMode.DISABLED).mechanism
        == "ordinary_generation:no_thinking_capability"
    )


class ThinkingBackend:
    capabilities = ModelCapabilities(lifecycle=True, thinking=True)

    def __init__(self, model: str, runtime: dict) -> None:
        self.model_name = model
        self.runtime = runtime
        self.engine = SimpleNamespace(settings=None, model_runtime={})

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, str]:
        return {"name": self.model_name, "chat_template_checksum": "template"}

    def resolve_thinking(self, requested: ThinkingMode) -> ThinkingResolution:
        return ThinkingResolution(
            requested=requested,
            effective=requested,
            mechanism="chat_template_kwarg:enable_thinking",
        )

    def infer(self, request, on_text=None):
        del on_text
        self.engine.settings = _settings(request)
        return InferenceResponse(
            request_id="request",
            content="answer",
            model=request.model,
            provider="local",
            backend="fake",
            finish_reason=FinishReason.STOP,
        )


def test_service_reports_effective_thinking_and_fingerprints_it() -> None:
    service = InferenceService(ModelManager({"fake": ThinkingBackend}))
    response = service.infer(
        InferenceRequest(
            model="model",
            prompt="hello",
            backend="fake",
            thinking=ThinkingMode.ENABLED,
        )
    )

    assert response.extensions["thinking"] == {
        "requested": "enabled",
        "effective": "enabled",
        "mechanism": "chat_template_kwarg:enable_thinking",
    }
    fingerprint = response.extensions["runtime_fingerprint"]
    assert fingerprint["decoding"]["thinking"] == response.extensions["thinking"]
    assert fingerprint["decoding"]["effective_generation"]["enable_thinking"] is True


def test_unsupported_backend_rejects_explicit_enabled_thinking() -> None:
    class Unsupported(ThinkingBackend):
        capabilities = ModelCapabilities(lifecycle=True, thinking=False)

    service = InferenceService(ModelManager({"fake": Unsupported}))
    with pytest.raises(ValueError, match="thinking"):
        service.infer(
            InferenceRequest(
                model="model", prompt="hello", backend="fake", thinking="enabled"
            )
        )


def test_client_payload_propagates_thinking() -> None:
    payload = CognityxInferenceClient._inference_payload(
        InferenceRequest(model="model", prompt="hello", thinking="enabled"),
        DiscoveryPolicy.REQUIRE_EXISTING,
    )
    assert payload["cognityx"]["thinking"] == "enabled"


def test_http_payload_propagates_thinking() -> None:
    request = _request_from_chat_payload(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "cognityx": {"thinking": "enabled"},
        }
    )
    assert request.thinking is ThinkingMode.ENABLED


def test_research_pair_defaults_are_controlled() -> None:
    request = InferencePairRequest.from_dict(
        {
            "evaluation_manifest_uri": "storage://local-main/datasets/eval/manifest.json",
            "adapter_manifest_uri": "storage://local-main/models/adapter/manifest.json",
            "model": "model",
            "research_context": {"experiment_id": "exp"},
        }
    )
    assert request.thinking is ThinkingMode.DISABLED
    assert request.max_output_tokens == 512


def test_prediction_preserves_length_termination_and_thinking() -> None:
    response = InferenceResponse(
        request_id="request",
        content="partial",
        model="model",
        provider="local",
        backend="fake",
        finish_reason=FinishReason.LENGTH,
        extensions={
            "thinking": {
                "requested": "disabled",
                "effective": "disabled",
                "mechanism": "chat_template_kwarg:enable_thinking",
            }
        },
    )
    row = InferencePairRunner._prediction_row(
        "run",
        "pair",
        SimpleNamespace(
            manifest={
                "evaluation_set_id": "eval",
                "evaluation_set_version": "v1",
                "research_role": "paraphrase_evaluation",
            }
        ),
        {"record_id": "record", "question": "Question?"},
        response,
        {"sha256": "runtime"},
        "base",
    )
    assert row["finish_reason"] == "length"
    assert row["thinking"]["effective"] == "disabled"
