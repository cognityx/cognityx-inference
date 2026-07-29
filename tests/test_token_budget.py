from __future__ import annotations

import pytest

from cognityx_inference.capabilities import (
    CertifiedContextProfile,
    CertifiedInferenceProfile,
    RuntimeCompatibility,
    hardware_fingerprint,
)
from cognityx_inference.contracts import (
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
)
from cognityx_inference.errors import ContextWindowExceeded
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.service import InferenceService
from cognityx_inference.token_budget import (
    apply_token_budget,
    serialized_request,
)


def test_automatic_output_budget_uses_profile_limit() -> None:
    request = InferenceRequest(model="model", prompt="hello")
    dispatched, budget = apply_token_budget(
        request,
        CertifiedContextProfile(
            max_context_tokens=100,
            max_output_tokens_limit=20,
            reserved_tokens=5,
        ),
        lambda _: 10,
    )

    assert dispatched.max_output_tokens == 20
    assert budget.effective_max_output_tokens == 20
    assert budget.requested_max_output_tokens is None
    assert budget.source == "automatic"


def test_manual_output_budget_overrides_default_without_clamping() -> None:
    request = InferenceRequest(
        model="model", prompt="hello", max_output_tokens=40
    )
    dispatched, budget = apply_token_budget(
        request,
        CertifiedContextProfile(
            max_context_tokens=100,
            max_output_tokens_limit=20,
            reserved_tokens=5,
        ),
        lambda _: 10,
    )

    assert dispatched.max_output_tokens == 40
    assert budget.source == "manual"


def test_context_overflow_is_rejected_before_dispatch() -> None:
    request = InferenceRequest(
        model="model", prompt="hello", max_output_tokens=11
    )
    with pytest.raises(ContextWindowExceeded):
        apply_token_budget(
            request,
            CertifiedContextProfile(
                max_context_tokens=20,
                reserved_tokens=2,
            ),
            lambda _: 8,
        )


def test_serialized_request_includes_history_tools_and_schema() -> None:
    request = InferenceRequest(
        model="model",
        messages=(
            {"role": "system", "content": "system marker"},
            {"role": "assistant", "content": "history marker"},
            {"role": "user", "content": "question marker"},
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            },
        ),
        tool_choice={"type": "function", "function": {"name": "weather"}},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"forecast": {"type": "string"}},
                },
            },
        },
    )

    serialized = serialized_request(request)

    for marker in (
        "system marker",
        "history marker",
        "question marker",
        "weather",
        "city",
        "forecast",
    ):
        assert marker in serialized


def test_service_dispatches_calculated_budget_from_new_certified_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {"gpu_name": "test-gpu"}
    compatibility = RuntimeCompatibility(
        hardware_fingerprint=hardware_fingerprint(inventory),
        model="model",
        model_revision="revision",
        backend="fake",
        backend_version="1",
        profile="bf16",
        kv_cache_precision="auto",
    )
    profile = CertifiedInferenceProfile(
        profile_id="profile",
        created_at="2026-01-01T00:00:00+00:00",
        compatibility=compatibility,
        model_context_limit=100,
        maximum_certified_context_length=100,
        context=CertifiedContextProfile(
            max_context_tokens=100,
            max_output_tokens_limit=25,
            reserved_tokens=5,
        ),
    )

    class Profiles:
        def find_compatible(self, selected):
            return profile if selected == compatibility else None

    class Backend:
        capabilities = ModelCapabilities()
        dispatched: InferenceRequest | None = None

        def __init__(self, model, runtime):
            self.model = model

        def load(self):
            return None

        def count_input_tokens(self, request):
            assert "tool-marker" in serialized_request(request)
            return 10

        def infer(self, request, on_text=None):
            Backend.dispatched = request
            return InferenceResponse(
                request_id="request",
                content="OK",
                model=request.model,
                provider="local",
                backend="fake",
            )

        def unload(self):
            return None

    monkeypatch.setattr(
        "cognityx_inference.service.discover_model_context_limit",
        lambda *args, **kwargs: (100, "revision"),
    )
    monkeypatch.setattr(
        "cognityx_inference.service._backend_version",
        lambda backend: "1",
    )
    service = InferenceService(
        ModelManager({"fake": Backend}),
        certified_profiles=Profiles(),
        inventory=lambda: inventory,
    )

    response = service.infer(
        InferenceRequest(
            model="model",
            messages=({"role": "user", "content": "hello"},),
            tools=(
                {
                    "type": "function",
                    "function": {"name": "tool-marker"},
                },
            ),
            backend="fake",
        )
    )

    assert Backend.dispatched is not None
    assert Backend.dispatched.max_output_tokens == 25
    assert response.token_budget is not None
    assert response.token_budget.effective_max_output_tokens == 25
