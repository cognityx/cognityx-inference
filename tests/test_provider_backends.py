from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.configuration import InferenceConfiguration
from cognityx_inference.contracts import InferenceRequest
from cognityx_inference.errors import ProviderCredentialMissing
from cognityx_inference.providers import OpenAICompatibleProvider
from cognityx_inference.security import redact


def _provider(api_key: str | None = "secret-value") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "test",
        "https://provider.invalid/v1",
        api_key,
        api_key_env="TEST_PROVIDER_KEY",
        model_capabilities={
            "model": CertifiedContextProfile(
                max_context_tokens=100,
                reserved_tokens=5,
                allow_estimated_counting=True,
            )
        },
    )


def test_provider_maps_common_parameters_and_filters_extensions() -> None:
    provider = _provider()
    payload = provider._payload(
        InferenceRequest(
            model="model",
            messages=({"role": "user", "content": "hello"},),
            provider="test",
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=12,
            stop=("DONE",),
            tools=({"type": "function", "function": {"name": "lookup"}},),
            response_format={"type": "json_object"},
            extensions={"frequency_penalty": 0.1},
        )
    )

    assert payload["max_completion_tokens"] == 12
    assert payload["temperature"] == 0.2
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["frequency_penalty"] == 0.1
    assert "max_output_tokens" not in payload

    with pytest.raises(ValueError, match="unsupported extension"):
        provider._payload(
            InferenceRequest(
                model="model",
                prompt="hello",
                extensions={"api_key": "must-not-forward"},
            )
        )


def test_secret_free_configuration_loads_profiles_and_provider_capabilities(
    tmp_path,
) -> None:
    path = tmp_path / "inference.toml"
    path.write_text(
        """
[manager]
port = 9010

[server_profiles.local]
model = "org/model"
certified_profile_id = "profile-1"

[providers.groq]
adapter = "openai_compatible"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
smoke_test_model = "configured-model"

[providers.groq.models.configured-model.context]
max_context_tokens = 8192
max_output_tokens_limit = 1024
reserved_tokens = 128
allow_estimated_counting = true
""",
        encoding="utf-8",
    )

    configuration = InferenceConfiguration.load(path)

    assert configuration.manager.port == 9010
    assert (
        configuration.server_profiles["local"].certified_profile_id
        == "profile-1"
    )
    groq = configuration.providers["groq"]
    assert groq.api_key_env == "GROQ_API_KEY"
    assert groq.model_capabilities[
        "configured-model"
    ].max_output_tokens_limit == 1024


def test_missing_credential_names_variable_without_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    provider = _provider(api_key=None)

    with pytest.raises(ProviderCredentialMissing) as captured:
        provider.list_models()

    assert "TEST_PROVIDER_KEY" in str(captured.value)
    assert "secret" not in str(captured.value)


def test_secret_redaction_removes_known_and_credential_shaped_values() -> None:
    value = {
        "Authorization": "Bearer secret-value",
        "nested": {"api_key": "secret-value"},
        "message": "request used sk-test-abcdefghijklmnopqrstuvwxyz",
    }

    rendered = json.dumps(redact(value, secrets=("secret-value",)))

    assert "secret-value" not in rendered
    assert "sk-test-" not in rendered
    assert "[REDACTED]" in rendered


def test_openai_compatible_provider_against_mock_http_server() -> None:
    received: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/v1/models"
            assert self.headers["Authorization"] == "Bearer test-key"
            body = json.dumps({"data": [{"id": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            received.append(payload)
            if payload.get("stream"):
                events = (
                    'data: {"id":"req-stream","model":"model",'
                    '"choices":[{"delta":{"content":"O"}}]}\n\n'
                    'data: {"id":"req-stream","model":"model",'
                    '"choices":[{"delta":{"content":"K"},'
                    '"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":2,"completion_tokens":1,'
                    '"total_tokens":3}}\n\n'
                    "data: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(events)))
                self.end_headers()
                self.wfile.write(events)
                return
            body = json.dumps(
                {
                    "id": "req-plain",
                    "model": "model",
                    "choices": [
                        {
                            "message": {"content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleProvider(
            "test",
            f"http://127.0.0.1:{server.server_port}/v1",
            "test-key",
        )
        assert provider.list_models() == ("model",)
        request = InferenceRequest(
            model="model",
            prompt="Reply OK",
            provider="test",
            max_output_tokens=4,
        )
        plain = provider.infer(request)
        chunks: list[str] = []
        streamed = provider.infer(
            InferenceRequest(
                model="model",
                prompt="Reply OK",
                provider="test",
                max_output_tokens=4,
                stream=True,
            ),
            on_text=chunks.append,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert plain.content == "OK"
    assert plain.finish_reason.value == "stop"
    assert plain.usage.total_tokens == 3
    assert streamed.content == "OK"
    assert streamed.usage.total_tokens == 3
    assert chunks == ["O", "K"]
    assert received[0]["max_completion_tokens"] == 4
