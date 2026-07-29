from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.contracts import InferenceRequest
from cognityx_inference.errors import ProviderRequestError
from cognityx_inference.providers.factory import build_provider_adapters
from cognityx_inference.providers.native import (
    AnthropicProvider,
    GeminiNativeProvider,
)
from cognityx_inference.providers.openai_compatible import OpenAICompatibleProvider
from cognityx_inference.security import CredentialResolver


def _serve(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.mark.parametrize(
    ("name", "adapter_name"),
    (
        ("openai", "openai_compatible"),
        ("groq", "openai_compatible"),
        ("cerebras", "openai_compatible"),
        ("openrouter", "openai_compatible"),
        ("github_models", "github_models"),
        ("cloudflare", "openai_compatible"),
    ),
)
def test_openai_compatible_presets_normalize_mocked_calls(
    name: str,
    adapter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler(_QuietHandler):
        def do_GET(self) -> None:
            body = json.dumps({"data": [{"id": "provider/model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            if payload.get("stream"):
                body = (
                    'data: {"id":"stream","model":"provider/model",'
                    '"choices":[{"delta":{"content":"OK"},'
                    '"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":1,"completion_tokens":1,'
                    '"total_tokens":2}}\n\n'
                    "data: [DONE]\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        "id": "plain",
                        "model": "provider/model",
                        "choices": [
                            {
                                "message": {"content": "OK"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread = _serve(Handler)
    environment = f"{name.upper()}_KEY"
    monkeypatch.setenv(environment, "not-a-real-secret")
    try:
        definition = ProviderDefinition(
            name=name,
            adapter=adapter_name,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key_env=environment,
            api_key_secret=environment,
            enabled=True,
            model_capabilities={
                "provider/model": CertifiedContextProfile(
                    max_context_tokens=1024,
                    allow_estimated_counting=True,
                )
            },
            options=(
                {"models_url": (f"http://127.0.0.1:{server.server_port}/v1/models")}
                if name == "github_models"
                else {}
            ),
        )
        adapter = build_provider_adapters({name: definition}, CredentialResolver())[
            name
        ]
        assert adapter.list_models() == ("provider/model",)
        request = InferenceRequest(
            model="provider/model",
            prompt="OK",
            provider=name,
            max_output_tokens=2,
        )
        assert adapter.infer(request).content == "OK"
        chunks: list[str] = []
        streamed = adapter.infer(
            InferenceRequest(
                model="provider/model",
                prompt="OK",
                provider=name,
                max_output_tokens=2,
                stream=True,
            ),
            chunks.append,
        )
        assert streamed.content == "OK"
        assert chunks == ["OK"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_gemini_native_normalizes_models_completion_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler(_QuietHandler):
        def do_GET(self) -> None:
            body = json.dumps({"models": [{"name": "models/gemini-test"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if "streamGenerateContent" in self.path:
                body = (
                    'data: {"candidates":[{"content":{"parts":[{"text":"OK"}]},'
                    '"finishReason":"STOP"}],"usageMetadata":'
                    '{"promptTokenCount":1,"candidatesTokenCount":1,'
                    '"totalTokenCount":2}}\n\n'
                ).encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "OK"}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 1,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 2,
                        },
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread = _serve(Handler)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-secret")
    definition = ProviderDefinition(
        name="gemini",
        adapter="gemini_native",
        base_url=f"http://127.0.0.1:{server.server_port}",
        api_key_env="GEMINI_API_KEY",
        enabled=True,
    )
    adapter = GeminiNativeProvider(definition, CredentialResolver())
    try:
        assert adapter.list_models() == ("models/gemini-test",)
        request = InferenceRequest(
            model="models/gemini-test",
            prompt="OK",
            provider="gemini",
            max_output_tokens=2,
        )
        assert adapter.infer(request).usage.total_tokens == 2
        chunks: list[str] = []
        result = adapter.infer(
            InferenceRequest(
                model="models/gemini-test",
                prompt="OK",
                provider="gemini",
                max_output_tokens=2,
                stream=True,
            ),
            chunks.append,
        )
        assert result.content == "OK"
        assert chunks == ["OK"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_anthropic_normalizes_models_completion_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler(_QuietHandler):
        def do_GET(self) -> None:
            body = json.dumps({"data": [{"id": "claude-test"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            if payload.get("stream"):
                body = (
                    'data: {"type":"message_start","message":{"usage":'
                    '{"input_tokens":1}}}\n\n'
                    'data: {"type":"content_block_delta","delta":'
                    '{"type":"text_delta","text":"OK"}}\n\n'
                    'data: {"type":"message_delta","delta":'
                    '{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
                ).encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        "id": "message",
                        "model": "claude-test",
                        "content": [{"type": "text", "text": "OK"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread = _serve(Handler)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-secret")
    definition = ProviderDefinition(
        name="anthropic",
        adapter="anthropic",
        base_url=f"http://127.0.0.1:{server.server_port}",
        api_key_env="ANTHROPIC_API_KEY",
        api_version="2023-06-01",
        enabled=True,
    )
    adapter = AnthropicProvider(definition, CredentialResolver())
    try:
        assert adapter.list_models() == ("claude-test",)
        request = InferenceRequest(
            model="claude-test",
            prompt="OK",
            provider="anthropic",
            max_output_tokens=2,
        )
        assert adapter.infer(request).usage.total_tokens == 2
        chunks: list[str] = []
        result = adapter.infer(
            InferenceRequest(
                model="claude-test",
                prompt="OK",
                provider="anthropic",
                max_output_tokens=2,
                stream=True,
            ),
            chunks.append,
        )
        assert result.content == "OK"
        assert chunks == ["OK"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    ("status", "category"),
    (
        (401, "authentication_failed"),
        (403, "permission_denied"),
        (429, "rate_limited"),
        (500, "provider_unavailable"),
    ),
)
def test_provider_http_errors_are_safely_classified(
    status: int,
    category: str,
) -> None:
    class Handler(_QuietHandler):
        def do_GET(self) -> None:
            body = b'{"error":{"type":"safe_type"}}'
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread = _serve(Handler)
    provider = OpenAICompatibleProvider(
        "test",
        f"http://127.0.0.1:{server.server_port}/v1",
        "not-a-real-secret",
    )
    try:
        with pytest.raises(ProviderRequestError) as captured:
            provider.list_models()
        assert captured.value.category == category
        assert "not-a-real-secret" not in str(captured.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_invalid_json_and_malformed_stream_are_normalized() -> None:
    class Handler(_QuietHandler):
        def do_GET(self) -> None:
            body = b"not-json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            body = b"data: {not-json}\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread = _serve(Handler)
    provider = OpenAICompatibleProvider(
        "test",
        f"http://127.0.0.1:{server.server_port}/v1",
        "not-a-real-secret",
    )
    try:
        with pytest.raises(ProviderRequestError, match="invalid_response"):
            provider.list_models()
        with pytest.raises(ProviderRequestError, match="invalid_response"):
            provider.infer(
                InferenceRequest(
                    model="provider/model",
                    prompt="OK",
                    provider="test",
                    stream=True,
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_timeout_is_normalized_without_exposing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise TimeoutError

    monkeypatch.setattr(
        "cognityx_inference.providers.openai_compatible.urllib_request.urlopen",
        timeout,
    )
    provider = OpenAICompatibleProvider(
        "test", "https://provider.invalid/v1", "not-a-real-secret"
    )

    with pytest.raises(ProviderRequestError) as captured:
        provider.list_models(timeout_seconds=0.01)

    assert captured.value.category == "network_error"
    assert "not-a-real-secret" not in str(captured.value)
