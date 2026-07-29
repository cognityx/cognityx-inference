from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.configuration import (
    InferenceConfiguration,
    ProviderDefinition,
)
from cognityx_inference.contracts import InferenceRequest
from cognityx_inference.errors import UnsupportedProviderParameter
from cognityx_inference.providers.factory import build_provider_adapters
from cognityx_inference.providers.models import (
    ProviderAccountLimits,
    ProviderModelProfile,
    ProviderStatus,
)
from cognityx_inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from cognityx_inference.providers.registry import ProviderRegistry
from cognityx_inference.routing import (
    PurposeRoute,
    RouteTarget,
    RoutingPolicy,
)
from cognityx_inference.security import CredentialResolver

PRIMARY = {
    "local",
    "openai",
    "groq",
    "gemini",
    "cerebras",
    "openrouter",
    "github_models",
    "cloudflare",
    "anthropic",
}


def test_default_registry_contains_primary_and_enterprise_extension_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_MODELS_TOKEN",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    configuration = InferenceConfiguration.load(cwd="/nonexistent")
    assert PRIMARY <= set(configuration.providers)
    assert {"azure_openai", "aws_bedrock", "google_vertex"} <= set(
        configuration.providers
    )
    registry = ProviderRegistry(
        configuration.providers,
        build_provider_adapters(
            configuration.providers, configuration.credential_resolver()
        ),
        configuration.credential_resolver(),
    )
    statuses = {item.provider: item for item in registry.list_statuses()}
    assert statuses["local"].credential_status == "not_required"
    assert statuses["openai"].credential_status == "not_configured"
    assert statuses["anthropic"].credential_status == "not_configured"
    assert statuses["aws_bedrock"].configuration_status == "disabled"
    groq = registry.profile("groq", "llama-3.1-8b-instant")
    assert groq.state.value == "unverified"
    assert groq.context is None
    assert "logprobs" in groq.parameter_policy.rejected_when_supplied
    assert "logprobs" in groq.to_dict()["parameter_policy"]["rejected_when_supplied"]


def test_secrets_file_supports_uppercase_and_legacy_lowercase_aliases(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "secrets.json"
    path.write_text('{"groq_api_key":"legacy"}', encoding="utf-8")
    resolver = CredentialResolver(str(path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert (
        resolver.resolve(
            environment_name="GROQ_API_KEY",
            secret_name="GROQ_API_KEY",
        )
        == "legacy"
    )
    monkeypatch.setenv("GROQ_API_KEY", "environment")
    assert (
        resolver.resolve(
            environment_name="GROQ_API_KEY",
            secret_name="GROQ_API_KEY",
        )
        == "environment"
    )


def test_groq_policy_rejects_explicit_unsupported_parameters() -> None:
    provider = OpenAICompatibleProvider(
        "groq",
        "https://api.groq.invalid/openai/v1",
        "unused",
    )
    base = dict(
        model="model",
        messages=({"role": "user", "content": "hello"},),
        provider="groq",
    )
    with pytest.raises(UnsupportedProviderParameter, match="logprobs"):
        provider._payload(InferenceRequest(**base, log_probabilities=True))
    with pytest.raises(UnsupportedProviderParameter, match=r"messages\[\]\.name"):
        provider._payload(
            InferenceRequest(
                **{
                    **base,
                    "messages": (
                        {"role": "user", "name": "caller", "content": "hello"},
                    ),
                }
            )
        )
    with pytest.raises(UnsupportedProviderParameter, match="n"):
        provider._payload(InferenceRequest(**base, extensions={"n": 2}))

    payload = provider._payload(InferenceRequest(**base, extensions={"n": 1}))
    assert "n" not in payload
    assert "logprobs" not in payload


def test_model_discovery_is_cached_and_preserves_exact_ids() -> None:
    class Adapter:
        name = "test"
        calls = 0

        def credential_available(self) -> bool:
            return True

        def list_models(self, *, timeout_seconds: float = 20) -> tuple[str, ...]:
            self.calls += 1
            return ("owner/model:free", "models/vendor/model")

    definition = ProviderDefinition(
        name="test",
        adapter="openai_compatible",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_KEY",
        enabled=True,
        discovery_cache_seconds=60,
    )
    adapter = Adapter()
    registry = ProviderRegistry(
        {"test": definition},
        {"test": adapter},
        CredentialResolver(),
    )
    first = registry.discover_models("test")
    second = registry.discover_models("test")

    assert first.models == ("owner/model:free", "models/vendor/model")
    assert second.cached is True
    assert adapter.calls == 1


def test_model_context_and_account_limits_are_separate() -> None:
    profile = ProviderModelProfile(
        provider="test",
        model_id="model",
        context=CertifiedContextProfile(max_context_tokens=8192),
        limits=ProviderAccountLimits(tokens_per_minute=1000),
    )
    assert profile.context.max_context_tokens == 8192
    assert profile.limits.tokens_per_minute == 1000


def test_sensitive_routing_fails_closed_and_honours_override() -> None:
    policy = RoutingPolicy(
        purposes={
            "validation": PurposeRoute(
                preferred=(
                    RouteTarget("local", "local-profile"),
                    RouteTarget("openai", "remote-profile"),
                )
            )
        }
    )
    statuses = (
        ProviderStatus(
            "local",
            "local",
            True,
            "not_required",
            "configured",
            None,
            "local-profile",
        ),
        ProviderStatus(
            "openai",
            "openai_compatible",
            True,
            "configured",
            "configured",
            "model",
            "remote-profile",
        ),
    )
    assert (
        policy.select("validation", statuses, data_classification="sensitive").provider
        == "local"
    )
    remote_only = (statuses[1],)
    with pytest.raises(PermissionError, match="sensitive data was not routed"):
        policy.select("validation", remote_only, data_classification="sensitive")
    with pytest.raises(PermissionError, match="not allowed"):
        policy.select(
            "validation",
            statuses,
            data_classification="sensitive",
            explicit_provider="openai",
        )


def test_prior_groq_403_is_classified_before_inference() -> None:
    posted = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"error": {"type": "invalid_api_key"}}).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            nonlocal posted
            posted = True
            self.send_response(500)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = OpenAICompatibleProvider(
            "groq",
            f"http://127.0.0.1:{server.server_port}/openai/v1",
            "rejected",
        )
        # The environment resolver is tested separately; direct adapter proves
        # the exact sanitized status classification without storing a key.
        with pytest.raises(Exception) as captured:
            adapter.list_models()
        assert getattr(captured.value, "category", None) == "permission_denied"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert posted is False
