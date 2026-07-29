from __future__ import annotations

import os
from pathlib import Path

import pytest

from cognityx_inference.configuration import InferenceConfiguration
from cognityx_inference.providers.diagnostics import test_provider as diagnose


@pytest.mark.parametrize(
    "provider_name",
    (
        "openai",
        "groq",
        "gemini",
        "cerebras",
        "openrouter",
        "github_models",
        "cloudflare",
        "anthropic",
    ),
)
def test_opt_in_live_provider(provider_name: str) -> None:
    if os.environ.get("COGNITYX_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("set COGNITYX_LIVE_PROVIDER_TESTS=1 to enable live calls")
    configuration_path = os.environ.get("COGNITYX_INFERENCE_CONFIG")
    if not configuration_path:
        pytest.skip("COGNITYX_INFERENCE_CONFIG is not configured")
    configuration = InferenceConfiguration.load(Path(configuration_path))
    definition = configuration.providers[provider_name]
    if (
        provider_name == "groq"
        and os.environ.get("COGNITYX_LIVE_GROQ_CONFIRMED") != "1"
    ):
        pytest.skip(
            "set COGNITYX_LIVE_GROQ_CONFIRMED=1 only after rotating the "
            "previously rejected Groq credential"
        )
    resolver = configuration.credential_resolver()
    if not resolver.available(
        environment_name=definition.api_key_env,
        secret_name=definition.api_key_secret,
    ):
        pytest.skip(f"No credential source is available for {provider_name}")

    result = diagnose(definition, credential_resolver=resolver)

    assert result.status == "passed", result.error_category
