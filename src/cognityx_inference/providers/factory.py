"""Configuration-driven provider adapter construction."""

from __future__ import annotations

from typing import Any, Mapping

from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.providers.native import (
    AnthropicProvider,
    GeminiNativeProvider,
    GitHubModelsProvider,
)
from cognityx_inference.providers.openai_compatible import OpenAICompatibleProvider
from cognityx_inference.security import CredentialResolver


def build_provider_adapters(
    definitions: Mapping[str, ProviderDefinition],
    resolver: CredentialResolver,
) -> dict[str, Any]:
    adapters: dict[str, Any] = {}
    for name, definition in definitions.items():
        if not definition.enabled or definition.adapter in {"local", "extension"}:
            continue
        if definition.adapter == "openai_compatible":
            adapters[name] = OpenAICompatibleProvider.from_definition(
                definition,
                credential_resolver=resolver,
            )
        elif definition.adapter == "gemini_native":
            adapters[name] = GeminiNativeProvider(definition, resolver)
        elif definition.adapter == "anthropic":
            adapters[name] = AnthropicProvider(definition, resolver)
        elif definition.adapter == "github_models":
            adapters[name] = GitHubModelsProvider.from_definition(
                definition,
                credential_resolver=resolver,
            )
        else:
            raise ValueError(
                f"Provider '{name}' uses unsupported adapter '{definition.adapter}'."
            )
    return adapters
