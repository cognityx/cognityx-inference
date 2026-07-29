"""Provider adapter registry, safe status, and cached model discovery."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Mapping, Protocol

from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.contracts import ModelCapabilities
from cognityx_inference.providers.models import (
    DiscoveredModels,
    ParameterPolicy,
    ProfileState,
    ProviderModelProfile,
    ProviderStatus,
)
from cognityx_inference.security import CredentialResolver


class ProviderAdapter(Protocol):
    name: str

    def credential_available(self) -> bool: ...

    def list_models(self, *, timeout_seconds: float = 20) -> tuple[str, ...]: ...


class ProviderRegistry:
    """Own configured identities and instantiated protocol adapters."""

    def __init__(
        self,
        definitions: Mapping[str, ProviderDefinition],
        adapters: Mapping[str, ProviderAdapter],
        resolver: CredentialResolver,
        *,
        local_ready: Callable[[], bool] | None = None,
    ) -> None:
        self.definitions = dict(definitions)
        self.adapters = dict(adapters)
        self.resolver = resolver
        self.local_ready = local_ready or (lambda: False)
        self._models: dict[str, tuple[float, DiscoveredModels]] = {}
        self._last_success: dict[str, str] = {}

    def list_statuses(self) -> tuple[ProviderStatus, ...]:
        return tuple(self.status(name) for name in sorted(self.definitions))

    def status(self, name: str) -> ProviderStatus:
        definition = self._definition(name)
        configured = self._configuration_status(definition)
        if name == "local":
            credential = "not_required"
            configured = "ready" if self.local_ready() else "configured"
        elif not definition.enabled:
            credential = "not_configured"
        else:
            adapter = self.adapters.get(name)
            try:
                credential = (
                    "configured"
                    if adapter is not None and adapter.credential_available()
                    else "not_configured"
                )
            except ValueError:
                credential = "invalid_configuration"
        return ProviderStatus(
            provider=name,
            adapter=definition.adapter,
            enabled=definition.enabled,
            credential_status=credential,
            configuration_status=configured,
            default_model=definition.default_model,
            default_profile=definition.default_profile,
            last_successful_test=self._last_success.get(name),
        )

    def discover_models(
        self,
        name: str,
        *,
        refresh: bool = False,
        timeout_seconds: float = 20,
    ) -> DiscoveredModels:
        definition = self._definition(name)
        if not definition.enabled:
            raise LookupError(f"Provider '{name}' is disabled.")
        now = time.monotonic()
        cached = self._models.get(name)
        if (
            not refresh
            and cached is not None
            and now - cached[0] < definition.discovery_cache_seconds
        ):
            return replace(cached[1], cached=True)
        adapter = self.adapters.get(name)
        if adapter is None:
            raise LookupError(f"Provider '{name}' does not support model discovery.")
        result = DiscoveredModels.now(
            name,
            tuple(adapter.list_models(timeout_seconds=timeout_seconds)),
        )
        self._models[name] = (now, result)
        return result

    def profile(self, provider: str, model: str) -> ProviderModelProfile:
        definition = self._definition(provider)
        configured = definition.model_profiles.get(model)
        if configured is not None:
            return configured
        adapter = self.adapters.get(provider)
        discovered = (
            provider in self._models and model in self._models[provider][1].models
        )
        return ProviderModelProfile(
            provider=provider,
            model_id=model,
            capabilities=getattr(adapter, "capabilities", ModelCapabilities()),
            parameter_policy=getattr(adapter, "parameter_policy", ParameterPolicy()),
            state=(
                ProfileState.TESTED
                if provider in self._last_success
                else ProfileState.DISCOVERED
                if discovered
                else ProfileState.UNVERIFIED
            ),
            verification_source=(
                "provider_diagnostic"
                if provider in self._last_success
                else "provider_discovery"
                if discovered
                else "configuration"
            ),
            verified_at=self._last_success.get(provider),
        )

    def mark_success(self, provider: str, when: str) -> None:
        self._last_success[provider] = when

    def _definition(self, name: str) -> ProviderDefinition:
        try:
            return self.definitions[name]
        except KeyError as exc:
            raise LookupError(f"Unknown provider: {name}") from exc

    @staticmethod
    def _configuration_status(definition: ProviderDefinition) -> str:
        if not definition.enabled:
            return "disabled"
        if definition.adapter == "local":
            return "configured"
        if not definition.base_url or not definition.api_key_env:
            return "invalid"
        if "{account_id}" in definition.base_url and not definition.account_id_env:
            return "invalid"
        return "configured"
