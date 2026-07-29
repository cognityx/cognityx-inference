"""Fail-closed, purpose-based provider routing for future DataForge callers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from cognityx_inference.providers.models import ProviderStatus


@dataclass(frozen=True, slots=True)
class RouteTarget:
    provider: str
    profile: str


@dataclass(frozen=True, slots=True)
class PurposeRoute:
    preferred: tuple[RouteTarget, ...] = ()
    allowed_providers: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    purposes: Mapping[str, PurposeRoute] = field(default_factory=dict)
    sensitive_classifications: frozenset[str] = frozenset(
        {"sensitive", "restricted", "confidential"}
    )

    def select(
        self,
        purpose: str,
        statuses: Sequence[ProviderStatus],
        *,
        data_classification: str = "internal",
        explicit_provider: str | None = None,
        permit_external_sensitive_data: bool = False,
    ) -> RouteTarget:
        route = self.purposes.get(purpose)
        if route is None:
            raise LookupError(
                f"No provider route is configured for purpose '{purpose}'."
            )
        available = {
            item.provider
            for item in statuses
            if item.enabled
            and item.configuration_status in {"configured", "ready"}
            and item.credential_status in {"configured", "not_required"}
        }
        sensitive = data_classification in self.sensitive_classifications
        allowed = route.allowed_providers
        if sensitive and not permit_external_sensitive_data:
            allowed = frozenset({"local"}) if allowed is None else allowed & {"local"}

        if explicit_provider is not None:
            if allowed is not None and explicit_provider not in allowed:
                raise PermissionError(
                    f"Provider '{explicit_provider}' is not allowed for this route."
                )
            if explicit_provider not in available:
                raise LookupError(
                    f"Explicit provider '{explicit_provider}' is unavailable."
                )
            for target in route.preferred:
                if target.provider == explicit_provider:
                    return target
            raise LookupError(
                f"Explicit provider '{explicit_provider}' has no configured profile."
            )

        for target in route.preferred:
            if target.provider in available and (
                allowed is None or target.provider in allowed
            ):
                return target
        if sensitive:
            raise PermissionError(
                "No permitted provider is available; sensitive data was not routed."
            )
        raise LookupError(f"No provider is available for purpose '{purpose}'.")
