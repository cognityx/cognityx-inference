"""Minimal caller identity boundary for owner-scoped local jobs.

The resolver is intentionally small so a future Cognityx authentication and
authorization service can replace it without changing discovery endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol


class PrincipalResolver(Protocol):
    """Resolve a stable principal for an incoming HTTP request."""

    def resolve(self, authorization: str | None) -> str: ...


@dataclass(frozen=True, slots=True)
class EnvironmentPrincipalResolver:
    """Use configured bearer keys, or one explicit local-development owner."""

    key_to_principal: dict[str, str]
    local_principal: str = "local"

    @classmethod
    def from_environment(cls) -> "EnvironmentPrincipalResolver":
        raw = os.environ.get("COGNITYX_INFERENCE_API_KEYS_JSON", "")
        try:
            configured = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "COGNITYX_INFERENCE_API_KEYS_JSON must be a JSON object."
            ) from exc
        if not isinstance(configured, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in configured.items()
        ):
            raise RuntimeError(
                "COGNITYX_INFERENCE_API_KEYS_JSON must map bearer keys to principal IDs."
            )
        return cls(
            key_to_principal=configured,
            local_principal=os.environ.get("COGNITYX_INFERENCE_LOCAL_PRINCIPAL", "local"),
        )

    def resolve(self, authorization: str | None) -> str:
        if not self.key_to_principal:
            return self.local_principal
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise PermissionError("A bearer API key is required.")
        try:
            return self.key_to_principal[authorization[len(prefix) :]]
        except KeyError as exc:
            raise PermissionError("The bearer API key is not recognized.") from exc
