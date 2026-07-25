from __future__ import annotations

import pytest

from cognityx_inference.auth import EnvironmentPrincipalResolver


def test_configured_bearer_key_resolves_to_its_owner(monkeypatch) -> None:
    monkeypatch.setenv(
        "COGNITYX_INFERENCE_API_KEYS_JSON", '{"alice-key": "alice"}'
    )
    resolver = EnvironmentPrincipalResolver.from_environment()

    assert resolver.resolve("Bearer alice-key") == "alice"
    with pytest.raises(PermissionError):
        resolver.resolve("Bearer bob-key")


def test_local_mode_uses_one_explicit_principal(monkeypatch) -> None:
    monkeypatch.delenv("COGNITYX_INFERENCE_API_KEYS_JSON", raising=False)
    monkeypatch.setenv("COGNITYX_INFERENCE_LOCAL_PRINCIPAL", "workstation-user")

    assert EnvironmentPrincipalResolver.from_environment().resolve(None) == "workstation-user"
