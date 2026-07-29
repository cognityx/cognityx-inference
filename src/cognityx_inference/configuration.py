"""TOML configuration for manager profiles and commercial providers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.security import CredentialResolver


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    name: str
    adapter: str
    base_url: str
    api_key_env: str
    api_key_secret: str | None = None
    smoke_test_model: str | None = None
    model_capabilities: Mapping[str, CertifiedContextProfile] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class LocalServerProfile:
    name: str
    model: str
    backend: str = "vllm"
    load_profile: str = "bf16"
    certified_profile_id: str | None = None
    host: str = "127.0.0.1"
    port: int = 8100
    runtime: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManagerConfiguration:
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout_seconds: float = 600
    stop_timeout_seconds: float = 30
    health_poll_seconds: float = 0.25


@dataclass(frozen=True, slots=True)
class InferenceConfiguration:
    manager: ManagerConfiguration = field(default_factory=ManagerConfiguration)
    server_profiles: Mapping[str, LocalServerProfile] = field(default_factory=dict)
    providers: Mapping[str, ProviderDefinition] = field(default_factory=dict)
    secrets_file: str | None = None
    source: str | None = None

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        cwd: str | Path | None = None,
    ) -> "InferenceConfiguration":
        selected = _select_path(path, cwd=cwd)
        if selected is None:
            return cls(
                providers=_default_providers(),
                secrets_file=os.environ.get("COGNITYX_SECRETS_FILE"),
            )
        with selected.open("rb") as source:
            document = tomllib.load(source)
        manager = ManagerConfiguration(**dict(document.get("manager") or {}))
        profiles = {
            name: LocalServerProfile(
                name=name,
                model=str(value["model"]),
                backend=str(value.get("backend", "vllm")),
                load_profile=str(value.get("load_profile", "bf16")),
                certified_profile_id=value.get("certified_profile_id"),
                host=str(value.get("host", "127.0.0.1")),
                port=int(value.get("port", 8100)),
                runtime=dict(value.get("runtime") or {}),
            )
            for name, value in (document.get("server_profiles") or {}).items()
        }
        providers = _default_providers()
        providers.update(
            {
                name: _provider(name, value)
                for name, value in (document.get("providers") or {}).items()
            }
        )
        return cls(
            manager=manager,
            server_profiles=profiles,
            providers=providers,
            secrets_file=_secrets_file(document, selected),
            source=str(selected),
        )

    def credential_resolver(self) -> CredentialResolver:
        """Build a lazy resolver without loading credential values."""
        return CredentialResolver(self.secrets_file)


def _provider(name: str, value: Mapping[str, Any]) -> ProviderDefinition:
    capabilities = {}
    for model, raw in (value.get("models") or {}).items():
        context = raw.get("context") if isinstance(raw, Mapping) else None
        if not isinstance(context, Mapping):
            raise ValueError(
                f"providers.{name}.models.{model}.context is required"
            )
        capabilities[str(model)] = CertifiedContextProfile(**dict(context))
    return ProviderDefinition(
        name=name,
        adapter=str(value.get("adapter", "openai_compatible")),
        base_url=str(value["base_url"]).rstrip("/"),
        api_key_env=str(value["api_key_env"]),
        api_key_secret=(
            str(value["api_key_secret"])
            if value.get("api_key_secret") is not None
            else None
        ),
        smoke_test_model=value.get("smoke_test_model"),
        model_capabilities=capabilities,
    )


def _default_providers() -> dict[str, ProviderDefinition]:
    return {
        "openai": ProviderDefinition(
            name="openai",
            adapter="openai_compatible",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key_secret="openai_api_key",
        ),
        "groq": ProviderDefinition(
            name="groq",
            adapter="openai_compatible",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            api_key_secret="groq_api_key",
        ),
    }


def _select_path(
    path: str | Path | None, *, cwd: str | Path | None
) -> Path | None:
    if path is not None:
        selected = Path(path).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Inference configuration not found: {selected}")
        return selected
    environment = os.environ.get("COGNITYX_INFERENCE_CONFIG")
    if environment:
        return _select_path(environment, cwd=cwd)
    project = Path(cwd or Path.cwd()) / ".cognityx" / "inference.toml"
    return project if project.is_file() else None


def _secrets_file(document: Mapping[str, Any], config_path: Path) -> str | None:
    configured = os.environ.get("COGNITYX_SECRETS_FILE")
    if configured is None:
        raw = document.get("secrets_file")
        configured = str(raw) if raw is not None else None
    if not configured:
        return None
    selected = Path(configured).expanduser()
    if not selected.is_absolute():
        selected = config_path.parent / selected
    return str(selected.resolve())
