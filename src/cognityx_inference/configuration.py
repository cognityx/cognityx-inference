"""TOML configuration for manager profiles and commercial providers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.contracts import ModelCapabilities
from cognityx_inference.providers.models import (
    ParameterPolicy,
    ProfileState,
    ProviderAccountLimits,
    ProviderModelProfile,
)
from cognityx_inference.routing import (
    PurposeRoute,
    RouteTarget,
    RoutingPolicy,
)
from cognityx_inference.security import CredentialResolver
from cognityx_inference.tracking import TrackingConfiguration


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    name: str
    adapter: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_secret: str | None = None
    smoke_test_model: str | None = None
    enabled: bool = False
    default_model: str | None = None
    default_profile: str | None = None
    discovery_cache_seconds: float = 3600
    account_id_env: str | None = None
    account_id_secret: str | None = None
    api_version: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    model_capabilities: Mapping[str, CertifiedContextProfile] = field(
        default_factory=dict
    )
    model_profiles: Mapping[str, ProviderModelProfile] = field(default_factory=dict)


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
    routing: RoutingPolicy = field(default_factory=RoutingPolicy)
    tracking: TrackingConfiguration = field(default_factory=TrackingConfiguration)
    source: str | None = None

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        cwd: str | Path | None = None,
    ) -> "InferenceConfiguration":
        return resolve_inference_configuration(path, cwd=cwd).configuration

    def credential_resolver(self) -> CredentialResolver:
        """Build a lazy resolver without loading credential values."""
        return CredentialResolver(self.secrets_file)


def _from_document(
    document: Mapping[str, Any], selected: Path
) -> InferenceConfiguration:
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
        return InferenceConfiguration(
            manager=manager,
            server_profiles=profiles,
            providers=providers,
            secrets_file=_secrets_file(document, selected),
            routing=_routing(document.get("routing") or {}),
            tracking=TrackingConfiguration(**dict(document.get("tracking") or {})),
            source=str(selected),
        )


@dataclass(frozen=True, slots=True)
class InferenceConfigResolution:
    """Static Inference selection used by both execution and CLI inspection."""

    configuration: InferenceConfiguration
    selected_by: str
    path: Path | None
    file_sha256: str | None
    changed_keys: tuple[str, ...]
    overrides: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        configuration_values = asdict(self.configuration)
        effective = _safe_plain(configuration_values)
        effective.pop("source", None)
        effective["secrets_file"] = {
            "configured": self.configuration.secrets_file is not None,
            "path": self.configuration.secrets_file,
            "contents_read": False,
        }
        source = str(self.path) if self.path is not None else "built-in"
        override_sources = {
            str(item["key"]): str(item["source"])
            for item in self.overrides
        }
        configuration_values.pop("source", None)
        field_sources = {
            key: override_sources.get(
                key,
                source if key in self.changed_keys else "built-in",
            )
            for key in _flatten_keys(configuration_values)
        }
        return {
            "component": "inference",
            "configuration_kind": "persistent-component",
            "valid": True,
            "master_config": {
                "kind": "file" if self.path is not None else "built-in",
                "path": str(self.path) if self.path is not None else None,
                "selected_by": self.selected_by,
                "sha256": self.file_sha256,
            },
            "config_layers": (
                [{
                    "path": str(self.path),
                    "selected_by": self.selected_by,
                    "sha256": self.file_sha256,
                    "changed_keys": list(self.changed_keys),
                }]
                if self.path is not None
                else []
            ),
            "field_sources": field_sources,
            "overrides": [dict(item) for item in self.overrides],
            "effective": effective,
            "warnings": [],
            "errors": [],
        }


def resolve_inference_configuration(
    path: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
) -> InferenceConfigResolution:
    """Load local Inference settings and provenance without runtime setup."""
    selected, selected_by = _select_path_with_reason(path, cwd=cwd)
    if selected is None:
        secrets = os.environ.get("COGNITYX_SECRETS_FILE")
        configuration = InferenceConfiguration(
            providers=_default_providers(),
            secrets_file=(str(Path(secrets).expanduser().resolve()) if secrets else None),
        )
        overrides: tuple[Mapping[str, Any], ...] = (
            ({
                "key": "secrets_file",
                "source": "COGNITYX_SECRETS_FILE",
                "previous": None,
                "effective": configuration.secrets_file,
                "changed": True,
            },)
            if secrets
            else ()
        )
        return InferenceConfigResolution(
            configuration, "built-in", None, None, (), overrides
        )
    selected = selected.resolve()
    raw = selected.read_bytes()
    document = tomllib.loads(raw.decode("utf-8"))
    configuration = _from_document(document, selected)
    file_secret = _configured_secret_path(document, selected)
    environment_secret = os.environ.get("COGNITYX_SECRETS_FILE")
    overrides = ()
    if environment_secret:
        effective_secret = str(Path(environment_secret).expanduser().resolve())
        if effective_secret != file_secret:
            overrides = ({
                "key": "secrets_file",
                "source": "COGNITYX_SECRETS_FILE",
                "previous": file_secret,
                "effective": effective_secret,
                "changed": True,
            },)
    return InferenceConfigResolution(
        configuration=configuration,
        selected_by=selected_by,
        path=selected,
        file_sha256=sha256(raw).hexdigest(),
        changed_keys=_changed_file_keys(configuration, file_secret=file_secret),
        overrides=overrides,
    )


def _flatten_keys(value: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    keys: list[str] = []
    for name, item in value.items():
        dotted = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(item, Mapping):
            nested = _flatten_keys(item, dotted)
            keys.extend(nested or (dotted,))
        else:
            keys.append(dotted)
    return tuple(keys)


def _flatten_values(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, item in value.items():
        dotted = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(item, Mapping):
            nested = _flatten_values(item, dotted)
            if nested:
                values.update(nested)
            else:
                values[dotted] = {}
        else:
            values[dotted] = item
    return values


def _changed_file_keys(
    configuration: InferenceConfiguration,
    *,
    file_secret: str | None,
) -> tuple[str, ...]:
    baseline = asdict(InferenceConfiguration(providers=_default_providers()))
    selected = asdict(configuration)
    baseline.pop("source", None)
    selected.pop("source", None)
    selected["secrets_file"] = file_secret
    baseline_values = _flatten_values(baseline)
    selected_values = _flatten_values(selected)
    return tuple(sorted(
        key
        for key in baseline_values.keys() | selected_values.keys()
        if baseline_values.get(key) != selected_values.get(key)
    ))


def _safe_plain(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in ("password", "token", "credential", "api_key")):
        return "<redacted>" if value is not None else None
    if isinstance(value, Mapping):
        return {str(name): _safe_plain(item, str(name)) for name, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_safe_plain(item, key) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and "://" in value:
        return _redacted_uri(value)
    return value


def _redacted_uri(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = (
        host
        if parsed.username is not None or parsed.password is not None
        else parsed.netloc
    )
    query = urlencode([
        (
            name,
            "<redacted>"
            if any(
                marker in name.lower()
                for marker in ("password", "token", "credential", "api_key")
            )
            else item,
        )
        for name, item in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def _provider(name: str, value: Mapping[str, Any]) -> ProviderDefinition:
    preset = _default_providers().get(name)
    capabilities: dict[str, CertifiedContextProfile] = {}
    profiles: dict[str, ProviderModelProfile] = {}
    for model, raw in (value.get("models") or {}).items():
        context = raw.get("context") if isinstance(raw, Mapping) else None
        if not isinstance(context, Mapping):
            raise ValueError(f"providers.{name}.models.{model}.context is required")
        context_profile = CertifiedContextProfile(**dict(context))
        capabilities[str(model)] = context_profile
        capability_value = dict(raw.get("capabilities") or {})
        policy_value = dict(raw.get("parameter_policy") or {})
        limits_value = dict(raw.get("limits") or {})
        profiles[str(model)] = ProviderModelProfile(
            provider=name,
            model_id=str(model),
            context=context_profile,
            tokenizer=raw.get("tokenizer") or context_profile.tokenizer,
            capabilities=ModelCapabilities(**capability_value),
            parameter_policy=_parameter_policy(policy_value),
            limits=ProviderAccountLimits(**limits_value),
            state=ProfileState(str(raw.get("state", "unverified"))),
            verification_source=str(
                (raw.get("verification") or {}).get("source", "manual")
            ),
            verified_at=(raw.get("verification") or {}).get("verified_at"),
        )
    base_url = value.get("base_url", preset.base_url if preset else None)
    api_key_env = value.get("api_key_env", preset.api_key_env if preset else None)
    return ProviderDefinition(
        name=name,
        adapter=str(
            value.get("adapter", preset.adapter if preset else "openai_compatible")
        ),
        base_url=str(base_url).rstrip("/") if base_url else None,
        api_key_env=str(api_key_env) if api_key_env else None,
        api_key_secret=(
            str(value["api_key_secret"])
            if value.get("api_key_secret") is not None
            else (preset.api_key_secret if preset else None)
        ),
        smoke_test_model=value.get(
            "smoke_test_model", preset.smoke_test_model if preset else None
        ),
        enabled=bool(value.get("enabled", True)),
        default_model=value.get(
            "default_model",
            value.get(
                "smoke_test_model",
                preset.default_model if preset else None,
            ),
        ),
        default_profile=value.get(
            "default_profile", preset.default_profile if preset else None
        ),
        discovery_cache_seconds=float(
            value.get(
                "discovery_cache_seconds",
                preset.discovery_cache_seconds if preset else 3600,
            )
        ),
        account_id_env=value.get(
            "account_id_env", preset.account_id_env if preset else None
        ),
        account_id_secret=value.get(
            "account_id_secret", preset.account_id_secret if preset else None
        ),
        api_version=value.get("api_version", preset.api_version if preset else None),
        options=dict(value.get("options") or {}),
        model_capabilities=capabilities,
        model_profiles=profiles,
    )


def _default_providers() -> dict[str, ProviderDefinition]:
    return {
        "local": ProviderDefinition(
            name="local",
            adapter="local",
            enabled=True,
            default_profile="local-default",
        ),
        "openai": ProviderDefinition(
            name="openai",
            adapter="openai_compatible",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_key_secret="OPENAI_API_KEY",
            default_profile="openai-default",
            enabled=True,
            default_model="gpt-4.1-mini",
            smoke_test_model="gpt-4.1-mini",
        ),
        "groq": ProviderDefinition(
            name="groq",
            adapter="openai_compatible",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            api_key_secret="GROQ_API_KEY",
            default_profile="groq-default",
            enabled=True,
            default_model="llama-3.1-8b-instant",
            smoke_test_model="llama-3.1-8b-instant",
        ),
        "gemini": ProviderDefinition(
            name="gemini",
            adapter="gemini_native",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            api_key_secret="GEMINI_API_KEY",
            default_profile="gemini-default",
            enabled=True,
            default_model="models/gemini-flash-latest",
            smoke_test_model="models/gemini-flash-latest",
        ),
        "gemini_openai": ProviderDefinition(
            name="gemini_openai",
            adapter="openai_compatible",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key_env="GEMINI_API_KEY",
            api_key_secret="GEMINI_API_KEY",
            default_profile="gemini-openai-default",
        ),
        "cerebras": ProviderDefinition(
            name="cerebras",
            adapter="openai_compatible",
            base_url="https://api.cerebras.ai/v1",
            api_key_env="CEREBRAS_API_KEY",
            api_key_secret="CEREBRAS_API_KEY",
            default_profile="cerebras-default",
            enabled=True,
            default_model="gpt-oss-120b",
            smoke_test_model="gpt-oss-120b",
        ),
        "openrouter": ProviderDefinition(
            name="openrouter",
            adapter="openai_compatible",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            api_key_secret="OPENROUTER_API_KEY",
            default_profile="openrouter-default",
            enabled=True,
            default_model="openai/gpt-4.1-mini",
            smoke_test_model="openai/gpt-4.1-mini",
        ),
        "github_models": ProviderDefinition(
            name="github_models",
            adapter="github_models",
            base_url="https://models.github.ai/inference",
            api_key_env="GITHUB_MODELS_TOKEN",
            api_key_secret="GITHUB_MODELS_TOKEN",
            default_profile="github-models-default",
            enabled=True,
            default_model="openai/gpt-4.1-mini",
            smoke_test_model="openai/gpt-4.1-mini",
            options={"models_url": "https://models.github.ai/catalog/models"},
        ),
        "cloudflare": ProviderDefinition(
            name="cloudflare",
            adapter="openai_compatible",
            base_url=(
                "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            ),
            api_key_env="CLOUDFLARE_API_TOKEN",
            api_key_secret="CLOUDFLARE_API_TOKEN",
            account_id_env="CLOUDFLARE_ACCOUNT_ID",
            account_id_secret="CLOUDFLARE_ACCOUNT_ID",
            default_profile="cloudflare-default",
            enabled=True,
            default_model="@cf/meta/llama-3.1-8b-instruct",
            smoke_test_model="@cf/meta/llama-3.1-8b-instruct",
            options={"model_discovery": False},
        ),
        "anthropic": ProviderDefinition(
            name="anthropic",
            adapter="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            api_key_secret="ANTHROPIC_API_KEY",
            api_version="2023-06-01",
            default_profile="anthropic-default",
            enabled=True,
            default_model="claude-sonnet-4-20250514",
            smoke_test_model="claude-sonnet-4-20250514",
        ),
        "azure_openai": ProviderDefinition(
            name="azure_openai", adapter="extension", default_profile=None
        ),
        "aws_bedrock": ProviderDefinition(
            name="aws_bedrock", adapter="extension", default_profile=None
        ),
        "google_vertex": ProviderDefinition(
            name="google_vertex", adapter="extension", default_profile=None
        ),
    }


def _parameter_policy(value: Mapping[str, Any]) -> ParameterPolicy:
    return ParameterPolicy(
        supported=frozenset(str(item) for item in value.get("supported", ())),
        translated={
            str(name): str(target)
            for name, target in dict(value.get("translated") or {}).items()
        },
        ignored_when_unset=frozenset(
            str(item) for item in value.get("ignored_when_unset", ())
        ),
        rejected_when_supplied=frozenset(
            str(item) for item in value.get("rejected_when_supplied", ())
        ),
        defaults=dict(value.get("defaults") or {}),
    )


def _routing(value: Mapping[str, Any]) -> RoutingPolicy:
    purposes = {}
    for purpose, raw in value.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"routing.{purpose} must be a table")
        preferred = tuple(
            RouteTarget(
                provider=str(item["provider"]),
                profile=str(item["profile"]),
            )
            for item in raw.get("preferred", ())
        )
        allowed_raw = raw.get("allowed_providers")
        purposes[str(purpose)] = PurposeRoute(
            preferred=preferred,
            allowed_providers=(
                frozenset(str(item) for item in allowed_raw)
                if allowed_raw is not None
                else None
            ),
        )
    return RoutingPolicy(purposes=purposes)


def _select_path(path: str | Path | None, *, cwd: str | Path | None) -> Path | None:
    return _select_path_with_reason(path, cwd=cwd)[0]


def _select_path_with_reason(
    path: str | Path | None, *, cwd: str | Path | None
) -> tuple[Path | None, str]:
    if path is not None:
        selected = Path(path).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Inference configuration not found: {selected}")
        return selected, "explicit"
    environment = os.environ.get("COGNITYX_INFERENCE_CONFIG")
    if environment:
        selected = Path(environment).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Inference configuration not found: {selected}")
        return selected, "environment"
    project = Path(cwd or Path.cwd()) / ".cognityx" / "inference.toml"
    return (project, "project") if project.is_file() else (None, "built-in")


def _configured_secret_path(
    document: Mapping[str, Any], config_path: Path
) -> str | None:
    raw = document.get("secrets_file")
    if raw is None:
        return None
    selected = Path(str(raw)).expanduser()
    if not selected.is_absolute():
        selected = config_path.parent / selected
    return str(selected.resolve())


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
