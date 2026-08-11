"""Run the Cognityx inference HTTP service."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from cognityx_jobs import JobRepository

from cognityx_inference.adapters import AdapterRepository
from cognityx_inference.api import create_app
from cognityx_inference.backends import TransformersBackend, VLLMBackend
from cognityx_inference.chat import (
    ChatContextError,
    ChatModelNotLoadedError,
    ChatSettings,
    CognityxChatSession,
)
from cognityx_inference.client import CognityxInferenceClient, InferenceAPIError
from cognityx_inference.configuration import InferenceConfiguration
from cognityx_inference.discovery import BoundaryDiscoveryCoordinator, DiscoveryConfig
from cognityx_inference.environment import configure_huggingface_cache
from cognityx_inference.errors import ProviderRequestError
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.manager import InferenceManager
from cognityx_inference.manager_api import create_manager_app
from cognityx_inference.presentation import render
from cognityx_inference.providers import (
    XAIProvider,
)
from cognityx_inference.providers.diagnostics import test_provider
from cognityx_inference.providers.factory import build_provider_adapters
from cognityx_inference.providers.registry import ProviderRegistry
from cognityx_inference.research import (
    EvaluationSetRepository,
    InferencePairRunner,
    ResearchPublisher,
)
from cognityx_inference.service import InferenceService
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    CertifiedProfileRepository,
    ChatSessionRepository,
    InferenceArtifactRepository,
    ManagerStateRepository,
)
from cognityx_inference.telemetry import ResourceMonitor, read_windows_bridge
from cognityx_inference.tracking import build_tracker
from cognityx_inference.vllm_runtime import VLLMRuntimeError, ensure_vllm_runtime
from llm_benchmark.reporting import get_system_metadata

DEFAULT_BASE_URL = os.environ.get("COGNITYX_INFERENCE_URL") or "http://127.0.0.1:8000"
DEFAULT_MANAGER_URL = (
    os.environ.get("COGNITYX_INFERENCE_MANAGER_URL") or "http://127.0.0.1:8000"
)
PRIMARY_PROVIDERS = (
    "local",
    "openai",
    "groq",
    "gemini",
    "cerebras",
    "openrouter",
    "github_models",
    "cloudflare",
    "anthropic",
)


def _add_output_format(
    parser: argparse.ArgumentParser, *, default: str = "detail"
) -> None:
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("table", "detail", "json"),
        default=default,
        help="Terminal presentation format; use json for machine-readable output.",
    )


def _add_thinking_control(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_const",
        const="enabled",
        help="Enable model-native thinking when the selected model supports it.",
    )
    group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_const",
        const="disabled",
        help="Disable model-native thinking (the default).",
    )
    parser.set_defaults(thinking="disabled")


def build_service(
    configuration: InferenceConfiguration | None = None,
) -> InferenceService:
    """Build the default service without loading a model."""
    configure_huggingface_cache()
    selected = configuration or InferenceConfiguration.load()
    credential_resolver = selected.credential_resolver()
    providers = build_provider_adapters(selected.providers, credential_resolver)
    if key := os.environ.get("XAI_API_KEY"):
        providers["xai"] = XAIProvider(key)
    models = ModelManager(
        {
            "transformers": TransformersBackend,
            "vllm": VLLMBackend,
        }
    )
    provider_registry = ProviderRegistry(
        selected.providers,
        providers,
        credential_resolver,
        local_ready=lambda: any(
            status.state.value == "ready" for status in models.statuses()
        ),
    )
    from cognityx_storage import StorageClient, StorageRuntime

    storage = StorageClient().for_shared_data()
    storage_runtime = StorageRuntime.load()
    profiles = CertifiedProfileRepository(storage)
    inventory = get_system_metadata
    discovery = BoundaryDiscoveryCoordinator(
        models,
        profiles,
        BoundaryArtifactRepository(storage),
        inventory,
        config=DiscoveryConfig.from_toml(
            Path(__file__).resolve().parents[2]
            / "examples"
            / "boundary"
            / "config.toml"
        ),
        jobs=JobRepository(
            os.environ.get("COGNITYX_JOBS_DATABASE", "cognityx_jobs.sqlite3")
        ),
    )
    service = InferenceService(
        models,
        providers,
        InferenceArtifactRepository(storage),
        profiles,
        inventory,
        discovery,
        provider_registry,
        AdapterRepository(storage_runtime),
    )
    service.research_runner = InferencePairRunner(
        service,
        EvaluationSetRepository(storage_runtime),
        ResearchPublisher(storage_runtime.for_role("artifact")),
        build_tracker(selected.tracking),
    )
    return service


def build_manager(
    configuration: InferenceConfiguration,
) -> InferenceManager:
    from cognityx_storage import StorageClient

    storage = StorageClient().for_shared_data()
    return InferenceManager(
        configuration.server_profiles,
        configuration.manager,
        jobs=JobRepository(
            os.environ.get("COGNITYX_JOBS_DATABASE", "cognityx_jobs.sqlite3")
        ),
        state_repository=ManagerStateRepository(storage),
    )


def _print_provider_rows(rows: list[dict[str, Any]], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    headers = (
        "provider",
        "adapter",
        "enabled",
        "credential_status",
        "configuration_status",
        "default_model",
        "default_profile",
        "last_successful_test",
    )
    widths = {
        header: max(
            len(header),
            *(len(str(row.get(header) or "-")) for row in rows),
        )
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(
            "  ".join(
                str(row.get(header) or "-").ljust(widths[header]) for header in headers
            )
        )


def _provider_setup(configuration: InferenceConfiguration, args: Any) -> None:
    selected = [args.provider] if args.provider else list(PRIMARY_PROVIDERS[1:])
    requirements = {
        name: {
            "api_key_env": configuration.providers[name].api_key_env,
            "account_id_env": configuration.providers[name].account_id_env,
        }
        for name in selected
        if name in configuration.providers
    }
    target = Path(
        configuration.secrets_file
        or os.environ.get("COGNITYX_SECRETS_FILE")
        or ".cognityx/secrets.json"
    ).expanduser()
    result: dict[str, Any] = {
        "providers": requirements,
        "secrets_file": str(target),
        "created": False,
        "instructions": (
            "Edit the secrets file manually or export the listed variables. "
            "Do not place credentials in command arguments."
        ),
    }
    if args.create_template:
        if not args.yes:
            result["reason"] = "Use --yes to confirm empty template creation."
            print(json.dumps(result, indent=2, sort_keys=True))
            raise SystemExit(2)
        if target.exists():
            result["reason"] = "Target already exists; it was not overwritten."
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            placeholders: dict[str, str] = {}
            for values in requirements.values():
                for name in values.values():
                    if name:
                        placeholders[str(name)] = ""
            target.write_text(
                json.dumps(placeholders, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                target.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            result["created"] = True
    print(json.dumps(result, indent=2, sort_keys=True))


def _test_local_provider(
    configuration: InferenceConfiguration,
    args: Any,
) -> dict[str, Any]:
    profile = next(iter(configuration.server_profiles.values()), None)
    if profile is None:
        return {
            "provider": "local",
            "status": "skipped",
            "reason": "not_configured",
        }
    manager_client = CognityxInferenceClient(
        args.base_url,
        manager_url=args.manager_url,
        timeout_seconds=args.timeout,
    )
    client = manager_client
    try:
        if args.start_local:
            managed = CognityxInferenceClient(
                args.base_url,
                backend="local",
                profile=profile.name,
                auto_start=True,
                manager_url=args.manager_url,
                startup_timeout_seconds=max(args.timeout, 600),
            )
            worker_url = managed.ensure_server_ready()
            client = CognityxInferenceClient(
                worker_url,
                timeout_seconds=args.timeout,
            )
        statuses = client.model_status()
        if not any(item.get("state") == "ready" for item in statuses):
            return {
                "provider": "local",
                "model": args.model or profile.model,
                "status": "skipped",
                "reason": "worker_not_ready",
            }
        model = args.model or profile.model
        common = dict(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            provider="local",
            backend=profile.backend,
            profile=profile.load_profile,
            max_output_tokens=8,
            load_policy="require_loaded",
        )
        plain = client.chat(**common)
        chunks = list(client.stream_chat(**common))
        normalized = plain.get("cognityx") or {}
        result = {
            "provider": "local",
            "model": model,
            "status": "passed",
            "non_streaming": ("passed" if normalized.get("content") else "failed"),
            "streaming": "passed" if chunks else "failed",
            "token_budget": (
                "passed" if normalized.get("token_budget") is not None else "failed"
            ),
            "usage": normalized.get("usage"),
        }
        if "failed" in {
            result["non_streaming"],
            result["streaming"],
            result["token_budget"],
        }:
            result["status"] = "failed"
        return result
    except (InferenceAPIError, RuntimeError, OSError, ValueError):
        return {
            "provider": "local",
            "model": args.model or profile.model,
            "status": "failed",
            "error_category": "provider_unavailable",
        }
    finally:
        if args.stop_local:
            try:
                manager_client.server_stop()
            except (InferenceAPIError, RuntimeError, OSError):
                pass


def _safe_provider_error(exc: Exception) -> str:
    if isinstance(exc, ProviderRequestError):
        return exc.category
    if isinstance(exc, LookupError):
        return "not_configured"
    return "unknown_provider_error"


def main(argv: list[str] | None = None) -> None:
    original_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", type=Path)

    manager = subparsers.add_parser("manager")
    manager_commands = manager.add_subparsers(dest="manager_command", required=True)
    manager_serve = manager_commands.add_parser("serve")
    manager_serve.add_argument("--config", type=Path)
    manager_serve.add_argument("--host")
    manager_serve.add_argument("--port", type=int)

    server = subparsers.add_parser("server")
    server_commands = server.add_subparsers(dest="server_command", required=True)
    server_start = server_commands.add_parser("start")
    server_start.add_argument("--manager-url", default=DEFAULT_MANAGER_URL)
    server_start.add_argument("--profile", required=True)
    server_stop = server_commands.add_parser("stop")
    server_stop.add_argument("--manager-url", default=DEFAULT_MANAGER_URL)
    server_status = server_commands.add_parser("status")
    server_status.add_argument("--manager-url", default=DEFAULT_MANAGER_URL)
    server_watch = server_commands.add_parser("watch")
    server_watch.add_argument("--manager-url", default=DEFAULT_MANAGER_URL)
    server_watch.add_argument("--after", type=int, default=0)

    providers_parser = subparsers.add_parser("providers")
    provider_commands = providers_parser.add_subparsers(
        dest="providers_command", required=True
    )
    for command in ("list", "status"):
        selected = provider_commands.add_parser(command)
        selected.add_argument("--config", type=Path)
        selected.add_argument("--json", action="store_true")
    provider_models = provider_commands.add_parser("models")
    provider_models.add_argument("--config", type=Path)
    provider_models.add_argument("--provider", required=True)
    provider_models.add_argument("--refresh", action="store_true")
    provider_models.add_argument("--timeout", type=float, default=20)
    provider_models.add_argument("--json", action="store_true")
    provider_capabilities = provider_commands.add_parser("capabilities")
    provider_capabilities.add_argument("--config", type=Path)
    provider_capabilities.add_argument("--provider", required=True)
    provider_capabilities.add_argument("--model", required=True)
    provider_capabilities.add_argument("--json", action="store_true")
    provider_test = provider_commands.add_parser("test")
    provider_test.add_argument("--config", type=Path)
    selection = provider_test.add_mutually_exclusive_group(required=True)
    selection.add_argument("--provider", choices=PRIMARY_PROVIDERS)
    selection.add_argument("--all", action="store_true")
    provider_test.add_argument("--model")
    provider_test.add_argument("--stream", action="store_true")
    provider_test.add_argument("--structured-output", action="store_true")
    provider_test.add_argument("--timeout", type=float, default=30)
    provider_test.add_argument("--json", action="store_true")
    provider_test.add_argument("--verbose-safe", action="store_true")
    provider_test.add_argument("--start-local", action="store_true")
    provider_test.add_argument("--stop-local", action="store_true")
    provider_test.add_argument("--base-url", default=DEFAULT_BASE_URL)
    provider_test.add_argument("--manager-url", default=DEFAULT_MANAGER_URL)
    provider_setup = provider_commands.add_parser("setup")
    provider_setup.add_argument("--config", type=Path)
    provider_setup.add_argument("--provider", choices=PRIMARY_PROVIDERS[1:])
    provider_setup.add_argument("--create-template", action="store_true")
    provider_setup.add_argument("--yes", action="store_true")

    model = subparsers.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    load = model_commands.add_parser("load")
    load.add_argument("--base-url", default=DEFAULT_BASE_URL)
    load.add_argument("--model", required=True)
    load.add_argument("--backend", default="vllm")
    load.add_argument("--profile", default="bf16")
    load.add_argument(
        "--discovery-policy",
        choices=("ask", "auto", "require_existing"),
        default="ask",
    )
    _add_output_format(load)
    status = model_commands.add_parser("status")
    status.add_argument("--base-url", default=DEFAULT_BASE_URL)
    _add_output_format(status)
    unload = model_commands.add_parser("unload")
    unload.add_argument("--base-url", default=DEFAULT_BASE_URL)
    unload.add_argument("--model", required=True)
    _add_output_format(unload)
    unload_all = model_commands.add_parser("unload-all")
    unload_all.add_argument("--base-url", default=DEFAULT_BASE_URL)
    _add_output_format(unload_all)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--base-url", default=DEFAULT_BASE_URL)
    infer.add_argument("--model", required=True)
    infer.add_argument("--model-revision")
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--backend", default="vllm")
    infer.add_argument("--profile", default="bf16")
    infer.add_argument(
        "--max-output-tokens", "--max-tokens", dest="max_output_tokens", type=int
    )
    infer.add_argument("--required-context-length", type=int)
    infer.add_argument("--adapter-manifest")
    infer.add_argument(
        "--adapter-purpose",
        choices=("evaluation",),
        default="evaluation",
    )
    infer.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Wait for completion and print the full JSON response.",
    )
    _add_thinking_control(infer)
    chat = subparsers.add_parser(
        "chat", help="Interactive conversation-aware local chat."
    )
    chat.add_argument("--base-url", default=DEFAULT_BASE_URL)
    chat.add_argument("--model")
    chat.add_argument("--backend", default="vllm")
    chat.add_argument("--profile", default="bf16")
    chat.add_argument("--system-prompt")
    chat.add_argument("--temperature", type=float, default=0.6)
    chat.add_argument("--top-p", type=float)
    chat.add_argument("--top-k", type=int)
    chat.add_argument("--min-p", type=float)
    chat.add_argument(
        "--max-output-tokens",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=512,
    )
    chat.add_argument("--stop", action="append", default=[])
    chat.add_argument("--seed", type=int)
    chat.add_argument("--log-probabilities", action="store_true")
    chat.add_argument("--top-log-probabilities", type=int)
    chat.add_argument("--reasoning", action="store_true")
    _add_thinking_control(chat)
    chat.add_argument("--timeout", type=float)
    chat.add_argument("--first-token-timeout", type=float)
    chat.add_argument("--no-token-progress-timeout", type=float)
    chat.add_argument("--autosave", action="store_true")
    chat.add_argument(
        "--owner-id", default=os.environ.get("COGNITYX_INFERENCE_USER", "local")
    )
    chat.add_argument("--telemetry-interval", type=float, default=0.25)
    chat.add_argument("--windows-bridge-path")
    chat.add_argument("--windows-bridge-max-age", type=float, default=5)
    _add_output_format(infer, default="json")
    infer.set_defaults(stream=True)
    infer.add_argument(
        "--discovery-policy",
        choices=("ask", "auto", "require_existing"),
        default="ask",
    )
    research = subparsers.add_parser(
        "research", help="Execute immutable research-grade inference runs."
    )
    research_commands = research.add_subparsers(dest="research_command", required=True)
    pair = research_commands.add_parser("pair")
    pair.add_argument("--base-url", default=DEFAULT_BASE_URL)
    pair.add_argument("--evaluation-manifest", required=True)
    pair.add_argument("--adapter-manifest", required=True)
    pair.add_argument("--model", required=True)
    pair.add_argument("--model-revision")
    pair.add_argument("--backend", default="vllm")
    pair.add_argument("--profile", default="bf16")
    pair.add_argument("--experiment-id", required=True)
    pair.add_argument("--comparison-id")
    pair.add_argument("--arm-id")
    pair.add_argument("--parent-run-id")
    pair.add_argument("--research-package-id")
    pair.add_argument("--training-variant-id")
    pair.add_argument("--training-run-id")
    pair.add_argument("--seed", type=int, default=0)
    pair.add_argument("--temperature", type=float, default=0.0)
    pair.add_argument("--top-p", type=float, default=1.0)
    pair.add_argument("--top-k", type=int)
    pair.add_argument("--max-output-tokens", type=int, default=512)
    _add_thinking_control(pair)
    pair.add_argument("--stop", action="append", default=[])
    pair.add_argument("--required-context-length", type=int)
    _add_output_format(pair, default="json")
    discovery = subparsers.add_parser("discovery")
    discovery_commands = discovery.add_subparsers(
        dest="discovery_command", required=True
    )
    start = discovery_commands.add_parser("start")
    start.add_argument("--base-url", default=DEFAULT_BASE_URL)
    start.add_argument("--model", required=True)
    start.add_argument("--backend", default="vllm")
    start.add_argument("--profile", default="bf16")
    _add_output_format(start)
    discovery_status = discovery_commands.add_parser("status")
    discovery_status.add_argument("--base-url", default=DEFAULT_BASE_URL)
    discovery_status.add_argument(
        "--all",
        action="store_true",
        help="Include completed, failed, and cancelled jobs.",
    )
    _add_output_format(discovery_status)
    watch = discovery_commands.add_parser("watch")
    watch.add_argument("--base-url", default=DEFAULT_BASE_URL)
    watch.add_argument("job_id")
    _add_output_format(watch)
    cancel = discovery_commands.add_parser("cancel")
    cancel.add_argument("--base-url", default=DEFAULT_BASE_URL)
    cancel.add_argument("job_id")
    _add_output_format(cancel)
    certified = subparsers.add_parser("certified-profiles")
    certified_commands = certified.add_subparsers(
        dest="certified_command", required=True
    )
    certified_list = certified_commands.add_parser("list")
    certified_list.add_argument("--base-url", default=DEFAULT_BASE_URL)
    certified_list.add_argument("--model")
    certified_list.add_argument("--backend")
    certified_list.add_argument("--profile")
    certified_list.add_argument("--kv-cache-precision")
    _add_output_format(certified_list)
    certified_show = certified_commands.add_parser("show")
    certified_show.add_argument("--base-url", default=DEFAULT_BASE_URL)
    certified_show.add_argument("profile_id")
    _add_output_format(certified_show)
    args = parser.parse_args(original_argv)
    if args.command == "manager":
        configuration = InferenceConfiguration.load(args.config)
        host = args.host or configuration.manager.host
        port = args.port or configuration.manager.port
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit(
                "Install API dependencies with: uv sync --extra api"
            ) from exc
        uvicorn.run(
            create_manager_app(build_manager(configuration)),
            host=host,
            port=port,
        )
        return
    if args.command == "server":
        client = CognityxInferenceClient(DEFAULT_BASE_URL, manager_url=args.manager_url)
        try:
            if args.server_command == "start":
                value = client.server_start(args.profile)
            elif args.server_command == "stop":
                value = client.server_stop()
            elif args.server_command == "status":
                value = client.server_status()
            else:
                for event in client.stream_server_events(after=args.after):
                    print(json.dumps(event, sort_keys=True), flush=True)
                return
        except (InferenceAPIError, RuntimeError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from None
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    if args.command == "providers":
        configuration = InferenceConfiguration.load(args.config)
        if args.providers_command == "setup":
            _provider_setup(configuration, args)
            return
        credential_resolver = configuration.credential_resolver()
        adapters = build_provider_adapters(configuration.providers, credential_resolver)
        registry = ProviderRegistry(
            configuration.providers,
            adapters,
            credential_resolver,
        )
        if args.providers_command in {"list", "status"}:
            rows = [item.to_dict() for item in registry.list_statuses()]
            _print_provider_rows(rows, as_json=args.json)
            return
        if args.providers_command == "models":
            try:
                value = registry.discover_models(
                    args.provider,
                    refresh=args.refresh,
                    timeout_seconds=args.timeout,
                ).to_dict()
            except (LookupError, RuntimeError) as exc:
                value = {
                    "provider": args.provider,
                    "status": "failed",
                    "error_category": _safe_provider_error(exc),
                }
                print(json.dumps(value, indent=2, sort_keys=True))
                raise SystemExit(1) from None
            print(json.dumps(value, indent=2, sort_keys=True))
            return
        if args.providers_command == "capabilities":
            try:
                profile = registry.profile(args.provider, args.model)
            except LookupError as exc:
                print(
                    json.dumps(
                        {
                            "provider": args.provider,
                            "model": args.model,
                            "status": "not_configured",
                            "error": str(exc),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                raise SystemExit(1) from None
            print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
            return
        names = PRIMARY_PROVIDERS if args.all else (args.provider,)
        results = []
        for name in names:
            if name == "local":
                results.append(_test_local_provider(configuration, args))
                continue
            definition = configuration.providers.get(str(name))
            if definition is None:
                results.append(
                    {
                        "provider": name,
                        "model": None,
                        "status": "failed",
                        "error_category": "provider_not_configured",
                    }
                )
            else:
                results.append(
                    test_provider(
                        definition,
                        credential_resolver=credential_resolver,
                        model=args.model,
                        structured_output=args.structured_output,
                        timeout_seconds=args.timeout,
                    ).to_dict()
                )
        print(json.dumps(results, indent=2, sort_keys=True))
        if any(result.get("status") == "failed" for result in results):
            raise SystemExit(1)
        return
    if args.command not in {None, "serve"}:
        client = CognityxInferenceClient(
            args.base_url,
            discovery_policy=getattr(args, "discovery_policy", "require_existing"),
            on_discovery_started=lambda event: print(
                render(
                    {
                        **event,
                        "watch": (
                            f"cognityx-inference discovery watch --base-url "
                            f"{args.base_url} {event['job_id']}"
                        ),
                        "cancel": (
                            f"cognityx-inference discovery cancel --base-url "
                            f"{args.base_url} {event['job_id']}"
                        ),
                    },
                    kind="detail",
                    output_format=getattr(args, "output_format", "detail"),
                ),
                flush=True,
            ),
            on_discovery_event=lambda event: print(
                render(
                    event,
                    kind="discovery_event",
                    output_format=getattr(args, "output_format", "detail"),
                ),
                flush=True,
            ),
        )
        try:
            if args.command == "model":
                if args.model_command == "load":
                    value = client.load_model(
                        args.model,
                        args.backend,
                        args.profile,
                        discovery_policy=args.discovery_policy,
                    )
                elif args.model_command == "status":
                    value = client.model_status()
                elif args.model_command == "unload":
                    value = client.unload_model(args.model, "vllm")
                else:
                    value = client.unload_all()
            elif args.command == "chat":
                _run_chat(args, client)
                return
            elif args.command == "research":
                value = client.run_research_pair(
                    {
                        "evaluation_manifest_uri": args.evaluation_manifest,
                        "adapter_manifest_uri": args.adapter_manifest,
                        "model": args.model,
                        "model_revision": args.model_revision,
                        "backend": args.backend,
                        "profile": args.profile,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_output_tokens": args.max_output_tokens,
                        "thinking": args.thinking,
                        "stop": args.stop,
                        "required_context_length": args.required_context_length,
                        "research_context": {
                            "experiment_id": args.experiment_id,
                            "comparison_id": args.comparison_id,
                            "arm_id": args.arm_id,
                            "seed": args.seed,
                            "parent_run_id": args.parent_run_id,
                            "research_package_id": args.research_package_id,
                            "training_variant_id": args.training_variant_id,
                            "training_run_id": args.training_run_id,
                        },
                    }
                )
            elif args.command == "certified-profiles":
                if args.certified_command == "list":
                    value = client.list_certified_profiles(
                        model=args.model,
                        backend=args.backend,
                        profile=args.profile,
                        kv_cache_precision=args.kv_cache_precision,
                    )
                else:
                    value = client.get_certified_profile(args.profile_id)
            else:
                if args.command == "discovery":
                    if args.discovery_command == "start":
                        value = client.start_discovery(
                            args.model, args.backend, args.profile
                        )
                    elif args.discovery_command == "cancel":
                        value = client.cancel_discovery(args.job_id)
                    elif args.discovery_command == "status":
                        value = client.list_discoveries(include_history=args.all)
                    else:
                        for event in client.stream_discovery(args.job_id):
                            print(
                                render(
                                    event,
                                    kind="discovery_event",
                                    output_format=args.output_format,
                                ),
                                flush=True,
                            )
                        return
                else:
                    parameters = {
                        "model": args.model,
                        "model_revision": args.model_revision,
                        "prompt": args.prompt,
                        "backend": args.backend,
                        "profile": args.profile,
                        "max_output_tokens": args.max_output_tokens,
                        "thinking": args.thinking,
                        "required_context_length": args.required_context_length,
                        "discovery_policy": args.discovery_policy,
                        "adapter_manifest_uri": args.adapter_manifest,
                        "adapter_purpose": (
                            args.adapter_purpose if args.adapter_manifest else None
                        ),
                    }
                    if args.stream:
                        for chunk in client.stream_chat(**parameters):
                            choices = chunk.get("choices") or ()
                            if not choices:
                                continue
                            content = (choices[0].get("delta") or {}).get("content")
                            if content:
                                print(content, end="", flush=True)
                        print(flush=True)
                        return
                    value = client.chat(**parameters)
        except InferenceAPIError as exc:
            print(json.dumps(exc.payload, indent=2), file=sys.stderr)
            raise SystemExit(2) from None
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from None
        kind = _output_kind(args)
        print(render(value, kind=kind, output_format=args.output_format))
        return
    configuration = InferenceConfiguration.load(getattr(args, "config", None))
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    try:
        ensure_vllm_runtime(original_argv)
    except VLLMRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install API dependencies with: uv sync --extra api") from exc
    uvicorn.run(create_app(build_service(configuration)), host=host, port=port)


def _run_chat(args: Any, client: CognityxInferenceClient) -> None:
    """Run the no-required-arguments interactive chat loop."""
    from cognityx_storage import StorageClient

    storage = StorageClient().for_user(args.owner_id)
    session = CognityxChatSession(
        client,
        ChatSessionRepository(storage),
        model=args.model,
        backend=args.backend,
        profile=args.profile,
        system_prompt=args.system_prompt,
        settings=ChatSettings(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            stop=args.stop,
            seed=args.seed,
            log_probabilities=args.log_probabilities,
            top_log_probabilities=args.top_log_probabilities,
            reasoning=args.reasoning,
            thinking=args.thinking,
            timeout_seconds=args.timeout,
            first_token_timeout_seconds=args.first_token_timeout,
            no_token_progress_timeout_seconds=args.no_token_progress_timeout,
        ),
        autosave=args.autosave,
    )
    monitor = ResourceMonitor(
        interval_seconds=args.telemetry_interval,
        windows_sampler=(
            lambda: read_windows_bridge(
                args.windows_bridge_path, max_age_seconds=args.windows_bridge_max_age
            )
        )
        if args.windows_bridge_path
        else None,
    )
    try:
        monitor.start()
    except RuntimeError as exc:
        monitor = None  # type: ignore[assignment]
        print(f"Telemetry unavailable: {exc}")
    if args.model:
        try:
            session.load_model(args.model, args.backend, args.profile)
        except (InferenceAPIError, RuntimeError) as exc:
            print(f"Model not ready: {exc}\nUse /load <model> [backend] [profile].")
    else:
        try:
            session.refresh_active_model()
        except (InferenceAPIError, RuntimeError):
            pass
    print(f"Interactive Cognityx chat connected to {client.base_url}.")
    print(
        "Use /status, /load, /settings, /history, /save, /load_chat, /autosave, /clear, or /quit."
    )
    try:
        while True:
            name = session.model or "no model"
            try:
                line = input(f"You [{name}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                line = "/quit"
            if not line:
                continue
            if line.startswith("/"):
                if _chat_command(line, session):
                    break
                continue
            try:
                print(f"{session.model or 'Assistant'}> ", end="", flush=True)
                final = session.send(
                    line, on_text=lambda text: print(text, end="", flush=True)
                )
                print()
                _print_chat_stats(
                    final, monitor.summary() if monitor else None, session.show
                )
            except ChatModelNotLoadedError as exc:
                print(f"\n{exc}\nUse /load <model> [backend] [profile].")
            except ChatContextError as exc:
                print(f"\nContext protection: {exc}")
            except (InferenceAPIError, RuntimeError, OSError) as exc:
                print(f"\n{_chat_inference_error(exc, session.client.base_url)}")
    finally:
        if session.dirty and not session.autosave:
            answer = input("Save this chat before exit? [y/N] ").strip().lower()
            if answer in {"y", "yes"}:
                print(f"Saved chat: {session.save()}")
        if monitor:
            monitor.stop()


def _chat_command(line: str, session: CognityxChatSession) -> bool:
    parts = line.split()
    command = parts[0].lower()
    if command == "/quit":
        return True
    if command == "/load":
        if len(parts) < 2:
            print("Usage: /load <model> [backend] [profile]")
            return False
        backend = parts[2] if len(parts) > 2 else "vllm"
        profile = parts[3] if len(parts) > 3 else "bf16"
        try:
            session.load_model(parts[1], backend, profile)
            print(
                f"Loaded {session.model}; certified context: {session.certified_context_length}"
            )
        except (InferenceAPIError, RuntimeError) as exc:
            print(f"Load failed: {exc}")
        return False
    if command == "/status":
        _chat_status(session)
        return False
    if command == "/save":
        print(f"Saved chat: {session.save(parts[1] if len(parts) > 1 else None)}")
        return False
    if command == "/load_chat":
        if len(parts) != 2:
            print("Usage: /load_chat <chat-id>")
            return False
        try:
            session.load_chat(parts[1])
            print(
                f"Restored chat {parts[1]} for {session.model}. Use /load if that model is not resident."
            )
        except (KeyError, RuntimeError) as exc:
            print(exc)
        return False
    if command == "/autosave":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            print(
                f"Autosave is {'on' if session.autosave else 'off'}. Usage: /autosave on|off"
            )
        else:
            session.autosave = parts[1].lower() == "on"
            print(f"Autosave {'enabled' if session.autosave else 'disabled'}.")
        return False
    if command == "/history":
        print(f"Summary: {session.summary or '-'}")
        for message in session.messages:
            print(f"{message['role']}: {message['content']}")
        return False
    if command == "/clear":
        session.summary = None
        session.messages.clear()
        session.dirty = True
        print("Conversation history cleared.")
        return False
    if command == "/settings":
        _chat_settings(parts[1:], session)
        return False
    print(
        "Unknown command. Use /status, /load, /settings, /history, /save, /load_chat, /autosave, /clear, or /quit."
    )
    return False


def _chat_settings(parts: list[str], session: CognityxChatSession) -> None:
    if not parts:
        print(
            json.dumps(
                {
                    "server_url": session.client.base_url,
                    "parameters": session.settings.request_parameters(),
                    "autosave": session.autosave,
                    "show": session.show,
                    "certified_context": session.certified_context_length,
                },
                indent=2,
            )
        )
        return
    if parts[0] == "url":
        if len(parts) == 1:
            print(f"Server URL: {session.client.base_url}")
            return
        if len(parts) != 2:
            print("Usage: /settings url <http://host:port>")
            return
        try:
            session.set_server_url(parts[1])
            print(
                f"Server URL changed to {session.client.base_url}. Use /status, then /load."
            )
        except ValueError as exc:
            print(f"Invalid server URL: {exc}")
        return
    if len(parts) == 3 and parts[0] == "show" and parts[1] in {"power", "cpu", "ram"}:
        session.show[parts[1]] = parts[2].lower() == "on"
        print(
            f"{parts[1]} display {'enabled' if session.show[parts[1]] else 'disabled'}."
        )
        return
    if len(parts) != 2:
        print(
            "Usage: /settings <url|temperature|top_p|top_k|min_p|max_tokens|log_probabilities|top_log_probabilities|reasoning|timeouts|show|perf> <value>"
        )
        return
    key, raw = parts[0], parts[1]
    if key == "perf":
        session.show["perf"] = raw.lower() == "on"
        print(
            f"Performance display {'enabled' if session.show['perf'] else 'disabled'}."
        )
        return
    if key == "stop":
        session.settings.stop = [] if raw.lower() in {"none", "clear"} else [raw]
        session.dirty = True
        print(f"stop = {session.settings.stop}")
        return
    if key.startswith("show_") and key[5:] in session.show:
        session.show[key[5:]] = raw.lower() == "on"
        return
    if not hasattr(session.settings, key):
        print(f"Unknown setting: {key}")
        return
    try:
        current = getattr(session.settings, key)
        if key in {"top_k", "max_tokens", "seed", "top_log_probabilities"}:
            value = int(raw)
        elif isinstance(current, bool):
            value: Any = raw.lower() in {"on", "true", "yes", "1"}
        elif isinstance(current, float) or key in {"temperature", "top_p", "min_p"}:
            value = float(raw)
        else:
            value = raw
        setattr(session.settings, key, value)
        session.dirty = True
        print(f"{key} = {value}")
    except ValueError as exc:
        print(f"Invalid value: {exc}")


def _chat_status(session: CognityxChatSession) -> None:
    diagnostic = session.client.diagnose_server(
        model=session.model,
        backend=session.backend,
        profile=session.profile,
    )
    print(f"Server: {diagnostic['base_url']}")
    if not diagnostic.get("reachable"):
        print(
            f"Status: unavailable ({diagnostic.get('detail', diagnostic.get('error'))})"
        )
        print(
            "Start this checkout with: uv run cognityx-inference serve --host 127.0.0.1 --port <port>"
        )
        return
    print("OpenAI /v1/models: available")
    lifecycle = diagnostic.get("lifecycle_endpoint")
    if lifecycle != "available":
        print(
            f"Cognityx lifecycle API: unavailable ({diagnostic.get('lifecycle_detail')})"
        )
        print(
            "This is not a current Cognityx inference server. Restart it from this checkout."
        )
        return
    loaded = diagnostic.get("loaded_models") or []
    if loaded:
        for item in loaded:
            identity = item.get("identity") or {}
            print(
                f"Loaded: {identity.get('model')} [{identity.get('backend')}] state={item.get('state')}"
            )
    else:
        print("Loaded: none; use /load <model> [backend] [profile].")
    token_state = diagnostic.get("token_count_endpoint")
    print(f"Token-count API: {token_state}")
    if token_state == "unavailable":
        print(f"Detail: {diagnostic.get('token_count_detail')}")
        print(
            "Chat needs this endpoint for safe history compression. Stop and restart the server from this checkout."
        )


def _chat_inference_error(exc: Exception, base_url: str) -> str:
    if isinstance(exc, InferenceAPIError) and exc.status == 404:
        return (
            f"Inference server {base_url} does not provide a required Cognityx chat endpoint.\n"
            "Run /status for the exact diagnostic. It is usually an older server process; "
            "stop it and restart `uv run cognityx-inference serve` from this checkout."
        )
    return f"Inference unavailable at {base_url}: {exc}\nRun /status, then use /load when the server is ready."


def _print_chat_stats(
    final: dict[str, Any], telemetry: dict[str, Any] | None, show: dict[str, bool]
) -> None:
    response = final.get("cognityx") or {}
    usage = response.get("usage") or {}
    quality = ((response.get("extensions") or {}).get("legacy_result") or {}).get(
        "quality_indicators"
    ) or {}
    print(
        f"Tokens: input={usage.get('prompt_tokens')} thinking={quality.get('thinking_tokens')} answer={quality.get('answer_tokens')} completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}"
    )
    if not show.get("perf", True) or not telemetry:
        return
    timings = response.get("timings") or {}
    print(
        f"Timing: generation={timings.get('token_generation_seconds')} s TTFT={timings.get('time_to_first_token_seconds')} s"
    )
    gpu = telemetry.get("gpu_usage") or {}
    if show.get("power", True):
        print(
            f"Power: avg={gpu.get('power_watts_average')} W peak={gpu.get('power_watts_peak')} W"
        )
    if show.get("ram", True):
        print(
            f"GPU memory: avg={gpu.get('dedicated_memory_used_bytes_average')} B peak={gpu.get('dedicated_memory_used_bytes_peak')} B | Host RAM avg={telemetry.get('host_ram_average_used_bytes')} B peak={telemetry.get('host_ram_peak_used_bytes')} B"
        )
    if show.get("cpu", True):
        print(
            f"CPU: host avg={telemetry.get('host_cpu_average_percent')}% peak={telemetry.get('host_cpu_peak_percent')}% | client avg={telemetry.get('process_cpu_average_percent')}% peak={telemetry.get('process_cpu_peak_percent')}%"
        )


def _output_kind(args: Any) -> str:
    if args.command == "model" and args.model_command == "status":
        return "model_status"
    if args.command == "discovery" and args.discovery_command == "status":
        return "discovery_list"
    if args.command == "certified-profiles":
        return (
            "certified_profile_list"
            if args.certified_command == "list"
            else "certified_profile_show"
        )
    return "detail"


if __name__ == "__main__":
    main()
