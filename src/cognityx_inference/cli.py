"""Run the Cognityx inference HTTP service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from cognityx_inference.api import create_app
from cognityx_inference.backends import TransformersBackend, VLLMBackend
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.client import CognityxInferenceClient, InferenceAPIError
from cognityx_inference.discovery import BoundaryDiscoveryCoordinator
from cognityx_inference.discovery import DiscoveryConfig
from cognityx_inference.environment import configure_huggingface_cache
from cognityx_inference.providers import OpenAIProvider, XAIProvider
from cognityx_inference.presentation import render
from cognityx_inference.service import InferenceService
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    CertifiedProfileRepository,
    InferenceArtifactRepository,
)
from cognityx_inference.vllm_runtime import VLLMRuntimeError, ensure_vllm_runtime
from cognityx_jobs import JobRepository
from llm_benchmark.reporting import get_system_metadata


DEFAULT_BASE_URL = os.environ.get("COGNITYX_INFERENCE_URL") or "http://127.0.0.1:8000"


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


def build_service() -> InferenceService:
    """Build the default service without loading a model."""
    configure_huggingface_cache()
    providers: dict[str, Any] = {}
    if key := os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = OpenAIProvider(key)
    if key := os.environ.get("XAI_API_KEY"):
        providers["xai"] = XAIProvider(key)
    models = ModelManager(
        {
            "transformers": TransformersBackend,
            "vllm": VLLMBackend,
        }
    )
    from cognityx_storage import StorageClient

    storage = StorageClient().for_shared_data()
    profiles = CertifiedProfileRepository(storage)
    inventory = get_system_metadata
    discovery = BoundaryDiscoveryCoordinator(
        models,
        profiles,
        BoundaryArtifactRepository(storage),
        inventory,
        config=DiscoveryConfig.from_toml(
            Path(__file__).resolve().parents[2] / "examples" / "boundary" / "config.toml"
        ),
        jobs=JobRepository(os.environ.get("COGNITYX_JOBS_DATABASE", "cognityx_jobs.sqlite3")),
    )
    return InferenceService(
        models,
        providers,
        InferenceArtifactRepository(storage),
        profiles,
        inventory,
        discovery,
    )


def main(argv: list[str] | None = None) -> None:
    original_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

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
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--backend", default="vllm")
    infer.add_argument("--profile", default="bf16")
    infer.add_argument("--max-tokens", type=int, default=256)
    infer.add_argument("--required-context-length", type=int)
    infer.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Wait for completion and print the full JSON response.",
    )
    _add_output_format(infer, default="json")
    infer.set_defaults(stream=True)
    infer.add_argument(
        "--discovery-policy",
        choices=("ask", "auto", "require_existing"),
        default="ask",
    )
    discovery = subparsers.add_parser("discovery")
    discovery_commands = discovery.add_subparsers(dest="discovery_command", required=True)
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
    certified_commands = certified.add_subparsers(dest="certified_command", required=True)
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
    if args.command not in {None, "serve"}:
        client = CognityxInferenceClient(
            args.base_url,
            discovery_policy=getattr(
                args, "discovery_policy", "require_existing"
            ),
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
                        value = client.start_discovery(args.model, args.backend, args.profile)
                    elif args.discovery_command == "cancel":
                        value = client.cancel_discovery(args.job_id)
                    elif args.discovery_command == "status":
                        value = client.list_discoveries(include_history=args.all)
                    else:
                        for event in client.stream_discovery(args.job_id):
                            print(render(event, kind="discovery_event", output_format=args.output_format), flush=True)
                        return
                else:
                    parameters = {
                        "model": args.model,
                        "prompt": args.prompt,
                        "backend": args.backend,
                        "profile": args.profile,
                        "max_tokens": args.max_tokens,
                        "required_context_length": args.required_context_length,
                        "discovery_policy": args.discovery_policy,
                    }
                    if args.stream:
                        for chunk in client.stream_chat(**parameters):
                            choices = chunk.get("choices") or ()
                            if not choices:
                                continue
                            content = (
                                choices[0].get("delta") or {}
                            ).get("content")
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
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    try:
        ensure_vllm_runtime(original_argv)
    except VLLMRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Install API dependencies with: uv sync --extra api"
        ) from exc
    uvicorn.run(create_app(build_service()), host=host, port=port)


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
