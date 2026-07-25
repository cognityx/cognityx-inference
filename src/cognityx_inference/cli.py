"""Run the Cognityx inference HTTP service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from cognityx_inference.api import create_app
from cognityx_inference.backends import TransformersBackend, VLLMBackend
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.client import CognityxInferenceClient, InferenceAPIError
from cognityx_inference.discovery import BoundaryDiscoveryCoordinator
from cognityx_inference.environment import configure_huggingface_cache
from cognityx_inference.providers import OpenAIProvider, XAIProvider
from cognityx_inference.service import InferenceService
from cognityx_inference.storage import (
    BoundaryArtifactRepository,
    CertifiedProfileRepository,
    InferenceArtifactRepository,
)
from cognityx_inference.vllm_runtime import VLLMRuntimeError, ensure_vllm_runtime
from cognityx_jobs import JobRepository
from llm_benchmark.reporting import get_system_metadata


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
    load.add_argument("--base-url", default="http://127.0.0.1:8000")
    load.add_argument("--model", required=True)
    load.add_argument("--backend", default="vllm")
    load.add_argument("--profile", default="bf16")
    load.add_argument(
        "--discovery-policy",
        choices=("ask", "auto", "require_existing"),
        default="ask",
    )
    status = model_commands.add_parser("status")
    status.add_argument("--base-url", default="http://127.0.0.1:8000")
    unload = model_commands.add_parser("unload")
    unload.add_argument("--base-url", default="http://127.0.0.1:8000")
    unload.add_argument("--model", required=True)
    unload_all = model_commands.add_parser("unload-all")
    unload_all.add_argument("--base-url", default="http://127.0.0.1:8000")

    infer = subparsers.add_parser("infer")
    infer.add_argument("--base-url", default="http://127.0.0.1:8000")
    infer.add_argument("--model", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--backend", default="vllm")
    infer.add_argument("--profile", default="bf16")
    infer.add_argument("--max-tokens", type=int, default=256)
    infer.add_argument("--required-context-length", type=int)
    infer.add_argument(
        "--discovery-policy",
        choices=("ask", "auto", "require_existing"),
        default="ask",
    )
    args = parser.parse_args(original_argv)
    if args.command not in {None, "serve"}:
        client = CognityxInferenceClient(
            args.base_url,
            discovery_policy=getattr(
                args, "discovery_policy", "require_existing"
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
            else:
                value = client.chat(
                    model=args.model,
                    prompt=args.prompt,
                    backend=args.backend,
                    profile=args.profile,
                    max_tokens=args.max_tokens,
                    required_context_length=args.required_context_length,
                    discovery_policy=args.discovery_policy,
                )
        except InferenceAPIError as exc:
            print(json.dumps(exc.payload, indent=2), file=sys.stderr)
            raise SystemExit(2) from None
        print(json.dumps(value, indent=2))
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


if __name__ == "__main__":
    main()
