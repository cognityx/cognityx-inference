"""OpenAI-compatible FastAPI application factory."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict
from typing import Any

from cognityx_inference.auth import EnvironmentPrincipalResolver, PrincipalResolver
from cognityx_inference.backends.legacy import discover_model_context_limit
from cognityx_inference.capabilities import (
    CertifiedContextLimitExceeded,
    HardwareDiscoveryRequired,
    ModelContextLimitExceeded,
)
from cognityx_inference.contracts import (
    DiscoveryPolicy,
    InferenceRequest,
    LoadPolicy,
)
from cognityx_inference.errors import (
    ContextWindowExceeded,
    InferenceContractError,
    ModelMetadataUnavailableError,
    ProviderCredentialMissing,
    ProviderRequestError,
    TokenCountingUnavailable,
    UnsupportedProviderParameter,
)
from cognityx_inference.lifecycle import ModelIdentity
from cognityx_inference.providers.diagnostics import test_provider
from cognityx_inference.research import InferencePairRequest
from cognityx_inference.service import InferenceService, ModelNotLoadedError


def _status(status: Any) -> dict[str, Any]:
    value = asdict(status)
    value["state"] = status.state.value
    value["identity"]["runtime"] = dict(status.identity.runtime)
    return value


def _model_metadata_http_exception(exc: ModelMetadataUnavailableError) -> Any:
    """Normalize missing local model metadata into a useful API error."""
    from fastapi import HTTPException

    return HTTPException(
        status_code=404,
        detail={
            "error": "model_metadata_unavailable",
            "message": str(exc),
            "hint": "Check the model identifier and Hugging Face cache.",
        },
    )


def _request_from_chat_payload(payload: dict[str, Any]) -> InferenceRequest:
    """Normalize the OpenAI-compatible HTTP body into the public contract."""
    extension = payload.get("cognityx") or {}
    timeouts = extension.get("timeouts") or {}
    return InferenceRequest(
        model=str(payload["model"]),
        model_revision=extension.get("model_revision"),
        messages=tuple(payload.get("messages") or ()),
        tools=tuple(payload.get("tools") or ()),
        tool_choice=payload.get("tool_choice"),
        response_format=payload.get("response_format"),
        client_type=str(extension.get("client_type", "openai")),
        provider=str(extension.get("provider", "local")),
        backend=str(extension.get("backend", "vllm")),
        profile=str(extension.get("profile", "bf16")),
        adapter_manifest_uri=extension.get("adapter_manifest_uri"),
        adapter_purpose=extension.get("adapter_purpose"),
        load_policy=(
            LoadPolicy(extension.get("load_policy", "auto"))
            if extension
            else LoadPolicy.REQUIRE_LOADED
        ),
        discovery_policy=DiscoveryPolicy(
            extension.get("discovery_policy", "require_existing")
        ),
        required_context_length=extension.get("required_context_length"),
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=extension.get("top_k"),
        min_p=extension.get("min_p"),
        max_output_tokens=(
            extension.get("max_output_tokens")
            if extension.get("max_output_tokens") is not None
            else payload.get("max_completion_tokens", payload.get("max_tokens"))
        ),
        stop=tuple(
            [payload["stop"]]
            if isinstance(payload.get("stop"), str)
            else payload.get("stop") or ()
        ),
        seed=payload.get("seed"),
        log_probabilities=bool(payload.get("logprobs", False)),
        top_log_probabilities=payload.get("top_logprobs"),
        reasoning=extension.get("reasoning") or {},
        thinking=extension.get("thinking", "disabled"),
        stream=bool(payload.get("stream", False)),
        timeout_seconds=timeouts.get("request"),
        first_token_timeout_seconds=timeouts.get("first_token"),
        no_token_progress_timeout_seconds=timeouts.get("no_token_progress"),
        execution_context=extension.get("execution_context") or {},
        request_metadata=extension.get("request_metadata") or {},
        extensions=extension.get("extensions") or {},
    )


def create_app(
    service: InferenceService,
    principal_resolver: PrincipalResolver | None = None,
) -> Any:
    """Build the optional HTTP application without requiring FastAPI at import."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "API support is optional; install cognityx-inference[api]."
        ) from exc
    # FastAPI resolves postponed endpoint annotations from module globals.
    # Keep this import optional while making Request available to that resolver.
    globals()["Request"] = Request

    app = FastAPI(title="Cognityx Inference", version="0.1.0")
    resolver = principal_resolver or EnvironmentPrincipalResolver.from_environment()

    def owner_id(request: Request) -> str:
        try:
            return resolver.resolve(request.headers.get("Authorization"))
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        data = []
        for item in service.models.statuses():
            data.append(
                {
                    "id": item.identity.model,
                    "object": "model",
                    "owned_by": "cognityx",
                    "cognityx": _status(item),
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/cognityx/providers")
    @app.get("/v1/cognityx/providers/status")
    def provider_statuses() -> dict[str, Any]:
        registry = service.provider_registry
        if registry is None:
            raise HTTPException(
                status_code=503, detail="Provider registry unavailable."
            )
        return {
            "object": "list",
            "data": [item.to_dict() for item in registry.list_statuses()],
        }

    @app.get("/v1/cognityx/providers/{provider}/models")
    def provider_models(provider: str, refresh: bool = False) -> dict[str, Any]:
        registry = service.provider_registry
        if registry is None:
            raise HTTPException(
                status_code=503, detail="Provider registry unavailable."
            )
        if provider == "local":
            return {
                "provider": "local",
                "models": [item.identity.model for item in service.models.statuses()],
                "cached": False,
            }
        try:
            return registry.discover_models(provider, refresh=refresh).to_dict()
        except LookupError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "not_configured", "message": str(exc)},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/v1/cognityx/providers/{provider}/capabilities")
    def provider_capabilities(provider: str, model: str) -> dict[str, Any]:
        registry = service.provider_registry
        if registry is None:
            raise HTTPException(
                status_code=503, detail="Provider registry unavailable."
            )
        profile = registry.profile(provider, model)
        return profile.to_dict()

    @app.post("/v1/cognityx/providers/{provider}/test")
    def provider_test(
        provider: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        registry = service.provider_registry
        if registry is None:
            raise HTTPException(
                status_code=503, detail="Provider registry unavailable."
            )
        try:
            definition = registry.definitions[provider]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown provider.") from exc
        if provider == "local":
            return {
                "provider": "local",
                "status": (
                    "passed"
                    if any(
                        item.state.value == "ready"
                        for item in service.models.statuses()
                    )
                    else "skipped"
                ),
                "reason": (
                    None
                    if any(
                        item.state.value == "ready"
                        for item in service.models.statuses()
                    )
                    else "worker_not_ready"
                ),
            }
        selected = payload or {}
        result = test_provider(
            definition,
            credential_resolver=registry.resolver,
            model=selected.get("model"),
            structured_output=bool(selected.get("structured_output", False)),
            timeout_seconds=float(selected.get("timeout_seconds", 30)),
        )
        if result.status == "passed" and result.tested_at:
            registry.mark_success(provider, result.tested_at)
        return result.to_dict()

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any], request: Request) -> Any:
        try:
            normalized = _request_from_chat_payload(payload)
            if normalized.stream:
                return StreamingResponse(
                    _stream_chat(service, normalized, owner_id(request)),
                    media_type="text/event-stream",
                )
            response = service.infer(normalized, owner_id=owner_id(request))
        except HardwareDiscoveryRequired as exc:
            raise HTTPException(status_code=428, detail=exc.to_dict()) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "model_not_loaded", "message": str(exc)},
            ) from exc
        except (CertifiedContextLimitExceeded, ModelContextLimitExceeded) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ContextWindowExceeded, TokenCountingUnavailable) as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": type(exc).__name__, "message": str(exc)},
            ) from exc
        except ModelMetadataUnavailableError as exc:
            raise _model_metadata_http_exception(exc) from exc
        except ProviderCredentialMissing as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_configured",
                    "provider": exc.provider,
                },
            ) from exc
        except UnsupportedProviderParameter as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unsupported_parameter",
                    "provider": exc.provider,
                    "parameters": list(exc.parameters),
                },
            ) from exc
        except ProviderRequestError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": exc.category,
                    "provider": exc.provider,
                    "provider_status": exc.status,
                },
            ) from exc
        except InferenceContractError as exc:
            raise HTTPException(status_code=422, detail=exc.to_dict()) from exc
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": response.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                        **(
                            {"reasoning_content": response.reasoning_content}
                            if response.reasoning_content is not None
                            else {}
                        ),
                    },
                    "finish_reason": response.finish_reason.value,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "cognityx": response.to_dict(),
        }

    @app.post("/v1/cognityx/research/pairs")
    def research_pair(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        runner = service.research_runner
        if runner is None:
            raise HTTPException(
                status_code=503, detail="Research execution is unavailable."
            )
        try:
            return runner.run(
                InferencePairRequest.from_dict(payload), owner_id=owner_id(request)
            )
        except InferenceContractError as exc:
            raise HTTPException(status_code=422, detail=exc.to_dict()) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/cognityx/models/load")
    def load(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        requested_model = str(payload["model"])
        try:
            result, context = service.load_model(
                requested_model,
                str(payload.get("backend", "vllm")),
                str(payload.get("profile", "bf16")),
                payload.get("runtime") or {},
                payload.get("required_context_length"),
                DiscoveryPolicy(payload.get("discovery_policy", "require_existing")),
                owner_id=owner_id(request),
            )
        except HardwareDiscoveryRequired as exc:
            raise HTTPException(status_code=428, detail=exc.to_dict()) from exc
        except (CertifiedContextLimitExceeded, ModelContextLimitExceeded) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelMetadataUnavailableError as exc:
            raise _model_metadata_http_exception(exc) from exc
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **_status(result),
            "requested_model": requested_model,
            "resolved_model": result.identity.model,
            "context_resolution": asdict(context) if context is not None else None,
        }

    @app.get("/v1/cognityx/models/status")
    def statuses() -> list[dict[str, Any]]:
        return [_status(item) for item in service.models.statuses()]

    @app.post("/v1/cognityx/tokens/count")
    def count_tokens(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = InferenceRequest(
                model=str(payload["model"]),
                messages=tuple(payload.get("messages") or ()),
                prompt=payload.get("prompt"),
                tools=tuple(payload.get("tools") or ()),
                tool_choice=payload.get("tool_choice"),
                response_format=payload.get("response_format"),
                backend=str(payload.get("backend", "vllm")),
                profile=str(payload.get("profile", "bf16")),
                load_policy=LoadPolicy.REQUIRE_LOADED,
            )
            return {"input_tokens": service.count_input_tokens(request)}
        except ModelNotLoadedError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "model_not_loaded", "message": str(exc)},
            ) from exc
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/cognityx/certified-profiles")
    def certified_profiles(
        model: str | None = None,
        backend: str | None = None,
        profile: str | None = None,
        kv_cache_precision: str | None = None,
    ) -> list[dict[str, Any]]:
        repository = service.certified_profiles
        if repository is None:
            raise HTTPException(
                status_code=503, detail="Certified profiles are unavailable."
            )
        return [
            item.to_dict()
            for item in repository.list_profiles(
                model=model,
                backend=backend,
                profile=profile,
                kv_cache_precision=kv_cache_precision,
            )
        ]

    @app.get("/v1/cognityx/certified-profiles/{profile_id}")
    def certified_profile(profile_id: str) -> dict[str, Any]:
        repository = service.certified_profiles
        if repository is None:
            raise HTTPException(
                status_code=503, detail="Certified profiles are unavailable."
            )
        result = repository.get_profile(profile_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Certified profile not found.")
        return result.to_dict()

    @app.post("/v1/cognityx/models/unload")
    def unload(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("runtime"):
            return {"unloaded": service.models.unload_active(str(payload["model"]))}
        identity = ModelIdentity.create(
            str(payload["model"]),
            str(payload["backend"]),
            payload.get("runtime") or {},
        )
        return {"unloaded": service.models.unload(identity)}

    @app.post("/v1/cognityx/models/unload-all")
    def unload_all() -> dict[str, Any]:
        outcomes = service.models.unload_all()
        return {
            "models": [
                {
                    "model": identity.model,
                    "backend": identity.backend,
                    "unloaded": unloaded,
                }
                for identity, unloaded in outcomes.items()
            ]
        }

    @app.get("/v1/cognityx/discoveries/{job_id}")
    def discovery_status(job_id: str, request: Request) -> dict[str, Any]:
        coordinator = service.discovery
        status = (
            coordinator.status(job_id, owner_id=owner_id(request))
            if coordinator is not None and hasattr(coordinator, "status")
            else None
        )
        if status is None:
            raise HTTPException(status_code=404, detail="Discovery job not found.")
        return status

    @app.get("/v1/cognityx/discoveries")
    def list_discoveries(request: Request, all: bool = False) -> list[dict[str, Any]]:
        coordinator = service.discovery
        if coordinator is None or not hasattr(coordinator, "list"):
            raise HTTPException(status_code=503, detail="Discovery is unavailable.")
        return coordinator.list(owner_id(request), include_history=all)

    @app.post("/v1/cognityx/discoveries")
    def start_discovery(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        coordinator = service.discovery
        if coordinator is None:
            raise HTTPException(status_code=503, detail="Discovery is unavailable.")
        model = str(payload["model"])
        backend = str(payload.get("backend", "vllm"))
        profile = str(payload.get("profile", "bf16"))
        model_limit, _ = discover_model_context_limit(model)
        job_id = coordinator.start(
            model=model,
            backend=backend,
            profile=profile,
            owner_id=owner_id(request),
            model_context_limit=model_limit,
            runtime=payload.get("runtime") or {},
        )
        return {"job_id": job_id, "state": "queued"}

    @app.post("/v1/cognityx/discoveries/{job_id}/cancel")
    def cancel_discovery(job_id: str, request: Request) -> dict[str, Any]:
        coordinator = service.discovery
        if coordinator is None or not coordinator.cancel(
            job_id, owner_id=owner_id(request)
        ):
            raise HTTPException(
                status_code=404, detail="Discovery job not found or already finished."
            )
        return {"job_id": job_id, "cancel_requested": True}

    @app.get("/v1/cognityx/discoveries/{job_id}/events")
    def discovery_events(job_id: str, request: Request, after: int = 0) -> Any:
        coordinator = service.discovery
        principal = owner_id(request)
        if (
            coordinator is None
            or coordinator.status(job_id, owner_id=principal) is None
        ):
            raise HTTPException(status_code=404, detail="Discovery job not found.")

        def stream() -> Any:
            cursor = after
            while True:
                events = coordinator.events(job_id, cursor, owner_id=principal)
                for event in events:
                    cursor = event["sequence"]
                    yield f"id: {cursor}\nevent: {event['event']}\ndata: {json.dumps(event)}\n\n"
                status = coordinator.status(job_id, owner_id=principal)
                if status and status["state"] in {"completed", "failed", "cancelled"}:
                    break
                yield ": heartbeat\n\n"
                time.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _stream_chat(
    service: InferenceService, request: InferenceRequest, owner_id: str
) -> Any:
    """Bridge synchronous backend callbacks to OpenAI-style SSE chunks."""
    chunks: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_text(text: str) -> None:
        chunks.put(("text", text))

    def run() -> None:
        try:
            response = service.infer(request, on_text=on_text, owner_id=owner_id)
            chunks.put(("response", response))
        except Exception as exc:
            chunks.put(("error", exc))

    threading.Thread(target=run, daemon=True).start()
    created = int(time.time())
    while True:
        kind, value = chunks.get()
        if kind == "text":
            payload = {
                "id": "",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": value},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        elif kind == "response":
            payload = {
                "id": value.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": value.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": value.finish_reason.value,
                    }
                ],
                "usage": asdict(value.usage),
                "cognityx": value.to_dict(),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        else:
            payload = {"error": {"message": str(value), "type": type(value).__name__}}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
