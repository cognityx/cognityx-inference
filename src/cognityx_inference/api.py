"""OpenAI-compatible FastAPI application factory."""

from __future__ import annotations

from dataclasses import asdict
import json
import queue
import threading
import time
from typing import Any

from cognityx_inference.capabilities import HardwareDiscoveryRequired
from cognityx_inference.capabilities import (
    CertifiedContextLimitExceeded,
    ModelContextLimitExceeded,
)
from cognityx_inference.errors import ModelMetadataUnavailableError
from cognityx_inference.contracts import (
    DiscoveryPolicy,
    InferenceRequest,
    LoadPolicy,
)
from cognityx_inference.lifecycle import ModelIdentity
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


def create_app(service: InferenceService) -> Any:
    """Build the optional HTTP application without requiring FastAPI at import."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "API support is optional; install cognityx-inference[api]."
        ) from exc

    app = FastAPI(title="Cognityx Inference", version="0.1.0")

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

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> Any:
        extension = payload.get("cognityx") or {}
        timeouts = extension.get("timeouts") or {}
        try:
            normalized = InferenceRequest(
                model=str(payload["model"]),
                messages=tuple(payload.get("messages") or ()),
                provider=str(extension.get("provider", "local")),
                backend=str(extension.get("backend", "vllm")),
                profile=str(extension.get("profile", "bf16")),
                load_policy=(
                    LoadPolicy(extension.get("load_policy", "auto"))
                    if extension
                    else LoadPolicy.REQUIRE_LOADED
                ),
                discovery_policy=DiscoveryPolicy(
                    extension.get("discovery_policy", "require_existing")
                ),
                required_context_length=extension.get(
                    "required_context_length"
                ),
                temperature=payload.get("temperature"),
                top_p=payload.get("top_p"),
                top_k=extension.get("top_k"),
                min_p=extension.get("min_p"),
                max_tokens=payload.get("max_tokens"),
                stop=tuple(
                    [payload["stop"]]
                    if isinstance(payload.get("stop"), str)
                    else payload.get("stop") or ()
                ),
                seed=payload.get("seed"),
                log_probabilities=bool(payload.get("logprobs", False)),
                top_log_probabilities=payload.get("top_logprobs"),
                reasoning=extension.get("reasoning") or {},
                stream=bool(payload.get("stream", False)),
                timeout_seconds=timeouts.get("request"),
                first_token_timeout_seconds=timeouts.get("first_token"),
                no_token_progress_timeout_seconds=timeouts.get("no_token_progress"),
                extensions=extension.get("extensions") or {},
            )
            if normalized.stream:
                return StreamingResponse(
                    _stream_chat(service, normalized),
                    media_type="text/event-stream",
                )
            response = service.infer(normalized)
        except HardwareDiscoveryRequired as exc:
            raise HTTPException(status_code=428, detail=exc.to_dict()) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "model_not_loaded", "message": str(exc)},
            ) from exc
        except (CertifiedContextLimitExceeded, ModelContextLimitExceeded) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelMetadataUnavailableError as exc:
            raise _model_metadata_http_exception(exc) from exc
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

    @app.post("/v1/cognityx/models/load")
    def load(payload: dict[str, Any]) -> dict[str, Any]:
        requested_model = str(payload["model"])
        try:
            result, context = service.load_model(
                requested_model,
                str(payload.get("backend", "vllm")),
                str(payload.get("profile", "bf16")),
                payload.get("runtime") or {},
                payload.get("required_context_length"),
                DiscoveryPolicy(
                    payload.get("discovery_policy", "require_existing")
                ),
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

    @app.post("/v1/cognityx/models/unload")
    def unload(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("runtime"):
            return {
                "unloaded": service.models.unload_active(
                    str(payload["model"])
                )
            }
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
    def discovery_status(job_id: str) -> dict[str, Any]:
        coordinator = service.discovery
        status = (
            coordinator.status(job_id)
            if coordinator is not None and hasattr(coordinator, "status")
            else None
        )
        if status is None:
            raise HTTPException(status_code=404, detail="Discovery job not found.")
        return status

    @app.post("/v1/cognityx/discoveries/{job_id}/cancel")
    def cancel_discovery(job_id: str) -> dict[str, Any]:
        coordinator = service.discovery
        if coordinator is None or not coordinator.cancel(job_id):
            raise HTTPException(status_code=404, detail="Discovery job not found or already finished.")
        return {"job_id": job_id, "cancel_requested": True}

    @app.get("/v1/cognityx/discoveries/{job_id}/events")
    def discovery_events(job_id: str, after: int = 0) -> Any:
        coordinator = service.discovery
        if coordinator is None or coordinator.status(job_id) is None:
            raise HTTPException(status_code=404, detail="Discovery job not found.")

        def stream() -> Any:
            cursor = after
            while True:
                events = coordinator.events(job_id, cursor)
                for event in events:
                    cursor = event["sequence"]
                    yield f"id: {cursor}\nevent: {event['event']}\ndata: {json.dumps(event)}\n\n"
                status = coordinator.status(job_id)
                if status and status["state"] in {"completed", "failed", "cancelled"}:
                    break
                yield ": heartbeat\n\n"
                time.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _stream_chat(
    service: InferenceService, request: InferenceRequest
) -> Any:
    """Bridge synchronous backend callbacks to OpenAI-style SSE chunks."""
    chunks: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_text(text: str) -> None:
        chunks.put(("text", text))

    def run() -> None:
        try:
            response = service.infer(request, on_text=on_text)
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
