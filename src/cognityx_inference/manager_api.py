"""FastAPI management surface for one local inference worker."""

from __future__ import annotations

import json
import time
from typing import Any

from cognityx_inference.auth import EnvironmentPrincipalResolver, PrincipalResolver
from cognityx_inference.manager import InferenceManager, ServerState


def create_manager_app(
    manager: InferenceManager,
    principal_resolver: PrincipalResolver | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "Manager API support requires cognityx-inference[api]."
        ) from exc
    globals()["Request"] = Request
    app = FastAPI(title="Cognityx Inference Manager", version="0.1.0")
    resolver = principal_resolver or EnvironmentPrincipalResolver.from_environment()

    def owner_id(request: Request) -> str:
        try:
            return resolver.resolve(request.headers.get("Authorization"))
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/management/v1/server/start")
    def start(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        try:
            return manager.start(
                str(payload["profile"]), owner_id=owner_id(request)
            ).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/management/v1/server/stop")
    def stop(request: Request) -> dict[str, Any]:
        owner_id(request)
        return manager.stop().to_dict()

    @app.get("/management/v1/server/status")
    def status(request: Request) -> dict[str, Any]:
        owner_id(request)
        return manager.status().to_dict()

    @app.get("/management/v1/server/events")
    def events(request: Request, after: int = 0) -> Any:
        owner_id(request)

        def stream() -> Any:
            cursor = max(0, after)
            while True:
                available = manager.events(after=cursor)
                for event in available:
                    cursor = max(cursor, int(event.get("sequence", cursor)))
                    yield f"data: {json.dumps(event)}\n\n"
                state = manager.status().state
                if state in {
                    ServerState.READY,
                    ServerState.FAILED,
                    ServerState.STOPPED,
                }:
                    return
                time.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
