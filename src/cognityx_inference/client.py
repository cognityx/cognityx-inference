"""Thin client for the OpenAI-compatible and Cognityx control APIs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Iterator
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib import request as urllib_request

from cognityx_inference.contracts import DiscoveryPolicy, InferenceRequest


class InferenceAPIError(RuntimeError):
    """Structured error returned by the inference service."""

    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(f"Inference API returned HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


class CognityxInferenceClient:
    """Call a remote Cognityx inference service."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120,
        discovery_policy: DiscoveryPolicy | str = DiscoveryPolicy.REQUIRE_EXISTING,
        discovery_poll_seconds: float = 2,
        discovery_wait_timeout_seconds: float = 3600,
        input_fn: Any = input,
        on_discovery_started: Any | None = None,
        on_discovery_event: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("COGNITYX_INFERENCE_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.discovery_policy = DiscoveryPolicy(discovery_policy)
        self.discovery_poll_seconds = discovery_poll_seconds
        self.discovery_wait_timeout_seconds = discovery_wait_timeout_seconds
        self.input_fn = input_fn
        self.on_discovery_started = on_discovery_started
        self.on_discovery_event = on_discovery_event

    def infer(self, request: InferenceRequest) -> dict[str, Any]:
        policy = (
            request.discovery_policy
            if request.discovery_policy is not DiscoveryPolicy.REQUIRE_EXISTING
            else self.discovery_policy
        )
        payload = self._inference_payload(request, policy)
        return self._with_discovery(
            "/v1/chat/completions", payload, policy
        )

    def stream_infer(
        self, request: InferenceRequest
    ) -> Iterator[dict[str, Any]]:
        """Load a local model if needed, then yield OpenAI-compatible SSE chunks."""
        policy = (
            request.discovery_policy
            if request.discovery_policy is not DiscoveryPolicy.REQUIRE_EXISTING
            else self.discovery_policy
        )
        if request.provider == "local":
            runtime = request.extensions.get("runtime", {})
            self.load_model(
                request.model,
                request.backend,
                request.profile,
                runtime if isinstance(runtime, Mapping) else {},
                discovery_policy=policy,
                required_context_length=request.required_context_length,
            )
            request = replace(
                request,
                load_policy="require_loaded",
                discovery_policy=DiscoveryPolicy.REQUIRE_EXISTING,
                stream=True,
            )
        else:
            request = replace(request, stream=True)
        payload = self._inference_payload(
            request, request.discovery_policy
        )
        yield from self._stream_request("/v1/chat/completions", payload)

    @staticmethod
    def _inference_payload(
        request: InferenceRequest, policy: DiscoveryPolicy
    ) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": list(request.messages)
            if request.messages
            else [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": list(request.stop) or None,
            "seed": request.seed,
            "logprobs": request.log_probabilities,
            "stream": request.stream,
            "cognityx": {
                "provider": request.provider,
                "backend": request.backend,
                "profile": request.profile,
                "load_policy": request.load_policy.value,
                "discovery_policy": policy.value,
                "required_context_length": request.required_context_length,
                "top_k": request.top_k,
                "min_p": request.min_p,
                "reasoning": dict(request.reasoning),
                "timeouts": {
                    "request": request.timeout_seconds,
                    "first_token": request.first_token_timeout_seconds,
                    "no_token_progress": request.no_token_progress_timeout_seconds,
                },
                "extensions": dict(request.extensions),
            },
        }
        return {key: value for key, value in payload.items() if value is not None}

    def chat(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | None = None,
        prompt: str | None = None,
        backend: str = "vllm",
        profile: str = "bf16",
        required_context_length: int | None = None,
        discovery_policy: DiscoveryPolicy | str | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Convenience wrapper around the normalized request contract."""
        return self.infer(
            InferenceRequest(
                model=model,
                messages=tuple(messages or ()),
                prompt=prompt,
                backend=backend,
                profile=profile,
                required_context_length=required_context_length,
                discovery_policy=(
                    DiscoveryPolicy(discovery_policy)
                    if discovery_policy is not None
                    else self.discovery_policy
                ),
                **parameters,
            )
        )

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | None = None,
        prompt: str | None = None,
        backend: str = "vllm",
        profile: str = "bf16",
        required_context_length: int | None = None,
        discovery_policy: DiscoveryPolicy | str | None = None,
        **parameters: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream normalized chat-completion chunks as they arrive."""
        yield from self.stream_infer(
            InferenceRequest(
                model=model,
                messages=tuple(messages or ()),
                prompt=prompt,
                backend=backend,
                profile=profile,
                required_context_length=required_context_length,
                discovery_policy=(
                    DiscoveryPolicy(discovery_policy)
                    if discovery_policy is not None
                    else self.discovery_policy
                ),
                stream=True,
                **parameters,
            )
        )

    def load_model(
        self,
        model: str,
        backend: str = "vllm",
        profile: str = "bf16",
        runtime: Mapping[str, Any] | None = None,
        *,
        discovery_policy: DiscoveryPolicy | str | None = None,
        required_context_length: int | None = None,
    ) -> dict[str, Any]:
        policy = DiscoveryPolicy(discovery_policy or self.discovery_policy)
        return self._with_discovery(
            "/v1/cognityx/models/load",
            {
                "model": model,
                "backend": backend,
                "profile": profile,
                "runtime": dict(runtime or {}),
                "discovery_policy": policy.value,
                "required_context_length": required_context_length,
            },
            policy,
        )

    def model_status(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/cognityx/models/status")

    def unload_model(
        self, model: str, backend: str, runtime: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/cognityx/models/unload",
            {"model": model, "backend": backend, "runtime": dict(runtime or {})},
        )

    def unload_all(self) -> dict[str, Any]:
        return self._request("POST", "/v1/cognityx/models/unload-all", {})

    def discovery_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/cognityx/discoveries/{job_id}")

    def list_discoveries(self, *, include_history: bool = False) -> list[dict[str, Any]]:
        suffix = "?all=true" if include_history else ""
        return self._request("GET", f"/v1/cognityx/discoveries{suffix}")

    def start_discovery(self, model: str, backend: str = "vllm", profile: str = "bf16", runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", "/v1/cognityx/discoveries", {"model": model, "backend": backend, "profile": profile, "runtime": dict(runtime or {})})

    def cancel_discovery(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/cognityx/discoveries/{job_id}/cancel", {})

    def stream_discovery(self, job_id: str, *, after: int = 0) -> Iterator[dict[str, Any]]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib_request.Request(
            f"{self.base_url}/v1/cognityx/discoveries/{job_id}/events?after={after}",
            headers=headers,
        )
        with urllib_request.urlopen(req, timeout=self.discovery_wait_timeout_seconds) as response:
            event: dict[str, Any] = {}
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                elif not line and event:
                    yield event
                    event = {}

    def _stream_request(
        self, path: str, payload: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        outgoing = urllib_request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                outgoing, timeout=self.timeout_seconds
            ) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(
                            event["error"].get(
                                "message", "Inference stream failed."
                            )
                        )
                    yield event
        except HTTPError as exc:
            try:
                error = json.loads(exc.read())
            except (ValueError, OSError):
                error = {
                    "error": "http_error",
                    "message": str(exc),
                    "method": "POST",
                    "url": outgoing.full_url,
                }
            raise InferenceAPIError(exc.code, error) from exc

    def _with_discovery(
        self,
        path: str,
        payload: dict[str, Any],
        policy: DiscoveryPolicy,
    ) -> Any:
        try:
            return self._request("POST", path, payload)
        except InferenceAPIError as exc:
            detail = _error_detail(exc.payload)
            if exc.status != 428 or detail.get("error") != "hardware_discovery_required":
                raise
            if policy is DiscoveryPolicy.REQUIRE_EXISTING:
                raise
            if policy is DiscoveryPolicy.ASK:
                try:
                    answer = self.input_fn(
                        "No compatible hardware certification exists. "
                        "Start discovery now? [y/N] "
                    )
                except EOFError:
                    # A non-interactive caller cannot consent to starting a
                    # potentially long-running discovery job. Preserve the
                    # server's structured precondition response instead.
                    raise exc from None
                if str(answer).strip().lower() not in {"y", "yes"}:
                    raise
                policy = DiscoveryPolicy.AUTO
                payload = dict(payload)
                if path.endswith("chat/completions"):
                    payload["cognityx"] = {
                        **payload["cognityx"],
                        "discovery_policy": "auto",
                    }
                else:
                    payload["discovery_policy"] = "auto"
                try:
                    return self._request("POST", path, payload)
                except InferenceAPIError as started:
                    detail = _error_detail(started.payload)
                    if started.status != 428:
                        raise
            job_id = detail.get("discovery_request_id")
            if not job_id:
                raise
            if self.on_discovery_started is not None:
                self.on_discovery_started(
                    {
                        "event": "hardware_discovery_started",
                        "job_id": job_id,
                        "model": detail.get("model"),
                        "backend": detail.get("backend"),
                        "profile": detail.get("profile"),
                    }
                )
            self._wait_for_discovery(job_id)
            if path.endswith("chat/completions"):
                payload["cognityx"] = {
                    **payload["cognityx"],
                    "load_policy": "require_loaded",
                    "discovery_policy": "require_existing",
                }
            else:
                payload["discovery_policy"] = "require_existing"
            return self._request("POST", path, payload)

    def _wait_for_discovery(self, job_id: str) -> None:
        deadline = time.monotonic() + self.discovery_wait_timeout_seconds
        cursor = 0
        while True:
            if self.on_discovery_event is not None:
                try:
                    for event in self.stream_discovery(job_id, after=cursor):
                        cursor = max(cursor, int(event.get("sequence", cursor)))
                        self.on_discovery_event(event)
                        terminal = self._terminal_discovery_state(
                            job_id, event.get("event", "").removeprefix("discovery_"),
                            event.get("error"),
                        )
                        if terminal:
                            return
                except (
                    HTTPError,
                    URLError,
                    TimeoutError,
                    OSError,
                    json.JSONDecodeError,
                ):
                    # The durable status endpoint is authoritative. An SSE
                    # disconnect must not terminate a long-running discovery.
                    pass
            try:
                status = self.discovery_status(job_id)
            except (URLError, TimeoutError, OSError) as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for hardware discovery job {job_id}."
                    ) from exc
                time.sleep(self.discovery_poll_seconds)
                continue
            if self._terminal_discovery_state(
                job_id, str(status["state"]), status.get("error")
            ):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for hardware discovery job {job_id}."
                )
            time.sleep(self.discovery_poll_seconds)

    @staticmethod
    def _terminal_discovery_state(
        job_id: str, state: str, error: Any = None
    ) -> bool:
        if state == "completed":
            return True
        if state == "failed":
            raise RuntimeError(f"Hardware discovery failed: {error}")
        if state == "cancelled":
            raise RuntimeError(f"Hardware discovery {job_id} was cancelled.")
        if state == "interrupted":
            raise RuntimeError(
                f"Hardware discovery {job_id} was interrupted by a server restart."
            )
        return False

    def _request(
        self, method: str, path: str, payload: Any | None = None
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        outgoing = urllib_request.Request(
            f"{self.base_url}{path}",
            data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(
                outgoing, timeout=self.timeout_seconds
            ) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read())
            except (ValueError, OSError):
                payload = {
                    "error": "http_error",
                    "message": str(exc),
                    "method": method,
                    "url": outgoing.full_url,
                }
            raise InferenceAPIError(exc.code, payload) from exc


def _error_detail(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    detail = payload.get("detail", payload)
    return detail if isinstance(detail, dict) else {}
