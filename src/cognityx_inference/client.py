"""Thin client for the OpenAI-compatible and Cognityx control APIs."""

from __future__ import annotations

import json
import time
from typing import Iterator
from typing import Any, Mapping
from urllib.error import HTTPError
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.discovery_policy = DiscoveryPolicy(discovery_policy)
        self.discovery_poll_seconds = discovery_poll_seconds
        self.discovery_wait_timeout_seconds = discovery_wait_timeout_seconds
        self.input_fn = input_fn

    def infer(self, request: InferenceRequest) -> dict[str, Any]:
        policy = (
            request.discovery_policy
            if request.discovery_policy is not DiscoveryPolicy.REQUIRE_EXISTING
            else self.discovery_policy
        )
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
        payload = {key: value for key, value in payload.items() if value is not None}
        return self._with_discovery(
            "/v1/chat/completions", payload, policy
        )

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

    def load_model(
        self,
        model: str,
        backend: str = "vllm",
        profile: str = "bf16",
        runtime: Mapping[str, Any] | None = None,
        *,
        discovery_policy: DiscoveryPolicy | str | None = None,
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

    def cancel_discovery(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/cognityx/discoveries/{job_id}/cancel", {})

    def stream_discovery(self, job_id: str, *, after: int = 0) -> Iterator[dict[str, Any]]:
        req = urllib_request.Request(
            f"{self.base_url}/v1/cognityx/discoveries/{job_id}/events?after={after}",
            headers={"Accept": "text/event-stream"},
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
                answer = self.input_fn(
                    "No compatible hardware certification exists. "
                    "Start discovery now? [y/N] "
                )
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
        while True:
            status = self.discovery_status(job_id)
            if status["state"] == "completed":
                return
            if status["state"] == "failed":
                raise RuntimeError(
                    f"Hardware discovery failed: {status.get('error')}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for hardware discovery job {job_id}."
                )
            time.sleep(self.discovery_poll_seconds)

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
                payload = {"message": str(exc)}
            raise InferenceAPIError(exc.code, payload) from exc


def _error_detail(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    detail = payload.get("detail", payload)
    return detail if isinstance(detail, dict) else {}
