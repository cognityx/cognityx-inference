"""Thin client for the OpenAI-compatible and Cognityx control APIs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Any, Iterator, Mapping
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse

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
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120,
        discovery_policy: DiscoveryPolicy | str = DiscoveryPolicy.REQUIRE_EXISTING,
        discovery_poll_seconds: float = 2,
        discovery_wait_timeout_seconds: float = 3600,
        input_fn: Any = input,
        on_discovery_started: Any | None = None,
        on_discovery_event: Any | None = None,
        backend: str | None = None,
        profile: str | None = None,
        auto_start: bool = False,
        manager_url: str | None = None,
        startup_timeout_seconds: float = 600,
        on_server_event: Any | None = None,
    ) -> None:
        selected_url = (
            (base_url or "").strip()
            or os.environ.get("COGNITYX_INFERENCE_URL", "").strip()
            or "http://127.0.0.1:8000"
        )
        self.base_url = self._normalise_base_url(selected_url)
        self.api_key = api_key or os.environ.get("COGNITYX_INFERENCE_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.discovery_policy = DiscoveryPolicy(discovery_policy)
        self.discovery_poll_seconds = discovery_poll_seconds
        self.discovery_wait_timeout_seconds = discovery_wait_timeout_seconds
        self.input_fn = input_fn
        self.on_discovery_started = on_discovery_started
        self.on_discovery_event = on_discovery_event
        self.client_backend = backend
        self.server_profile = profile
        self.auto_start = auto_start
        self.manager_url = self._normalise_base_url(manager_url or selected_url)
        self.startup_timeout_seconds = startup_timeout_seconds
        self.on_server_event = on_server_event
        if auto_start and backend not in {None, "local"}:
            raise ValueError("auto_start is available only for backend='local'")
        if auto_start and not profile:
            raise ValueError("auto_start requires a local server profile")

    @staticmethod
    def _normalise_base_url(value: str) -> str:
        selected_url = value.strip()
        parsed = urlparse(selected_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "Inference base URL must be an absolute http(s) URL, for example "
                "http://127.0.0.1:8000."
            )
        return selected_url.rstrip("/")

    def set_base_url(self, value: str) -> None:
        """Switch this client to another Cognityx inference server."""
        self.base_url = self._normalise_base_url(value)

    def diagnose_server(
        self,
        *,
        model: str | None = None,
        backend: str = "vllm",
        profile: str = "bf16",
    ) -> dict[str, Any]:
        """Return a non-throwing operator diagnostic for the selected server."""
        result: dict[str, Any] = {"base_url": self.base_url, "reachable": False}
        try:
            models = self._request("GET", "/v1/models")
            result.update(
                reachable=True,
                openai_models_endpoint="available",
                models=models.get("data", []),
            )
        except InferenceAPIError as exc:
            result.update(
                error="unexpected_http_response", detail=_diagnostic_detail(exc)
            )
            return result
        except (URLError, OSError, TimeoutError) as exc:
            result.update(error="connection_failed", detail=str(exc))
            return result

        try:
            statuses = self.model_status()
            result["lifecycle_endpoint"] = "available"
            result["loaded_models"] = statuses
        except InferenceAPIError as exc:
            result["lifecycle_endpoint"] = "unavailable"
            result["lifecycle_detail"] = _diagnostic_detail(exc)
            return result

        token_model = model or _status_model(statuses)
        if not token_model:
            result["token_count_endpoint"] = "not_checked_no_model_loaded"
            return result
        try:
            self.count_input_tokens(
                model=token_model,
                backend=backend,
                profile=profile,
                messages=[{"role": "user", "content": "diagnostic"}],
            )
            result["token_count_endpoint"] = "available"
        except InferenceAPIError as exc:
            result["token_count_endpoint"] = "unavailable"
            result["token_count_detail"] = _diagnostic_detail(exc)
        return result

    def infer(self, request: InferenceRequest) -> dict[str, Any]:
        if self.auto_start and request.provider == "local":
            worker_url = self.ensure_server_ready()
            managed = self._worker_client(worker_url)
            return managed.infer(replace(request, load_policy="require_loaded"))
        policy = (
            request.discovery_policy
            if request.discovery_policy is not DiscoveryPolicy.REQUIRE_EXISTING
            else self.discovery_policy
        )
        payload = self._inference_payload(request, policy)
        return self._with_discovery("/v1/chat/completions", payload, policy)

    def stream_infer(self, request: InferenceRequest) -> Iterator[dict[str, Any]]:
        """Load a local model if needed, then yield OpenAI-compatible SSE chunks."""
        policy = (
            request.discovery_policy
            if request.discovery_policy is not DiscoveryPolicy.REQUIRE_EXISTING
            else self.discovery_policy
        )
        if self.auto_start and request.provider == "local":
            worker_url = self.ensure_server_ready()
            yield from self._worker_client(worker_url).stream_infer(
                replace(request, load_policy="require_loaded", stream=True)
            )
            return
        if (
            request.provider == "local"
            and request.load_policy.value != "require_loaded"
        ):
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
        payload = self._inference_payload(request, request.discovery_policy)
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
            "tools": list(request.tools) or None,
            "tool_choice": request.tool_choice,
            "response_format": request.response_format,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens,
            "stop": list(request.stop) or None,
            "seed": request.seed,
            "logprobs": request.log_probabilities,
            "top_logprobs": request.top_log_probabilities,
            "stream": request.stream,
            "cognityx": {
                "client_type": request.client_type,
                "provider": request.provider,
                "backend": request.backend,
                "profile": request.profile,
                "model_revision": request.model_revision,
                "adapter_manifest_uri": request.adapter_manifest_uri,
                "adapter_purpose": (
                    request.adapter_purpose.value
                    if request.adapter_purpose is not None
                    else None
                ),
                "max_output_tokens": request.max_output_tokens,
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
                "execution_context": dict(request.execution_context),
                "request_metadata": dict(request.request_metadata),
                "extensions": dict(request.extensions),
            },
        }
        return {key: value for key, value in payload.items() if value is not None}

    def run_research_pair(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute and publish one frozen base/adapter research pair."""
        return self._request("POST", "/v1/cognityx/research/pairs", dict(payload))

    def chat(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | None = None,
        prompt: str | None = None,
        client_type: str = "openai",
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
                client_type=client_type,
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
        client_type: str = "openai",
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
                client_type=client_type,
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

    def list_providers(self) -> list[dict[str, Any]]:
        value = self._request("GET", "/v1/cognityx/providers")
        return list(value.get("data", ()))

    def provider_status(self) -> list[dict[str, Any]]:
        value = self._request("GET", "/v1/cognityx/providers/status")
        return list(value.get("data", ()))

    def provider_models(
        self,
        provider: str,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        query = "?refresh=true" if refresh else ""
        return self._request(
            "GET",
            f"/v1/cognityx/providers/{quote(provider, safe='')}/models{query}",
        )

    def provider_capabilities(
        self,
        provider: str,
        model: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            (
                f"/v1/cognityx/providers/{quote(provider, safe='')}/capabilities"
                f"?model={quote(model, safe='')}"
            ),
        )

    def test_provider(
        self,
        provider: str,
        *,
        model: str | None = None,
        structured_output: bool = False,
        timeout_seconds: float = 30,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/cognityx/providers/{quote(provider, safe='')}/test",
            {
                "model": model,
                "structured_output": structured_output,
                "timeout_seconds": timeout_seconds,
            },
        )

    def count_input_tokens(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]] | None = None,
        prompt: str | None = None,
        backend: str = "vllm",
        profile: str = "bf16",
    ) -> int | None:
        value = self._request(
            "POST",
            "/v1/cognityx/tokens/count",
            {
                "model": model,
                "messages": list(messages or ()),
                "prompt": prompt,
                "backend": backend,
                "profile": profile,
            },
        )
        return value.get("input_tokens") if isinstance(value, Mapping) else None

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

    def server_start(self, profile: str | None = None) -> dict[str, Any]:
        selected = profile or self.server_profile
        if not selected:
            raise ValueError("A local server profile is required.")
        return self._manager_request(
            "POST",
            "/management/v1/server/start",
            {"profile": selected},
        )

    def server_stop(self) -> dict[str, Any]:
        return self._manager_request("POST", "/management/v1/server/stop", {})

    def server_status(self) -> dict[str, Any]:
        return self._manager_request("GET", "/management/v1/server/status")

    def stream_server_events(self, *, after: int = 0) -> Iterator[dict[str, Any]]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        outgoing = urllib_request.Request(
            f"{self.manager_url}/management/v1/server/events?after={after}",
            headers=headers,
        )
        try:
            with urllib_request.urlopen(
                outgoing, timeout=self.startup_timeout_seconds
            ) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if line.startswith("data: "):
                        yield json.loads(line[6:])
        except HTTPError as exc:
            raise InferenceAPIError(
                exc.code,
                {"error": "manager_event_stream_http_error"},
            ) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot stream events from the Cognityx inference manager "
                f"at {self.manager_url}."
            ) from exc

    def ensure_server_ready(self) -> str:
        status = self.server_status()
        if (
            status.get("state") == "READY"
            and status.get("profile") == self.server_profile
            and status.get("worker_url")
        ):
            return str(status["worker_url"])
        status = self.server_start()
        deadline = time.monotonic() + self.startup_timeout_seconds
        cursor = 0
        while True:
            try:
                for event in self.stream_server_events(after=cursor):
                    cursor = max(cursor, int(event.get("sequence", cursor)))
                    if self.on_server_event:
                        self.on_server_event(event)
                    state = str(event.get("state", ""))
                    if state == "READY":
                        current = self.server_status()
                        return str(current["worker_url"])
                    if state == "FAILED":
                        raise RuntimeError(
                            f"Local inference server failed: {current_error(event)}"
                        )
                    if state == "STOPPED":
                        raise RuntimeError(
                            "Local inference server stopped during startup."
                        )
            except (URLError, OSError, TimeoutError, HTTPError):
                pass
            status = self.server_status()
            if status.get("state") == "READY":
                return str(status["worker_url"])
            if status.get("state") == "FAILED":
                raise RuntimeError(
                    f"Local inference server failed: {current_error(status)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for local server profile {self.server_profile}."
                )
            time.sleep(min(self.discovery_poll_seconds, 1.0))

    def _worker_client(self, worker_url: str) -> "CognityxInferenceClient":
        return CognityxInferenceClient(
            worker_url,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            discovery_policy=self.discovery_policy,
            discovery_poll_seconds=self.discovery_poll_seconds,
            discovery_wait_timeout_seconds=self.discovery_wait_timeout_seconds,
            input_fn=self.input_fn,
            on_discovery_started=self.on_discovery_started,
            on_discovery_event=self.on_discovery_event,
        )

    def _manager_request(
        self, method: str, path: str, payload: Any | None = None
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        outgoing = urllib_request.Request(
            f"{self.manager_url}{path}",
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
                value = json.loads(exc.read())
            except (ValueError, OSError):
                value = {"error": "manager_http_error", "status": exc.code}
            raise InferenceAPIError(exc.code, value) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot connect to the Cognityx inference manager at "
                f"{self.manager_url}. Start it with "
                "'cognityx-inference manager serve'."
            ) from exc

    def discovery_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/cognityx/discoveries/{job_id}")

    def list_discoveries(
        self, *, include_history: bool = False
    ) -> list[dict[str, Any]]:
        suffix = "?all=true" if include_history else ""
        return self._request("GET", f"/v1/cognityx/discoveries{suffix}")

    def list_certified_profiles(
        self,
        *,
        model: str | None = None,
        backend: str | None = None,
        profile: str | None = None,
        kv_cache_precision: str | None = None,
    ) -> list[dict[str, Any]]:
        query = {
            key: value
            for key, value in {
                "model": model,
                "backend": backend,
                "profile": profile,
                "kv_cache_precision": kv_cache_precision,
            }.items()
            if value is not None
        }
        suffix = f"?{urlencode(query)}" if query else ""
        return self._request("GET", f"/v1/cognityx/certified-profiles{suffix}")

    def get_certified_profile(self, profile_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/cognityx/certified-profiles/{profile_id}")

    def start_discovery(
        self,
        model: str,
        backend: str = "vllm",
        profile: str = "bf16",
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/cognityx/discoveries",
            {
                "model": model,
                "backend": backend,
                "profile": profile,
                "runtime": dict(runtime or {}),
            },
        )

    def cancel_discovery(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/cognityx/discoveries/{job_id}/cancel", {})

    def stream_discovery(
        self, job_id: str, *, after: int = 0
    ) -> Iterator[dict[str, Any]]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib_request.Request(
            f"{self.base_url}/v1/cognityx/discoveries/{job_id}/events?after={after}",
            headers=headers,
        )
        with urllib_request.urlopen(
            req, timeout=self.discovery_wait_timeout_seconds
        ) as response:
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
                            event["error"].get("message", "Inference stream failed.")
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
            if (
                exc.status != 428
                or detail.get("error") != "hardware_discovery_required"
            ):
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
                            job_id,
                            event.get("event", "").removeprefix("discovery_"),
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
    def _terminal_discovery_state(job_id: str, state: str, error: Any = None) -> bool:
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

    def _request(self, method: str, path: str, payload: Any | None = None) -> Any:
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


def _diagnostic_detail(exc: InferenceAPIError) -> str:
    detail = _error_detail(exc.payload)
    message = detail.get("message") if detail else None
    return str(message or exc)


def _status_model(statuses: list[dict[str, Any]]) -> str | None:
    if not statuses:
        return None
    identity = statuses[-1].get("identity") or {}
    model = identity.get("model")
    return str(model) if model else None


def current_error(value: Mapping[str, Any]) -> str:
    """Return only the manager's safe categorical failure detail."""
    return str(value.get("error_category") or "unknown_error")
