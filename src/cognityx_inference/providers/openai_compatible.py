"""Secure adapters for commercial OpenAI-compatible chat APIs."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib import request as urllib_request

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.contracts import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
    TokenDetail,
    TokenUsage,
)
from cognityx_inference.errors import (
    ProviderCredentialMissing,
    ProviderRequestError,
)
from cognityx_inference.token_budget import serialized_request


_ALLOWED_EXTENSIONS = {
    "frequency_penalty",
    "presence_penalty",
    "parallel_tool_calls",
    "reasoning_effort",
    "service_tier",
}


class OpenAICompatibleProvider:
    """Call one configured OpenAI-compatible endpoint."""

    capabilities = ModelCapabilities(
        streaming=True,
        stop_conditions=True,
        seed=True,
        log_probabilities=True,
        reasoning=True,
    )

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        *,
        api_key_env: str | None = None,
        model_capabilities: dict[str, CertifiedContextProfile] | None = None,
        default_timeout_seconds: float = 120,
        output_tokens_parameter: str = "max_completion_tokens",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.model_capabilities = dict(model_capabilities or {})
        self.default_timeout_seconds = default_timeout_seconds
        self.output_tokens_parameter = output_tokens_parameter

    @classmethod
    def from_definition(
        cls, definition: ProviderDefinition
    ) -> "OpenAICompatibleProvider":
        if definition.adapter != "openai_compatible":
            raise ValueError(
                f"Unsupported provider adapter: {definition.adapter}"
            )
        return cls(
            definition.name,
            definition.base_url,
            api_key_env=definition.api_key_env,
            model_capabilities=dict(definition.model_capabilities),
        )

    def context_profile(self, model: str) -> CertifiedContextProfile | None:
        return self.model_capabilities.get(model)

    def count_input_tokens(self, request: InferenceRequest) -> int | None:
        profile = self.context_profile(request.model)
        if profile is None or not profile.tokenizer:
            return None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                profile.tokenizer, local_files_only=True
            )
            return len(
                tokenizer.encode(
                    serialized_request(request),
                    add_special_tokens=False,
                )
            )
        except (OSError, TypeError, ValueError, AttributeError):
            return None

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> InferenceResponse:
        payload = self._payload(request)
        if request.stream:
            return self._stream(payload, request, on_text)
        body, elapsed, request_id = self._request(
            "chat/completions", payload, request.timeout_seconds
        )
        return self._response(
            body,
            request,
            elapsed=elapsed,
            request_id=request_id,
        )

    def list_models(self, *, timeout_seconds: float = 20) -> tuple[str, ...]:
        body, _, _ = self._request("models", None, timeout_seconds, method="GET")
        return tuple(
            str(item["id"])
            for item in body.get("data", ())
            if isinstance(item, dict) and item.get("id")
        )

    def _payload(self, request: InferenceRequest) -> dict[str, Any]:
        if request.top_k is not None or request.min_p is not None:
            raise ValueError(
                f"{self.name} does not expose top_k/min_p through its common API"
            )
        unknown = sorted(set(request.extensions) - _ALLOWED_EXTENSIONS)
        if unknown:
            raise ValueError(
                f"{self.name} unsupported extension parameter(s): "
                + ", ".join(unknown)
            )
        messages = (
            list(request.messages)
            if request.messages
            else [{"role": "user", "content": request.prompt}]
        )
        payload: dict[str, Any] = {"model": request.model, "messages": messages}
        optional = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            self.output_tokens_parameter: request.max_output_tokens,
            "seed": request.seed,
            "logprobs": request.log_probabilities or None,
            "top_logprobs": request.top_log_probabilities,
            "stop": list(request.stop) or None,
            "tools": list(request.tools) or None,
            "tool_choice": request.tool_choice,
            "response_format": request.response_format,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        payload.update(request.extensions)
        if request.reasoning:
            payload.setdefault(
                "reasoning_effort",
                request.reasoning.get("effort"),
            )
        return {key: value for key, value in payload.items() if value is not None}

    def _stream(
        self,
        payload: dict[str, Any],
        request: InferenceRequest,
        on_text: Callable[[str], None] | None,
    ) -> InferenceResponse:
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        outgoing = self._outgoing("chat/completions", payload, method="POST")
        started = time.monotonic()
        first_token: float | None = None
        content: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, Any] = {}
        model = request.model
        provider_request_id = ""
        finish = FinishReason.UNKNOWN
        try:
            with urllib_request.urlopen(
                outgoing,
                timeout=request.timeout_seconds or self.default_timeout_seconds,
            ) as response:
                header_id = response.headers.get("x-request-id", "")
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ProviderRequestError(
                            self.name, "invalid_stream_event"
                        ) from exc
                    provider_request_id = str(event.get("id") or header_id)
                    model = str(event.get("model") or model)
                    usage.update(event.get("usage") or {})
                    for choice in event.get("choices") or ():
                        delta = choice.get("delta") or {}
                        text = str(delta.get("content") or "")
                        thought = str(delta.get("reasoning_content") or "")
                        if text:
                            if first_token is None:
                                first_token = time.monotonic() - started
                            content.append(text)
                            if on_text:
                                on_text(text)
                        if thought:
                            reasoning.append(thought)
                        if choice.get("finish_reason"):
                            finish = _finish_reason(choice["finish_reason"])
        except HTTPError as exc:
            raise ProviderRequestError(
                self.name, _http_category(exc.code), status=exc.code
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "connection_error") from exc
        elapsed = time.monotonic() - started
        return InferenceResponse(
            request_id=provider_request_id,
            content="".join(content),
            reasoning_content="".join(reasoning) or None,
            model=model,
            provider=self.name,
            backend="commercial-api",
            finish_reason=finish,
            usage=_usage(usage),
            timings=InferenceTimings(
                latency_seconds=elapsed,
                time_to_first_token_seconds=first_token,
                token_generation_seconds=elapsed,
            ),
            requested_parameters=request.to_dict(),
            effective_parameters=_safe_parameters(payload),
            extensions={"streaming": True},
        )

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None,
        *,
        method: str = "POST",
    ) -> tuple[dict[str, Any], float, str]:
        outgoing = self._outgoing(path, payload, method=method)
        started = time.monotonic()
        try:
            with urllib_request.urlopen(
                outgoing,
                timeout=timeout_seconds or self.default_timeout_seconds,
            ) as response:
                body = json.loads(response.read())
                request_id = response.headers.get("x-request-id", "")
        except HTTPError as exc:
            raise ProviderRequestError(
                self.name, _http_category(exc.code), status=exc.code
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderRequestError(self.name, "invalid_json") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "connection_error") from exc
        return body, time.monotonic() - started, request_id

    def _outgoing(
        self, path: str, payload: dict[str, Any] | None, *, method: str
    ) -> urllib_request.Request:
        encoded = (
            json.dumps(payload).encode("utf-8") if payload is not None else None
        )
        return urllib_request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self._credential()}",
                "Content-Type": "application/json",
            },
            method=method,
        )

    def _credential(self) -> str:
        value = self._api_key
        if value is None and self.api_key_env:
            value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderCredentialMissing(
                self.name, self.api_key_env or "configured provider credential"
            )
        return value

    def _response(
        self,
        body: dict[str, Any],
        request: InferenceRequest,
        *,
        elapsed: float,
        request_id: str,
    ) -> InferenceResponse:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        return InferenceResponse(
            request_id=str(body.get("id") or request_id),
            content=str(message.get("content") or ""),
            reasoning_content=message.get("reasoning_content"),
            model=str(body.get("model") or request.model),
            provider=self.name,
            backend="commercial-api",
            finish_reason=_finish_reason(choice.get("finish_reason")),
            usage=_usage(usage),
            timings=InferenceTimings(latency_seconds=elapsed),
            token_details=self._token_details(choice),
            requested_parameters=request.to_dict(),
            effective_parameters=_safe_parameters(self._payload(request)),
            extensions={
                "system_fingerprint": body.get("system_fingerprint"),
            },
        )

    @staticmethod
    def _token_details(choice: dict[str, Any]) -> tuple[TokenDetail, ...]:
        content = (choice.get("logprobs") or {}).get("content") or ()
        return tuple(
            TokenDetail(
                token=str(item.get("token", "")),
                log_probability=item.get("logprob"),
                top_log_probabilities={
                    str(top.get("token", "")): top.get("logprob")
                    for top in item.get("top_logprobs", ())
                },
            )
            for item in content
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """Backward-compatible OpenAI adapter constructor."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            "openai",
            "https://api.openai.com/v1",
            api_key,
            api_key_env="OPENAI_API_KEY",
            **kwargs,
        )


class GroqProvider(OpenAICompatibleProvider):
    """Groq OpenAI-compatible adapter."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            "groq",
            "https://api.groq.com/openai/v1",
            api_key,
            api_key_env="GROQ_API_KEY",
            **kwargs,
        )


class XAIProvider(OpenAICompatibleProvider):
    """Existing xAI/Grok compatibility adapter."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            "xai",
            "https://api.x.ai/v1",
            api_key,
            api_key_env="XAI_API_KEY",
            **kwargs,
        )


def _finish_reason(value: Any) -> FinishReason:
    aliases = {"tool_calls": FinishReason.TOOL_CALL}
    if value in aliases:
        return aliases[value]
    try:
        return FinishReason(str(value or "unknown"))
    except ValueError:
        return FinishReason.UNKNOWN


def _usage(value: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=value.get("prompt_tokens"),
        completion_tokens=value.get("completion_tokens"),
        total_tokens=value.get("total_tokens"),
    )


def _safe_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"messages", "tools"}
    }


def _http_category(status: int) -> str:
    if status in {401, 403}:
        return "authentication"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "provider_unavailable"
    return "invalid_request"
