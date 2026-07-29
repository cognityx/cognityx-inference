"""Native Gemini, Anthropic, and GitHub Models provider adapters."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Callable, Mapping
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.configuration import ProviderDefinition
from cognityx_inference.contracts import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
    TokenUsage,
)
from cognityx_inference.errors import (
    ProviderCredentialMissing,
    ProviderRequestError,
    UnsupportedProviderParameter,
)
from cognityx_inference.providers.models import ParameterPolicy, supplied_parameters
from cognityx_inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _provider_http_error,
)
from cognityx_inference.providers.policies import ANTHROPIC_POLICY, GEMINI_POLICY
from cognityx_inference.security import CredentialResolver
from cognityx_inference.token_budget import conservative_token_estimate


class _NativeProvider:
    policy: ParameterPolicy

    def __init__(
        self,
        definition: ProviderDefinition,
        resolver: CredentialResolver,
    ) -> None:
        self.name = definition.name
        self.base_url = (definition.base_url or "").rstrip("/")
        self.api_key_env = definition.api_key_env
        self.api_key_secret = definition.api_key_secret
        self.resolver = resolver
        self.model_capabilities = dict(definition.model_capabilities)
        self.api_version = definition.api_version

    def credential_available(self) -> bool:
        return self.resolver.available(
            environment_name=self.api_key_env,
            secret_name=self.api_key_secret,
        )

    def _credential(self) -> str:
        value = self.resolver.resolve(
            environment_name=self.api_key_env,
            secret_name=self.api_key_secret,
        )
        if not value:
            raise ProviderCredentialMissing(
                self.name,
                self.api_key_env or "configured provider credential",
                self.api_key_secret,
            )
        return value

    def context_profile(self, model: str) -> CertifiedContextProfile | None:
        return self.model_capabilities.get(model)

    def count_input_tokens(self, request: InferenceRequest) -> int | None:
        profile = self.context_profile(request.model)
        if profile is None or not profile.allow_estimated_counting:
            return None
        return conservative_token_estimate(request)

    def _validate(self, request: InferenceRequest) -> None:
        unsupported = sorted(supplied_parameters(request) - self.policy.supported)
        if unsupported:
            raise UnsupportedProviderParameter(self.name, unsupported)

    @staticmethod
    def _read_json(response: Any, provider: str) -> dict[str, Any]:
        try:
            value = json.loads(response.read())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderRequestError(provider, "invalid_response") from exc
        if not isinstance(value, dict):
            raise ProviderRequestError(provider, "invalid_response")
        return value


class GeminiNativeProvider(_NativeProvider):
    """Google Gemini generateContent adapter."""

    policy = GEMINI_POLICY
    capabilities = ModelCapabilities(
        streaming=True,
        stop_conditions=True,
        top_k=True,
        seed=True,
        tools=True,
        structured_output=True,
        vision=True,
    )

    def list_models(self, *, timeout_seconds: float = 20) -> tuple[str, ...]:
        outgoing = urllib_request.Request(
            f"{self.base_url}/models",
            headers={"x-goog-api-key": self._credential()},
        )
        try:
            with urllib_request.urlopen(outgoing, timeout=timeout_seconds) as response:
                body = self._read_json(response, self.name)
        except HTTPError as exc:
            raise _provider_http_error(self.name, exc) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "network_error") from exc
        return tuple(
            str(item["name"])
            for item in body.get("models", ())
            if isinstance(item, dict) and item.get("name")
        )

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> InferenceResponse:
        self._validate(request)
        payload = self._payload(request)
        model_path = (
            request.model
            if request.model.startswith("models/")
            else f"models/{request.model}"
        )
        action = "streamGenerateContent" if request.stream else "generateContent"
        suffix = "?alt=sse" if request.stream else ""
        outgoing = urllib_request.Request(
            f"{self.base_url}/{model_path}:{action}{suffix}",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._credential(),
            },
        )
        started = time.monotonic()
        try:
            with urllib_request.urlopen(
                outgoing, timeout=request.timeout_seconds or 120
            ) as response:
                request_id = response.headers.get("x-request-id", "")
                if request.stream:
                    return self._stream_response(
                        response, request, started, request_id, on_text
                    )
                body = self._read_json(response, self.name)
        except HTTPError as exc:
            raise _provider_http_error(self.name, exc) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "network_error") from exc
        return self._response(body, request, time.monotonic() - started, request_id)

    def _payload(self, request: InferenceRequest) -> dict[str, Any]:
        system = []
        contents = []
        messages = request.messages or (
            {"role": "user", "content": request.prompt or ""},
        )
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if role == "system":
                system.append({"text": str(content)})
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": str(content)}],
                }
            )
        generation = {
            "temperature": request.temperature,
            "topP": request.top_p,
            "topK": request.top_k,
            "maxOutputTokens": request.max_output_tokens,
            "stopSequences": list(request.stop) or None,
            "seed": request.seed,
            "responseMimeType": (
                "application/json" if request.response_format is not None else None
            ),
            "thinkingConfig": _gemini_thinking_config(request.reasoning),
        }
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                key: value for key, value in generation.items() if value is not None
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": system}
        if request.tools:
            payload["tools"] = _gemini_tools(request.tools)
        return payload

    def _response(
        self,
        body: dict[str, Any],
        request: InferenceRequest,
        elapsed: float,
        request_id: str,
    ) -> InferenceResponse:
        candidate = (body.get("candidates") or [{}])[0]
        content = _gemini_text(candidate)
        return InferenceResponse(
            request_id=request_id,
            content=content,
            model=request.model,
            provider=self.name,
            backend="gemini-native",
            finish_reason=_gemini_finish(candidate.get("finishReason")),
            usage=_gemini_usage(body.get("usageMetadata") or {}),
            timings=InferenceTimings(latency_seconds=elapsed),
            requested_parameters=request.to_dict(),
            effective_parameters={"native": "generateContent"},
            extensions={
                "provider_metadata": {"prompt_feedback": body.get("promptFeedback")}
            },
        )

    def _stream_response(
        self,
        response: Any,
        request: InferenceRequest,
        started: float,
        request_id: str,
        on_text: Callable[[str], None] | None,
    ) -> InferenceResponse:
        content: list[str] = []
        usage: dict[str, Any] = {}
        finish = FinishReason.UNKNOWN
        first: float | None = None
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError as exc:
                raise ProviderRequestError(self.name, "invalid_response") from exc
            candidate = (event.get("candidates") or [{}])[0]
            text = _gemini_text(candidate)
            if text:
                if first is None:
                    first = time.monotonic() - started
                content.append(text)
                if on_text:
                    on_text(text)
            if candidate.get("finishReason"):
                finish = _gemini_finish(candidate["finishReason"])
            usage.update(event.get("usageMetadata") or {})
        elapsed = time.monotonic() - started
        return InferenceResponse(
            request_id=request_id,
            content="".join(content),
            model=request.model,
            provider=self.name,
            backend="gemini-native",
            finish_reason=finish,
            usage=_gemini_usage(usage),
            timings=InferenceTimings(
                latency_seconds=elapsed,
                time_to_first_token_seconds=first,
                token_generation_seconds=elapsed,
            ),
            requested_parameters=request.to_dict(),
            effective_parameters={"native": "streamGenerateContent"},
            extensions={"streaming": True},
        )


class AnthropicProvider(_NativeProvider):
    """Anthropic Messages API adapter."""

    policy = ANTHROPIC_POLICY
    capabilities = ModelCapabilities(
        streaming=True,
        stop_conditions=True,
        top_k=True,
        tools=True,
        vision=True,
        reasoning=True,
    )

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._credential(),
            "anthropic-version": self.api_version or "2023-06-01",
        }

    def list_models(self, *, timeout_seconds: float = 20) -> tuple[str, ...]:
        outgoing = urllib_request.Request(
            f"{self.base_url}/models", headers=self._headers()
        )
        try:
            with urllib_request.urlopen(outgoing, timeout=timeout_seconds) as response:
                body = self._read_json(response, self.name)
        except HTTPError as exc:
            raise _provider_http_error(self.name, exc) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "network_error") from exc
        return tuple(
            str(item["id"])
            for item in body.get("data", ())
            if isinstance(item, dict) and item.get("id")
        )

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> InferenceResponse:
        self._validate(request)
        payload = self._payload(request)
        outgoing = urllib_request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        started = time.monotonic()
        try:
            with urllib_request.urlopen(
                outgoing, timeout=request.timeout_seconds or 120
            ) as response:
                request_id = response.headers.get("request-id", "")
                if request.stream:
                    return self._stream_response(
                        response, request, started, request_id, on_text
                    )
                body = self._read_json(response, self.name)
        except HTTPError as exc:
            raise _provider_http_error(self.name, exc) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "network_error") from exc
        content = "".join(
            str(item.get("text", ""))
            for item in body.get("content", ())
            if isinstance(item, dict) and item.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return InferenceResponse(
            request_id=str(body.get("id") or request_id),
            content=content,
            model=str(body.get("model") or request.model),
            provider=self.name,
            backend="anthropic",
            finish_reason=_anthropic_finish(body.get("stop_reason")),
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                total_tokens=_sum_usage(
                    usage.get("input_tokens"), usage.get("output_tokens")
                ),
            ),
            timings=InferenceTimings(latency_seconds=time.monotonic() - started),
            requested_parameters=request.to_dict(),
            effective_parameters={"native": "messages"},
            extensions={
                "provider_metadata": {"stop_sequence": body.get("stop_sequence")}
            },
        )

    def _payload(self, request: InferenceRequest) -> dict[str, Any]:
        systems = []
        messages = []
        request_messages = request.messages or (
            {"role": "user", "content": request.prompt or ""},
        )
        for message in request_messages:
            role = str(message.get("role", "user"))
            if role == "system":
                systems.append(str(message.get("content", "")))
            else:
                messages.append(
                    {
                        "role": "assistant" if role == "assistant" else "user",
                        "content": message.get("content", ""),
                    }
                )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens or 1024,
            "stream": request.stream,
        }
        optional = {
            "system": "\n\n".join(systems) or None,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop_sequences": list(request.stop) or None,
            "tools": list(request.tools) or None,
            "tool_choice": request.tool_choice,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return payload

    def _stream_response(
        self,
        response: Any,
        request: InferenceRequest,
        started: float,
        request_id: str,
        on_text: Callable[[str], None] | None,
    ) -> InferenceResponse:
        content: list[str] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        finish = FinishReason.UNKNOWN
        first: float | None = None
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError as exc:
                raise ProviderRequestError(self.name, "invalid_response") from exc
            event_type = event.get("type")
            if event_type == "message_start":
                input_tokens = (event.get("message", {}).get("usage") or {}).get(
                    "input_tokens"
                )
            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                text = str(delta.get("text") or "")
                if text:
                    if first is None:
                        first = time.monotonic() - started
                    content.append(text)
                    if on_text:
                        on_text(text)
            elif event_type == "message_delta":
                delta = event.get("delta") or {}
                finish = _anthropic_finish(delta.get("stop_reason"))
                output_tokens = (event.get("usage") or {}).get("output_tokens")
        elapsed = time.monotonic() - started
        return InferenceResponse(
            request_id=request_id,
            content="".join(content),
            model=request.model,
            provider=self.name,
            backend="anthropic",
            finish_reason=finish,
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=_sum_usage(input_tokens, output_tokens),
            ),
            timings=InferenceTimings(
                latency_seconds=elapsed,
                time_to_first_token_seconds=first,
                token_generation_seconds=elapsed,
            ),
            requested_parameters=request.to_dict(),
            effective_parameters={"native": "messages"},
            extensions={"streaming": True},
        )


class GitHubModelsProvider(OpenAICompatibleProvider):
    """Named adapter family for GitHub Models' OpenAI-compatible protocol."""

    models_url = "https://models.github.ai/catalog/models"

    @classmethod
    def from_definition(
        cls,
        definition: ProviderDefinition,
        *,
        credential_resolver: CredentialResolver | None = None,
    ) -> "GitHubModelsProvider":
        adapter = super().from_definition(
            replace(definition, adapter="openai_compatible"),
            credential_resolver=credential_resolver,
        )
        adapter.models_url = str(
            definition.options.get(
                "models_url", "https://models.github.ai/catalog/models"
            )
        )
        return adapter

    def list_models(self, *, timeout_seconds: float = 20) -> tuple[str, ...]:
        outgoing = urllib_request.Request(
            getattr(self, "models_url", "https://models.github.ai/catalog/models"),
            headers={
                "Authorization": f"Bearer {self._credential()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib_request.urlopen(outgoing, timeout=timeout_seconds) as response:
                value = json.loads(response.read())
        except HTTPError as exc:
            raise _provider_http_error(self.name, exc) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderRequestError(self.name, "invalid_response") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(self.name, "network_error") from exc
        items = (
            value
            if isinstance(value, list)
            else value.get("data", value.get("models", ()))
            if isinstance(value, dict)
            else ()
        )
        return tuple(
            str(item.get("id") or item.get("name"))
            for item in (items or ())
            if isinstance(item, dict) and (item.get("id") or item.get("name"))
        )

    def _outgoing(
        self, path: str, payload: dict[str, Any] | None, *, method: str
    ) -> urllib_request.Request:
        outgoing = super()._outgoing(path, payload, method=method)
        outgoing.add_header("Accept", "application/vnd.github+json")
        outgoing.add_header("X-GitHub-Api-Version", "2022-11-28")
        return outgoing


def _gemini_text(candidate: dict[str, Any]) -> str:
    content = candidate.get("content") or {}
    return "".join(
        str(part.get("text", ""))
        for part in content.get("parts", ())
        if isinstance(part, dict)
    )


def _gemini_thinking_config(reasoning: Mapping[str, Any]) -> dict[str, Any] | None:
    if reasoning.get("thinking_level") is not None:
        return {"thinkingLevel": str(reasoning["thinking_level"])}
    if reasoning.get("thinking_budget") is not None:
        return {"thinkingBudget": int(reasoning["thinking_budget"])}
    return None


def _gemini_usage(value: dict[str, Any]) -> TokenUsage:
    prompt = value.get("promptTokenCount")
    answer = value.get("candidatesTokenCount")
    thoughts = value.get("thoughtsTokenCount")
    completion = _sum_usage(answer, thoughts)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=value.get("totalTokenCount") or _sum_usage(prompt, completion),
    )


def _gemini_finish(value: Any) -> FinishReason:
    return {
        "STOP": FinishReason.STOP,
        "MAX_TOKENS": FinishReason.LENGTH,
        "SAFETY": FinishReason.CONTENT_FILTER,
    }.get(str(value or "").upper(), FinishReason.UNKNOWN)


def _anthropic_finish(value: Any) -> FinishReason:
    return {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALL,
    }.get(str(value or ""), FinishReason.UNKNOWN)


def _sum_usage(prompt: Any, completion: Any) -> int | None:
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt + completion
    return None


def _gemini_tools(tools: tuple[Any, ...]) -> list[dict[str, Any]]:
    declarations = []
    for item in tools:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        declarations.append(
            {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            }
        )
    return [{"functionDeclarations": declarations}] if declarations else []
