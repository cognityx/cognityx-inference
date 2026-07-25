"""Adapters for commercial OpenAI-compatible chat APIs."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib import request as urllib_request

from cognityx_inference.contracts import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
    TokenDetail,
    TokenUsage,
)


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible endpoint using the standard library."""

    capabilities = ModelCapabilities(
        streaming=False,
        stop_conditions=True,
        seed=True,
        log_probabilities=True,
        reasoning=True,
    )

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        *,
        default_timeout_seconds: float = 120,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_timeout_seconds = default_timeout_seconds

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.stream:
            raise ValueError(
                f"{self.name} streaming is not implemented by this adapter yet"
            )
        if request.top_k is not None or request.min_p is not None:
            raise ValueError(
                f"{self.name} does not expose top_k/min_p through its common API"
            )
        payload = self._payload(request)
        encoded = json.dumps(payload).encode("utf-8")
        outgoing = urllib_request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        with urllib_request.urlopen(
            outgoing,
            timeout=request.timeout_seconds or self.default_timeout_seconds,
        ) as response:
            body = json.loads(response.read())
        elapsed = time.monotonic() - started
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = body.get("usage", {})
        details = self._token_details(choice)
        reason = choice.get("finish_reason") or "unknown"
        try:
            finish_reason = FinishReason(reason)
        except ValueError:
            finish_reason = FinishReason.UNKNOWN
        return InferenceResponse(
            request_id=str(body.get("id", "")),
            content=str(message.get("content") or ""),
            reasoning_content=message.get("reasoning_content"),
            model=str(body.get("model") or request.model),
            provider=self.name,
            backend="commercial-api",
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            timings=InferenceTimings(latency_seconds=elapsed),
            token_details=details,
            requested_parameters=request.to_dict(),
            effective_parameters=payload,
            extensions={
                "system_fingerprint": body.get("system_fingerprint"),
                "provider_response": body,
            },
        )

    @staticmethod
    def _payload(request: InferenceRequest) -> dict[str, Any]:
        messages = (
            list(request.messages)
            if request.messages
            else [{"role": "user", "content": request.prompt}]
        )
        payload: dict[str, Any] = {"model": request.model, "messages": messages}
        optional = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "logprobs": request.log_probabilities or None,
            "top_logprobs": request.top_log_probabilities,
            "stop": list(request.stop) or None,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        payload.update(request.extensions)
        return payload

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
    """OpenAI commercial API adapter."""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(
            "openai", "https://api.openai.com/v1", api_key, **kwargs
        )


class XAIProvider(OpenAICompatibleProvider):
    """xAI/Grok commercial API adapter."""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__("xai", "https://api.x.ai/v1", api_key, **kwargs)
