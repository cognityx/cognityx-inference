"""Certified request serialization and output-token budgeting."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Callable

from cognityx_inference.capabilities import CertifiedContextProfile
from cognityx_inference.contracts import InferenceRequest, TokenBudget
from cognityx_inference.errors import (
    ContextWindowExceeded,
    TokenCountingUnavailable,
)


def serialized_request(request: InferenceRequest) -> str:
    """Serialize every input-bearing common request field deterministically."""
    value = {
        "messages": (
            list(request.messages)
            if request.messages
            else [{"role": "user", "content": request.prompt}]
        ),
        "tools": list(request.tools),
        "tool_choice": request.tool_choice,
        "response_format": request.response_format,
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def conservative_token_estimate(request: InferenceRequest) -> int:
    """Explicitly labelled conservative byte-based estimate."""
    return max(1, math.ceil(len(serialized_request(request).encode("utf-8")) / 3))


def apply_token_budget(
    request: InferenceRequest,
    profile: CertifiedContextProfile,
    count_tokens: Callable[[InferenceRequest], int | None],
) -> tuple[InferenceRequest, TokenBudget]:
    """Validate a request and inject its effective output-token limit."""
    input_tokens = count_tokens(request)
    counting = "tokenizer"
    if input_tokens is None:
        if not profile.allow_estimated_counting:
            raise TokenCountingUnavailable(
                "Token counting is unavailable and the capability profile "
                "does not permit an estimate."
            )
        input_tokens = conservative_token_estimate(request)
        counting = "conservative_estimate"
    if input_tokens < 0:
        raise ValueError("Token counter returned a negative value")

    requested = request.max_output_tokens
    available = (
        profile.max_context_tokens - input_tokens - profile.reserved_tokens
    )
    if requested is None:
        effective = min(
            available,
            (
                profile.max_output_tokens_limit
                if profile.max_output_tokens_limit is not None
                else available
            ),
        )
        source = "automatic"
    else:
        effective = requested
        source = "manual"
    if available <= 0 or effective <= 0:
        raise ContextWindowExceeded(
            "The serialized input and reserved tokens leave no positive "
            "certified output-token budget."
        )
    if input_tokens + effective + profile.reserved_tokens > profile.max_context_tokens:
        raise ContextWindowExceeded(
            "input_tokens + max_output_tokens + reserved_tokens exceeds "
            f"the certified context limit {profile.max_context_tokens}."
        )
    budget = TokenBudget(
        input_tokens=input_tokens,
        max_context_tokens=profile.max_context_tokens,
        reserved_tokens=profile.reserved_tokens,
        requested_max_output_tokens=requested,
        effective_max_output_tokens=effective,
        source=source,
        counting=counting,
    )
    return replace(request, max_output_tokens=effective, max_tokens=None), budget
