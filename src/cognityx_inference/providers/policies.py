"""Safe parameter presets for supported provider protocols."""

from __future__ import annotations

from cognityx_inference.providers.models import ParameterPolicy

_OPENAI_COMMON = frozenset(
    {
        "messages",
        "model",
        "temperature",
        "top_p",
        "max_output_tokens",
        "stop",
        "seed",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "response_format",
        "frequency_penalty",
        "presence_penalty",
        "parallel_tool_calls",
        "reasoning_effort",
        "service_tier",
    }
)


OPENAI_COMPATIBLE_POLICIES = {
    "openai": ParameterPolicy(
        supported=_OPENAI_COMMON,
        translated={"max_output_tokens": "max_completion_tokens"},
    ),
    "groq": ParameterPolicy(
        supported=_OPENAI_COMMON - {"logprobs", "top_logprobs", "service_tier"},
        translated={"max_output_tokens": "max_completion_tokens"},
        rejected_when_supplied=frozenset(
            {"logprobs", "top_logprobs", "logit_bias", "n"}
        ),
    ),
    "cerebras": ParameterPolicy(
        supported=_OPENAI_COMMON
        - {"logprobs", "top_logprobs", "reasoning_effort", "service_tier"},
        translated={"max_output_tokens": "max_completion_tokens"},
    ),
    "openrouter": ParameterPolicy(
        supported=_OPENAI_COMMON,
        translated={"max_output_tokens": "max_tokens"},
    ),
    "gemini_openai": ParameterPolicy(
        supported=_OPENAI_COMMON
        - {"logprobs", "top_logprobs", "frequency_penalty", "presence_penalty"},
        translated={"max_output_tokens": "max_tokens"},
    ),
    "cloudflare": ParameterPolicy(
        supported=_OPENAI_COMMON
        - {
            "logprobs",
            "top_logprobs",
            "reasoning_effort",
            "service_tier",
            "parallel_tool_calls",
        },
        translated={"max_output_tokens": "max_tokens"},
    ),
    "github_models": ParameterPolicy(
        supported=_OPENAI_COMMON - {"reasoning_effort", "service_tier"},
        translated={"max_output_tokens": "max_tokens"},
    ),
}


GEMINI_POLICY = ParameterPolicy(
    supported=frozenset(
        {
            "messages",
            "model",
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop",
            "seed",
            "tools",
            "response_format",
            "reasoning",
        }
    )
)


ANTHROPIC_POLICY = ParameterPolicy(
    supported=frozenset(
        {
            "messages",
            "model",
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop",
            "tools",
            "tool_choice",
        }
    )
)
