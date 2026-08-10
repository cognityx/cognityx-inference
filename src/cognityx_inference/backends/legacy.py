"""Adapters around the proven llm-benchmark local engines."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import time
from typing import Any, Callable
import uuid

from cognityx_inference.contracts import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
    TokenUsage,
)
from cognityx_inference.errors import ModelMetadataUnavailableError
from cognityx_inference.errors import AdapterError
from cognityx_inference.token_budget import serialized_request
from llm_benchmark.llm import GenerationSettings, LocalLLM
from llm_benchmark.loading_decision import ModelLoadOptions
from llm_benchmark.vllm_engine import VLLMLLM
from transformers import AutoConfig


def _prompt(request: InferenceRequest) -> str:
    if request.prompt is not None:
        return request.prompt
    return "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in request.messages
    )


def _settings(request: InferenceRequest) -> GenerationSettings:
    defaults = GenerationSettings()
    values: dict[str, Any] = {}
    mapping = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "max_new_tokens": request.max_output_tokens,
    }
    for name, value in mapping.items():
        if value is not None:
            values[name] = value
    return replace(defaults, **values)


def _finish_reason(value: str | None) -> FinishReason:
    aliases = {"eos": FinishReason.STOP, "stop": FinishReason.STOP}
    try:
        return aliases.get(value or "", FinishReason(value or "unknown"))
    except ValueError:
        return FinishReason.UNKNOWN


class TransformersBackend:
    """Expose ``LocalLLM`` through the normalized backend contract."""

    capabilities = ModelCapabilities(
        streaming=True,
        lifecycle=True,
        top_k=True,
        seed=True,
        local_telemetry=True,
    )

    def __init__(
        self,
        model: str,
        runtime: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model
        self.runtime = dict(runtime or {})
        self.engine: LocalLLM | None = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self.engine is not None:
            return
        started = time.monotonic()
        profile = str(self.runtime.get("quantization", "bf16"))
        options = ModelLoadOptions(
            strict_gpu_only=bool(self.runtime.get("strict_gpu_only", False))
        )
        self.engine = LocalLLM(
            self.model_name,
            GenerationSettings(),
            profile,
            options,
        )
        self.load_seconds = time.monotonic() - started

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
        *,
        adapter: Any | None = None,
    ) -> InferenceResponse:
        self.load()
        assert self.engine is not None
        self.engine.settings = _settings(request)
        started = time.monotonic()
        result = self._generate(request, on_text=on_text, adapter=adapter)
        elapsed = time.monotonic() - started
        metrics = result.get("metrics", {})
        output = result.get("result", {})
        completion = metrics.get("generated_tokens")
        # The legacy Transformers result uses input_tokens while the vLLM
        # engine reports prompt_tokens. Preserve either without estimating.
        prompt_tokens = metrics.get("input_tokens", metrics.get("prompt_tokens"))
        return InferenceResponse(
            request_id=str(uuid.uuid4()),
            content=str(output.get("answer") or output.get("raw_output", "")),
            reasoning_content=output.get("thinking") or None,
            model=self.model_name,
            provider="local",
            backend="transformers",
            finish_reason=_finish_reason(result.get("finish_reason")),
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion,
                total_tokens=(
                    prompt_tokens + completion
                    if isinstance(prompt_tokens, int) and isinstance(completion, int)
                    else None
                ),
            ),
            timings=InferenceTimings(
                latency_seconds=elapsed,
                model_loading_seconds=self.load_seconds,
                prompt_processing_seconds=metrics.get("prompt_processing_seconds"),
                time_to_first_token_seconds=metrics.get(
                    "time_to_first_token_seconds"
                ),
                token_generation_seconds=metrics.get("generation_seconds"),
                tokens_per_second=metrics.get("tokens_per_second"),
            ),
            requested_parameters=request.to_dict(),
            effective_parameters=asdict(self.engine.settings),
            telemetry=result.get("metadata", {}),
            extensions={"legacy_result": result},
        )

    def _generate(
        self,
        request: InferenceRequest,
        *,
        on_text: Callable[[str], None] | None,
        adapter: Any | None,
    ) -> dict[str, Any]:
        if adapter is not None:
            raise AdapterError(
                "adapter_not_supported",
                "The Transformers backend does not support adapter selection",
            )
        assert self.engine is not None
        return self.engine.generate(_prompt(request), on_text=on_text)

    def unload(self) -> None:
        if self.engine is not None:
            self.engine.unload_model()
            self.engine = None

    def status(self) -> dict[str, Any]:
        return {"ready": self.engine is not None, "model": self.model_name}

    def model_context_limit(self) -> int | None:
        return discover_model_context_limit(
            self.model_name,
            allow_download=bool(self.runtime.get("allow_download", False)),
            revision=self.runtime.get("model_revision"),
        )[0]

    def count_input_tokens(self, request: InferenceRequest) -> int | None:
        """Count the complete common request with the loaded model tokenizer."""
        self.load()
        assert self.engine is not None
        tokenizer = getattr(self.engine, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            messages = (
                list(request.messages)
                if request.messages
                else [{"role": "user", "content": request.prompt}]
            )
            token_ids = tokenizer.apply_chat_template(
                messages,
                tools=list(request.tools) or None,
                tokenize=True,
                add_generation_prompt=True,
            )
            base = len(token_ids)
            metadata = {
                key: value
                for key, value in {
                    "tool_choice": request.tool_choice,
                    "response_format": request.response_format,
                }.items()
                if value is not None
            }
            if not metadata:
                return base
            metadata_tokens = tokenizer.encode(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                add_special_tokens=False,
            )
            return base + len(metadata_tokens)
        except (AttributeError, TypeError, ValueError):
            try:
                return len(
                    tokenizer.encode(
                        serialized_request(request),
                        add_special_tokens=False,
                    )
                )
            except (AttributeError, TypeError, ValueError):
                return None

    def runtime_identity(self) -> dict[str, Any]:
        self.load()
        assert self.engine is not None
        tokenizer = getattr(self.engine, "tokenizer", None)
        template = getattr(tokenizer, "chat_template", None)
        return {
            "name": self.model_name,
            "requested_revision": self.runtime.get("model_revision"),
            "resolved_revision": getattr(self.engine, "model_revision", None),
            "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
            "chat_template_checksum": (
                hashlib.sha256(template.encode("utf-8")).hexdigest()
                if isinstance(template, str)
                else None
            ),
        }


class VLLMBackend(TransformersBackend):
    """Expose the existing persistent vLLM engine."""

    capabilities = replace(TransformersBackend.capabilities, adapters=True)

    def load(self) -> None:
        if self.engine is not None:
            return
        max_model_len = self.runtime.get("context_length")
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise ValueError(
                "vLLM runtime.context_length must be a positive integer"
            )
        started = time.monotonic()
        self.engine = VLLMLLM(
            self.model_name,
            GenerationSettings(),
            str(self.runtime.get("quantization", "bf16")),
            ModelLoadOptions(
                strict_gpu_only=bool(self.runtime.get("strict_gpu_only", True))
            ),
            kv_cache_dtype=str(self.runtime.get("kv_cache_precision", "auto")),
            max_model_len=max_model_len,
            gpu_memory_utilization=float(
                self.runtime.get("gpu_memory_utilization", 0.9)
            ),
            enable_prefix_caching=bool(
                self.runtime.get("enable_prefix_caching", True)
            ),
            revision=self.runtime.get("model_revision"),
            tokenizer_revision=self.runtime.get("tokenizer_revision"),
            enable_lora=True,
        )
        self.load_seconds = time.monotonic() - started

    def infer(
        self,
        request: InferenceRequest,
        on_text: Callable[[str], None] | None = None,
        *,
        adapter: Any | None = None,
    ) -> InferenceResponse:
        response = super().infer(request, on_text, adapter=adapter)
        return replace(response, backend="vllm")

    def _generate(
        self,
        request: InferenceRequest,
        *,
        on_text: Callable[[str], None] | None,
        adapter: Any | None,
    ) -> dict[str, Any]:
        assert isinstance(self.engine, VLLMLLM)
        lora_request = None
        if adapter is not None:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:
                raise AdapterError(
                    "adapter_not_supported", "The vLLM LoRA API is unavailable"
                ) from exc
            lora_request = LoRARequest(
                adapter.adapter_id,
                adapter.lora_int_id,
                str(adapter.local_path),
                base_model_name=self.model_name,
            )
        return self.engine.generate(
            _prompt(request), on_text=on_text, lora_request=lora_request
        )


def discover_model_context_limit(
    model: str,
    *,
    allow_download: bool = False,
    revision: str | None = None,
) -> tuple[int | None, str | None]:
    """Read model-declared context and revision without loading weights."""
    try:
        config = AutoConfig.from_pretrained(
            model,
            local_files_only=not allow_download,
            revision=revision,
        )
    except OSError as exc:
        raise ModelMetadataUnavailableError(str(exc)) from exc
    sources = (config, getattr(config, "text_config", None))
    for source in sources:
        if source is None:
            continue
        for name in (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
        ):
            value = getattr(source, name, None)
            if isinstance(value, int) and value > 0:
                return value, getattr(config, "_commit_hash", None)
    return None, getattr(config, "_commit_hash", None)
