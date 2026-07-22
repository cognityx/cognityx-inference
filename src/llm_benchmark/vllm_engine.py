from __future__ import annotations

import gc
import time
from dataclasses import asdict
from typing import Any, Callable

from transformers import AutoConfig

from llm_benchmark.llm import (
    GenerationCancelled,
    GenerationSettings,
    build_quality_indicators,
    calculate_performance_metrics,
)
from llm_benchmark.loading_decision import ModelLoadOptions
from llm_benchmark.model_diagnostics import extract_architecture_metadata
from llm_benchmark.reporting import get_gpu_status


class VLLMEngineError(RuntimeError):
    """Raised when the optional vLLM runtime cannot satisfy a request."""


class VLLMLLM:
    """Small compatibility adapter around vLLM's persistent offline engine."""

    def __init__(
        self,
        model_name: str,
        settings: GenerationSettings,
        load_profile: str,
        load_options: ModelLoadOptions,
        *,
        kv_cache_dtype: str = "auto",
        max_model_len: int,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        if load_profile not in {"bf16", "fp16", "int4"}:
            raise VLLMEngineError(
                "The vLLM adapter currently supports bf16, fp16, and in-flight "
                "BitsAndBytes int4 profiles."
            )
        if load_options.cpu_offload_allowed:
            raise VLLMEngineError(
                "CPU offload is not enabled by this vLLM adapter; use the Transformers engine."
            )
        try:
            import vllm
            from vllm import LLM
        except ImportError as exc:
            raise VLLMEngineError(
                "vLLM is unavailable. Run this engine through .venv-vllm."
            ) from exc

        self.model_name = model_name
        self.settings = settings
        self.requested_load_profile = load_profile
        self.effective_load_profile = load_profile
        self.load_options = load_options
        self.kv_cache_dtype = kv_cache_dtype
        self.context_length_tokens = max_model_len
        self.vllm_version = vllm.__version__

        config = AutoConfig.from_pretrained(
            model_name, local_files_only=not load_options.allow_download
        )
        architecture = extract_architecture_metadata(config, None)
        declared_context_limit = architecture.get("max_position_embeddings")
        if (
            isinstance(declared_context_limit, (int, float))
            and max_model_len > int(declared_context_limit)
        ):
            raise VLLMEngineError(
                f"--vllm-max-model-len {max_model_len} exceeds the model's "
                f"declared context limit of {int(declared_context_limit)} tokens. "
                "Choose a value at or below that limit; the unsafe vLLM override "
                "is intentionally not enabled."
            )
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "dtype": "float16" if load_profile == "fp16" else "bfloat16",
            "max_model_len": max_model_len,
            "kv_cache_dtype": kv_cache_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": True,
        }
        if load_profile == "int4":
            kwargs.update(
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            )
        try:
            self.engine = LLM(**kwargs)
        except Exception as exc:
            cache_hint = (
                " while initializing FP8 KV cache"
                if kv_cache_dtype == "fp8"
                else ""
            )
            raise VLLMEngineError(
                f"vLLM failed to load {model_name!r}{cache_hint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.tokenizer = self.engine.get_tokenizer()
        load_seconds = time.perf_counter() - started
        self.processor = None
        self.model = self.engine
        self.model_load_metrics = {
            "tokenizer_seconds": None,
            "model_seconds": round(load_seconds, 3),
            "total_seconds": round(load_seconds, 3),
        }
        quantized = load_profile == "int4"
        self.load_profile_metadata = {
            "requested_load_profile": load_profile,
            "effective_load_profile": load_profile,
            "quantization_enabled": quantized,
            "quantization_bits": 4 if quantized else None,
            "quantization_method": "bitsandbytes" if quantized else None,
            "compute_dtype": kwargs["dtype"],
            "storage_dtype": "int4" if quantized else kwargs["dtype"],
        }
        self.quantization_resolution = {
            "engine": "vllm",
            "quantization_source": "runtime_bitsandbytes" if quantized else "explicit_dtype",
            "inflight_quantization": quantized,
        }
        self.loading_diagnostics = {
            **asdict(load_options),
            "capacity_classification": "gpu_only_inference",
            "gpu_only_capacity_pass": True,
            "uses_cpu_offload": False,
            "uses_disk_offload": False,
            "cpu_resident_modules": [],
            "gpu_resident_modules": ["vllm_engine"],
            "disk_offload_used": False,
        }
        self.model_runtime = {
            "engine": "vllm",
            "engine_version": self.vllm_version,
            "model_dtype": kwargs["dtype"],
            "devices_used": ["cuda:0"],
            "uses_cpu_offload": False,
            "uses_disk_offload": False,
            "attention_implementation": "vllm_selected",
            "kv_cache": {
                "use_cache": True,
                "cache_implementation": "vllm_paged_attention",
                "cache_dtype": kv_cache_dtype,
            },
            "architecture": architecture,
            "model_size": {"logical_parameters": architecture.get("total_parameters")},
            "model_family": {},
            "vllm": {
                "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "kv_cache_dtype": kv_cache_dtype,
                "wsl_legacy_runner": True,
                "flashinfer_sampler_disabled": True,
                "streaming_mode": "offline_delta",
                "kv_cache_scale_calibration": (
                    "not_provided" if kv_cache_dtype == "fp8" else None
                ),
            },
        }

    def unload_model(self) -> None:
        self.model = None
        if getattr(self, "engine", None) is not None:
            del self.engine
            self.engine = None
        gc.collect()

    def switch_model(self, *_args: Any, **_kwargs: Any) -> None:
        raise VLLMEngineError(
            "Interactive /model switching is not yet supported by the vLLM adapter; "
            "restart the isolated vLLM process for a different model."
        )

    @staticmethod
    def split_thinking_and_answer(text: str) -> tuple[str, str]:
        from llm_benchmark.llm import LocalLLM

        return LocalLLM.split_thinking_and_answer(text)

    def _prompt_token_ids(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.settings.enable_thinking,
            return_dict=False,
        )
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) for token_id in token_ids
        ):
            raise VLLMEngineError(
                "The tokenizer chat template did not return a flat list of token IDs."
            )
        return token_ids

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self._prompt_token_ids(prompt))

    def generate(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        run_type: str = "interactive",
        benchmark_name: str | None = None,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        prompt_started = time.perf_counter()
        prompt_ids = self._prompt_token_ids(prompt)
        prompt_preparation = time.perf_counter() - prompt_started
        if len(prompt_ids) + self.settings.max_new_tokens > self.context_length_tokens:
            raise VLLMEngineError(
                "Prompt plus max_new_tokens exceeds --vllm-max-model-len; input was not truncated."
            )
        sampling = SamplingParams(
            max_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature if self.settings.do_sample else 0.0,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            repetition_penalty=self.settings.repetition_penalty,
        )
        from vllm.sampling_params import RequestOutputKind

        sampling.output_kind = RequestOutputKind.DELTA
        engine_input = self.engine._preprocess_cmpl_one(
            {"prompt_token_ids": prompt_ids}
        )
        request_id = str(next(self.engine.request_counter))
        gpu_before = get_gpu_status()
        started = time.perf_counter()
        self.engine.llm_engine.add_request(
            request_id, engine_input, sampling, priority=0
        )
        output_parts: list[str] = []
        generated_ids: list[int] = []
        request = None
        first_streamed_output_seconds: float | None = None
        try:
            while self.engine.llm_engine.has_unfinished_requests():
                for current in self.engine.llm_engine.step():
                    if current.request_id != request_id or not current.outputs:
                        continue
                    request = current
                    delta = current.outputs[0]
                    output_parts.append(delta.text)
                    generated_ids.extend(delta.token_ids)
                    if delta.text:
                        if first_streamed_output_seconds is None:
                            first_streamed_output_seconds = (
                                time.perf_counter() - started
                            )
                        if on_text is not None:
                            on_text(delta.text)
        except KeyboardInterrupt as exc:
            self.engine.llm_engine.abort_request([request_id])
            raise GenerationCancelled(
                "Generation cancelled; the vLLM model remains loaded."
            ) from exc
        elapsed = time.perf_counter() - started
        if request is None or not request.finished:
            raise VLLMEngineError(
                "vLLM stopped without returning a completed request."
            )
        completion = request.outputs[0]
        raw_output = "".join(output_parts).strip()
        generated_tokens = len(generated_ids)
        metrics = request.metrics
        first_token_seconds = getattr(metrics, "first_token_latency", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        measured_generation = (
            last_token_ts - first_token_ts
            if first_token_ts and last_token_ts
            else elapsed
        )
        finish_reason = completion.finish_reason or "unknown"
        generation_truncated = finish_reason == "length"
        thinking, answer = self.split_thinking_and_answer(raw_output)
        quality = build_quality_indicators(
            self.tokenizer, thinking, answer, raw_output,
            finish_reason, generation_truncated,
        )
        gpu_after = get_gpu_status()
        performance = calculate_performance_metrics(
            prompt_tokenization_seconds=prompt_preparation,
            generation_seconds=elapsed,
            end_to_end_seconds=prompt_preparation + elapsed,
            prompt_tokens=len(prompt_ids),
            generated_tokens=generated_tokens,
            first_token_seconds=first_token_seconds,
            first_streamed_output_seconds=first_streamed_output_seconds,
        )
        return {
            "model": self.model_name,
            "engine": "vllm",
            "requested_load_profile": self.requested_load_profile,
            "effective_load_profile": self.effective_load_profile,
            **{key: self.load_profile_metadata[key] for key in (
                "quantization_enabled", "quantization_bits", "quantization_method",
                "compute_dtype", "storage_dtype",
            )},
            "quantization_resolution": self.quantization_resolution,
            "run_type": run_type,
            "benchmark_name": benchmark_name,
            "prompt": prompt,
            "settings": asdict(self.settings),
            "finish_reason": finish_reason,
            "generation_truncated": generation_truncated,
            "model_runtime": self.model_runtime,
            "quality_indicators": quality,
            "performance": performance,
            "result": {"raw_output": raw_output, "thinking": thinking, "answer": answer},
            "metrics": {
                "time_to_first_token_seconds": round(first_streamed_output_seconds, 3) if first_streamed_output_seconds is not None else None,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": generated_tokens,
                "total_tokens": len(prompt_ids) + generated_tokens,
                "total_context_length_tokens": self.context_length_tokens,
                "generation_seconds": round(elapsed, 3),
                "engine_reported_decode_seconds": round(measured_generation, 3),
                "tokens_per_second": round(generated_tokens / elapsed, 2) if elapsed else None,
                "peak_vram_gb": None,
                "gpu_allocated_before_gb": gpu_before["allocated_gb"],
                "gpu_reserved_before_gb": gpu_before["reserved_gb"],
                "gpu_allocated_after_gb": gpu_after["allocated_gb"],
                "gpu_reserved_after_gb": gpu_after["reserved_gb"],
                "gpu_memory_breakdown": None,
            },
        }
