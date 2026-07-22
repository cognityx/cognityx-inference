from __future__ import annotations

import gc
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)

from llm_benchmark.model_diagnostics import (
    collect_model_runtime,
    extract_quantization_metadata,
)
from llm_benchmark.model_loading import (
    ModelPlacementError,
    attempted_load_configuration,
    wrap_expected_model_load_failure,
)
from llm_benchmark.model_selection import select_model_loader
from llm_benchmark.quantization import (
    BitsAndBytesUnavailableError,
    bitsandbytes_available,
    detect_loaded_native_quantization_runtime,
    resolve_load_profile,
)
from llm_benchmark.reporting import bytes_to_gb, get_gpu_status
from llm_benchmark.loading_decision import ModelLoadOptions, load_diagnostics, process_ram_gib


@dataclass
class GenerationSettings:
    enable_thinking: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    do_sample: bool = True
    repetition_penalty: float = 1.0


LOAD_PROFILES = ("bf16", "fp16", "int8", "int4", "int4-double", "gptq3", "gptq2", "native", "auto")


class LoadProfileError(ValueError):
    pass


class TimingTextIteratorStreamer(TextIteratorStreamer):
    """Text streamer that timestamps the first raw generated token."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.first_generated_token_at: float | None = None

    def put(self, value: Any) -> None:
        is_prompt = self.skip_prompt and self.next_tokens_are_prompt
        if not is_prompt and self.first_generated_token_at is None:
            self.first_generated_token_at = time.perf_counter()
        super().put(value)


def validate_load_profile(profile: str) -> str:
    normalized = profile.lower()
    if normalized not in LOAD_PROFILES:
        raise LoadProfileError(
            f"Unknown load profile: {profile}. Choose from: {', '.join(LOAD_PROFILES)}"
        )
    return normalized


def build_model_load_kwargs(
    profile: str,
    *,
    has_bitsandbytes: bool | None = None,
) -> tuple[dict[str, Any], torch.dtype]:
    """Build legacy kwargs for an unquantized checkpoint."""
    profile = validate_load_profile(profile)
    if profile == "native":
        raise LoadProfileError("The native profile requires checkpoint configuration")

    class UnquantizedConfig:
        quantization_config = None

    try:
        resolution = resolve_load_profile(
            profile,
            UnquantizedConfig(),
            has_bitsandbytes=has_bitsandbytes,
        )
    except BitsAndBytesUnavailableError as exc:
        raise LoadProfileError(str(exc)) from exc
    if resolution.compute_dtype is None:
        raise LoadProfileError(f"Could not determine a compute dtype for {profile}")
    return resolution.load_kwargs(), resolution.compute_dtype


def count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_quality_indicators(
    tokenizer: Any,
    thinking: str,
    answer: str,
    raw_output: str,
    finish_reason: str,
    generation_truncated: bool,
) -> dict[str, Any]:
    return {
        "answer_complete": not generation_truncated,
        "reasoning_detected": bool(thinking.strip()),
        "thinking_characters": len(thinking),
        "thinking_tokens": count_text_tokens(tokenizer, thinking),
        "answer_characters": len(answer),
        "answer_tokens": count_text_tokens(tokenizer, answer),
        "raw_output_characters": len(raw_output),
        "empty_answer": not bool(answer.strip()),
        "finish_reason": finish_reason,
        "generation_truncated": generation_truncated,
    }


def calculate_performance_metrics(
    *,
    prompt_tokenization_seconds: float,
    generation_seconds: float,
    end_to_end_seconds: float,
    prompt_tokens: int,
    generated_tokens: int,
    first_token_seconds: float | None,
    first_streamed_output_seconds: float | None,
) -> dict[str, Any]:
    """Calculate prefill/decode approximations around streamed generation."""
    prefill_seconds = first_token_seconds
    decode_seconds = (
        max(0.0, generation_seconds - first_token_seconds)
        if first_token_seconds is not None
        else None
    )
    decode_tokens = max(0, generated_tokens - 1)
    return {
        "prompt_tokenization_seconds": round(prompt_tokenization_seconds, 3),
        "prefill_seconds": round(prefill_seconds, 3) if prefill_seconds is not None else None,
        "prefill_measurement_method": "generation_start_to_first_raw_streamer_token",
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": (
            round(prompt_tokens / prefill_seconds, 2)
            if prefill_seconds is not None and prefill_seconds > 0
            else None
        ),
        "time_to_first_token_seconds": (
            round(first_token_seconds, 3) if first_token_seconds is not None else None
        ),
        "first_token_latency_after_prompt_preparation_seconds": (
            round(first_token_seconds, 3) if first_token_seconds is not None else None
        ),
        "time_to_first_streamed_output_seconds": (
            round(first_streamed_output_seconds, 3)
            if first_streamed_output_seconds is not None
            else None
        ),
        "decode_seconds": round(decode_seconds, 3) if decode_seconds is not None else None,
        "decode_token_count": decode_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens_per_second": (
            round(decode_tokens / decode_seconds, 2)
            if decode_seconds is not None and decode_seconds > 0
            else 0.0 if decode_tokens == 0 else None
        ),
        "generation_seconds": round(generation_seconds, 3),
        "end_to_end_seconds": round(end_to_end_seconds, 3),
        "decode_measurement_method": (
            "generation_finish_minus_first_raw_token; throughput excludes the first token"
        ),
    }


def detect_finish_reason(
    generated_token_ids: list[int],
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> tuple[str, bool]:
    if any(token_id in eos_token_ids for token_id in generated_token_ids):
        return "eos", False

    if len(generated_token_ids) >= max_new_tokens:
        return "length", True

    return "unknown", False


class LocalLLM:
    def __init__(
        self,
        model_name: str,
        settings: GenerationSettings,
        load_profile: str = "bf16",
        load_options: ModelLoadOptions | None = None,
    ) -> None:
        self.model_name = model_name
        self.requested_load_profile = validate_load_profile(load_profile)
        self.effective_load_profile = self.requested_load_profile
        self.load_profile_metadata: dict[str, Any] | None = None
        self.settings = settings
        self.load_options = load_options or ModelLoadOptions()
        self.tokenizer: Any = None
        self.processor: Any = None
        self.model: Any = None
        self.model_load_metrics: dict[str, float] | None = None
        self.model_runtime: dict[str, Any] | None = None
        self.load_model(model_name, self.requested_load_profile)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load_model(self, model_name: str, load_profile: str = "bf16") -> None:
        load_profile = validate_load_profile(load_profile)
        attempted = attempted_load_configuration(load_profile)
        if self.model is not None:
            print(f"\nUnloading {self.model_name}...")
            self.unload_model()

        load_started = time.perf_counter()
        print("\nLoading configuration...")
        try:
            config = AutoConfig.from_pretrained(model_name, local_files_only=not self.load_options.allow_download)
        except Exception as exc:
            wrapped = wrap_expected_model_load_failure(
                exc,
                model_name,
                load_profile,
                attempted,
                loading_stage="configuration_download",
            )
            if wrapped is not None:
                raise wrapped from exc
            raise

        print("Selecting model architecture...")
        try:
            loader_selection = select_model_loader(config)
        except Exception as exc:
            loader_diagnostic = {
                "config_class": type(config).__name__,
                "task": "unknown",
                "auto_model_class": None,
                "selection_source": "no_supported_mapping",
                "text_only_execution": True,
            }
            wrapped = wrap_expected_model_load_failure(
                exc,
                model_name,
                load_profile,
                attempted,
                loader_diagnostic,
                "architecture_selection",
            )
            if wrapped is not None:
                raise wrapped from exc
            raise
        loader_diagnostic = loader_selection.diagnostic(config)
        print(f"Selected loader: {loader_diagnostic['auto_model_class']}")

        print("Loading tokenizer/processor...")
        print(f"Tokenizer/processor source: {model_name}")
        tokenizer_started = time.perf_counter()
        try:
            if loader_selection.uses_processor:
                self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=not self.load_options.allow_download)
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                if self.tokenizer is None:
                    raise ValueError(
                        "The selected multimodal processor does not expose a tokenizer "
                        "required for text-only streaming and token accounting."
                    )
            else:
                self.processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not self.load_options.allow_download)
        except Exception as exc:
            wrapped = wrap_expected_model_load_failure(
                exc, model_name, load_profile, attempted,
                loader_diagnostic, "preprocessing_download"
            )
            if wrapped is not None:
                raise wrapped from exc
            raise
        tokenizer_seconds = time.perf_counter() - tokenizer_started

        print("Resolving quantization...")
        try:
            resolution = resolve_load_profile(load_profile, config)
        except Exception as exc:
            wrapped = wrap_expected_model_load_failure(
                exc,
                model_name,
                load_profile,
                attempted,
                loader_diagnostic,
                "quantization_resolution",
            )
            if wrapped is not None:
                raise wrapped from exc
            raise
        load_kwargs = resolution.load_kwargs()
        load_kwargs["local_files_only"] = not self.load_options.allow_download
        if self.load_options.cpu_offload_allowed:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = self.load_options.max_memory
            quant_config = load_kwargs.get("quantization_config")
            if load_profile == "int8" and quant_config is not None:
                quant_config.llm_int8_enable_fp32_cpu_offload = True
        elif self.load_options.strict_gpu_only:
            load_kwargs["device_map"] = {"": 0}
        resolution_preload_diagnostic = resolution.diagnostic()
        native_runtime_preflight = resolution_preload_diagnostic[
            "native_quantization_runtime_preflight"
        ]
        requested_compute_dtype = resolution.compute_dtype
        attempted = attempted_load_configuration(
            load_profile,
            requested_compute_dtype,
            load_kwargs,
            resolution.diagnostic(),
        )
        print(f"Requested profile: {load_profile}")
        print(
            "Checkpoint quantization: "
            f"{resolution.checkpoint_quantization_method or 'none'}"
        )
        print(f"Effective profile: {resolution.effective_profile}")
        print(
            "Runtime bitsandbytes quantization: "
            f"{'enabled' if resolution.runtime_quantization_method == 'bitsandbytes' else 'none'}"
        )
        if resolution.warning:
            print(f"Warning: {resolution.warning}")
        if resolution.checkpoint_quantized:
            print(
                "Native quantization runtime preflight: "
                f"{native_runtime_preflight['status']}"
            )
            if native_runtime_preflight.get("reason"):
                print(f"Preflight reason: {native_runtime_preflight['reason']}")

        print("Loading model weights...")
        model_started = time.perf_counter()
        cpu_memory_before = process_ram_gib()
        gpu_memory_before = get_gpu_status()

        try:
            self.model = loader_selection.auto_model_class.from_pretrained(
                model_name,
                config=config,
                **load_kwargs,
            )
        except Exception as exc:
            wrapped = wrap_expected_model_load_failure(
                exc, model_name, load_profile, attempted,
                loader_diagnostic, "model_initialization"
            )
            if wrapped is not None:
                if wrapped.failure_category == "architecture":
                    print("Model architecture selection failed.")
                raise wrapped from exc
            raise
        self.model.eval()
        cpu_memory_after = process_ram_gib()
        gpu_memory_after = get_gpu_status()
        device_map = dict(getattr(self.model, "hf_device_map", {}) or {})
        placement = load_diagnostics(self.load_options, cpu_memory_before, cpu_memory_after, device_map)
        placement["gpu_memory_before"] = gpu_memory_before
        placement["gpu_memory_after"] = gpu_memory_after
        if self.load_options.strict_gpu_only and (placement["uses_cpu_offload"] or placement["disk_offload_used"]):
            raise ModelPlacementError(model_name, load_profile, RuntimeError("Strict GPU-only placement contains CPU or disk modules"), "cpu_offload_required", attempted)
        native_runtime = detect_loaded_native_quantization_runtime(
            self.model,
            resolution.checkpoint_quantization_method,
            native_runtime_preflight,
        )
        resolution_diagnostic = resolution.diagnostic()
        if load_profile in {"gptq3", "gptq2"}:
            checkpoint_quantization = getattr(config, "quantization_config", None)
            resolution_diagnostic["gptq_checkpoint"] = {
                "bits": checkpoint_quantization.get("bits") if isinstance(checkpoint_quantization, dict) else getattr(checkpoint_quantization, "bits", None),
                "group_size": checkpoint_quantization.get("group_size") if isinstance(checkpoint_quantization, dict) else getattr(checkpoint_quantization, "group_size", None),
                "backend": "gptqmodel_or_auto_gptq",
                "human_inspection_recommended": True,
            }
        resolution_diagnostic["native_quantization_runtime"] = native_runtime
        resolution_diagnostic["bf16_dequantization_fallback_attempted"] = (
            native_runtime.get("runtime_fallback") == "bf16"
        )
        print("Inspecting device map...")
        print("Validating placement...")
        print("Placement validated.")

        model_seconds = time.perf_counter() - model_started
        total_seconds = time.perf_counter() - load_started
        self.model_name = model_name
        self.requested_load_profile = load_profile
        self.load_profile_metadata = extract_quantization_metadata(
            self.model,
            resolution.effective_profile,
            requested_compute_dtype,
        )
        self.load_profile_metadata["requested_load_profile"] = load_profile
        self.load_profile_metadata["effective_load_profile"] = resolution.effective_profile
        if resolution.checkpoint_quantized:
            native_method = resolution.checkpoint_quantization_method
            embedded_quantization = getattr(self.model.config, "quantization_config", None)
            embedded_bits = (
                embedded_quantization.get("bits")
                if isinstance(embedded_quantization, dict)
                else getattr(embedded_quantization, "bits", None)
            )
            native_bits = 4 if native_method == "mxfp4" else embedded_bits
            model_config = getattr(self.model, "config", None)
            native_compute_dtype = (
                getattr(model_config, "dtype", None)
                or getattr(model_config, "torch_dtype", None)
            )
            self.load_profile_metadata.update(
                {
                    "quantization_enabled": True,
                    "quantization_bits": native_bits,
                    "quantization_method": native_method,
                    "storage_dtype": native_method,
                    "compute_dtype": (
                        str(native_compute_dtype)
                        if native_compute_dtype is not None
                        else self.load_profile_metadata.get("compute_dtype")
                    ),
                }
            )
            if native_runtime.get("runtime_fallback") == "bf16":
                self.load_profile_metadata.update(
                    {
                        "quantization_enabled": False,
                        "quantization_bits": None,
                        "quantization_method": None,
                        "storage_dtype": "torch.bfloat16",
                        "compute_dtype": "torch.bfloat16",
                    }
                )
        self.effective_load_profile = self.load_profile_metadata[
            "effective_load_profile"
        ]
        self.model_runtime = collect_model_runtime(
            self.model,
            self.load_profile_metadata,
        )
        self.model_runtime["model_loader"] = loader_diagnostic
        self.model_runtime["quantization_resolution"] = resolution_diagnostic
        self.model_runtime["loading"] = placement
        self.loading_diagnostics = placement
        self.quantization_resolution = resolution_diagnostic
        architecture = self.model_runtime.get("architecture", {})
        context_length = architecture.get("max_position_embeddings")
        self.context_length_tokens = (
            context_length
            if isinstance(context_length, int) and context_length > 0
            else None
        )
        self.model_load_metrics = {
            "tokenizer_seconds": round(tokenizer_seconds, 3),
            "model_seconds": round(model_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }

        print(f"Tokenizer/processor loaded in {tokenizer_seconds:.2f} seconds")
        print(f"Model loaded in {model_seconds:.2f} seconds")
        print(f"Total load time: {total_seconds:.2f} seconds")
        print(f"Requested profile: {self.requested_load_profile}")
        print(f"Effective profile: {self.effective_load_profile}")
        print(
            "Quantization: "
            f"{self.load_profile_metadata['quantization_method'] or 'disabled'}"
        )
        print(f"Quantization bits: {self.load_profile_metadata['quantization_bits']}")
        print(f"Compute dtype: {self.load_profile_metadata['compute_dtype']}")
        print(f"Storage dtype: {self.load_profile_metadata['storage_dtype']}")
        if resolution.checkpoint_quantized:
            print(f"Native quantization runtime: {native_runtime['status']}")
            print(
                "Runtime backend: "
                f"{native_runtime.get('runtime_backend') or 'unavailable'}"
            )
            print(
                "Runtime fallback: "
                f"{native_runtime.get('runtime_fallback') or 'none'}"
            )
            if native_runtime.get("reason"):
                print(f"Runtime reason: {native_runtime['reason']}")
            if native_runtime.get("expected_performance_impact"):
                print(
                    "Expected performance impact: "
                    f"{native_runtime['expected_performance_impact']}"
                )
        print(f"Device: {next(self.model.parameters()).device}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory reserved:  {reserved:.2f} GB")

    def switch_model(
        self,
        model_name: str,
        load_profile: str = "bf16",
        load_options: ModelLoadOptions | None = None,
    ) -> None:
        load_profile = validate_load_profile(load_profile)
        previous_model = self.model_name
        previous_profile = self.requested_load_profile
        previous_options = self.load_options
        if load_options is not None:
            self.load_options = load_options
        try:
            self.load_model(model_name, load_profile)
        except ModelPlacementError:
            self.unload_model()
            print(f"Attempting to restore {previous_model} ({previous_profile})...")
            self.load_options = previous_options
            self.load_model(previous_model, previous_profile)
            raise

    @staticmethod
    def split_thinking_and_answer(text: str) -> tuple[str, str]:
        """
        Qwen3 may emit:
            reasoning text </think> final answer

        Depending on the chat template, an explicit opening <think> token
        may not appear in the decoded generated portion.
        """
        cleaned = text.strip()

        if "</think>" in cleaned:
            thinking, answer = cleaned.split("</think>", maxsplit=1)
            thinking = thinking.replace("<think>", "", 1).strip()
            return thinking, answer.strip()

        if cleaned.startswith("<think>"):
            return cleaned.removeprefix("<think>").strip(), ""

        return "", cleaned

    def _prepare_text_inputs(self, prompt: str) -> Any:
        """Render and encode a text-only chat for the selected model family."""
        if self.processor is None:
            messages = [{"role": "user", "content": prompt}]
            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.settings.enable_thinking,
            )
            return self.tokenizer(rendered_prompt, return_tensors="pt")

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        template_owner = (
            self.processor
            if callable(getattr(self.processor, "apply_chat_template", None))
            else self.tokenizer
        )
        if not callable(getattr(template_owner, "apply_chat_template", None)):
            raise ValueError(
                "The multimodal processor/tokenizer has no chat template for "
                "verified text-only prompt preparation."
            )
        rendered_prompt = template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.processor(text=rendered_prompt, return_tensors="pt")

    def generate(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        run_type: str = "interactive",
        benchmark_name: str | None = None,
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        prompt_preparation_started = request_started
        inputs = self._prepare_text_inputs(prompt)
        prompt_tokenization_seconds = time.perf_counter() - prompt_preparation_started
        inputs = inputs.to(self.model.device)

        prompt_tokens = inputs.input_ids.shape[1]

        gpu_before = get_gpu_status()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.do_sample,
            "repetition_penalty": self.settings.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.settings.do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.settings.temperature,
                    "top_p": self.settings.top_p,
                    "top_k": self.settings.top_k,
                }
            )

        streamer = TimingTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_result: dict[str, Any] = {}

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    generation_result["outputs"] = self.model.generate(
                        **inputs,
                        **generation_kwargs,
                        streamer=streamer,
                    )
            except BaseException as exc:
                generation_result["error"] = exc
                streamer.end()

        started = time.perf_counter()
        generation_thread = threading.Thread(target=run_generation)
        generation_thread.start()

        first_output_at: float | None = None
        for text in streamer:
            if text and first_output_at is None:
                first_output_at = time.perf_counter()
            if on_text is not None:
                on_text(text)

        generation_thread.join()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - started
        first_token_seconds = (
            streamer.first_generated_token_at - started
            if streamer.first_generated_token_at is not None
            else None
        )
        first_streamed_output_seconds = (
            first_output_at - started if first_output_at is not None else None
        )

        if "error" in generation_result:
            raise generation_result["error"]

        outputs = generation_result["outputs"]
        generated_ids = outputs[0, prompt_tokens:]
        generated_tokens = generated_ids.shape[0]
        generated_token_ids = generated_ids.tolist()

        configured_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if configured_eos is None:
            configured_eos = self.tokenizer.eos_token_id
        if isinstance(configured_eos, int):
            eos_token_ids = {configured_eos}
        else:
            eos_token_ids = set(configured_eos or [])

        finish_reason, generation_truncated = detect_finish_reason(
            generated_token_ids,
            eos_token_ids,
            self.settings.max_new_tokens,
        )

        raw_output = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        thinking, answer = self.split_thinking_and_answer(raw_output)
        quality_indicators = build_quality_indicators(
            self.tokenizer,
            thinking,
            answer,
            raw_output,
            finish_reason,
            generation_truncated,
        )
        gpu_after = get_gpu_status()
        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = bytes_to_gb(torch.cuda.max_memory_allocated())
        end_to_end_seconds = time.perf_counter() - request_started
        performance = calculate_performance_metrics(
            prompt_tokenization_seconds=prompt_tokenization_seconds,
            generation_seconds=elapsed,
            end_to_end_seconds=end_to_end_seconds,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            first_token_seconds=first_token_seconds,
            first_streamed_output_seconds=first_streamed_output_seconds,
        )

        return {
            "model": self.model_name,
            "requested_load_profile": self.requested_load_profile,
            "effective_load_profile": self.effective_load_profile,
            "quantization_enabled": self.load_profile_metadata[
                "quantization_enabled"
            ],
            "quantization_bits": self.load_profile_metadata["quantization_bits"],
            "quantization_method": self.load_profile_metadata[
                "quantization_method"
            ],
            "quantization_type": "nf4" if self.requested_load_profile in {"int4", "int4-double"} else None,
            "compute_dtype": self.load_profile_metadata["compute_dtype"],
            "storage_dtype": self.load_profile_metadata["storage_dtype"],
            "quantization_resolution": self.quantization_resolution,
            "download_preflight": self.loading_diagnostics.get("download_preflight"),
            "cached_checkpoint": (self.loading_diagnostics.get("download_preflight") or {}).get("cached_checkpoint"),
            "cached_bytes": (self.loading_diagnostics.get("download_preflight") or {}).get("cached_bytes"),
            "missing_download_bytes": (self.loading_diagnostics.get("download_preflight") or {}).get("missing_download_bytes"),
            "loading_choice": self.loading_diagnostics["loading_choice"],
            "capacity_classification": self.loading_diagnostics["capacity_classification"],
            "gpu_only_capacity_pass": self.loading_diagnostics["gpu_only_capacity_pass"],
            "nested_quantization": self.requested_load_profile == "int4-double",
            "cpu_offload_allowed": self.loading_diagnostics["cpu_offload_allowed"],
            "cpu_resident_modules": self.loading_diagnostics["cpu_resident_modules"],
            "cpu_memory_before": self.loading_diagnostics["cpu_memory_before_gib"],
            "cpu_memory_after": self.loading_diagnostics["cpu_memory_after_gib"],
            "cpu_memory_peak": self.loading_diagnostics["cpu_memory_peak_gib"],
            "disk_offload_allowed": self.loading_diagnostics["disk_offload_allowed"],
            "disk_offload_used": self.loading_diagnostics["disk_offload_used"],
            "quantized_checkpoint_model_id": self.loading_diagnostics["quantized_checkpoint_model_id"],
            "user_confirmations": self.loading_diagnostics["user_confirmations"],
            "run_type": run_type,
            "benchmark_name": benchmark_name,
            "prompt": prompt,
            "settings": asdict(self.settings),
            "finish_reason": finish_reason,
            "generation_truncated": generation_truncated,
            "model_runtime": self.model_runtime,
            "quality_indicators": quality_indicators,
            "performance": performance,
            "result": {
                "raw_output": raw_output,
                "thinking": thinking,
                "answer": answer,
            },
            "metrics": {
                "time_to_first_token_seconds": (
                    round(first_output_at - started, 3)
                    if first_output_at is not None
                    else None
                ),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
                "total_context_length_tokens": self.context_length_tokens,
                "generation_seconds": round(elapsed, 3),
                "tokens_per_second": (
                    round(generated_tokens / elapsed, 2)
                    if elapsed > 0
                    else None
                ),
                "peak_vram_gb": (
                    peak_vram_gb if peak_vram_gb is not None else None
                ),
                "gpu_allocated_before_gb": gpu_before["allocated_gb"],
                "gpu_reserved_before_gb": gpu_before["reserved_gb"],
                "gpu_allocated_after_gb": gpu_after["allocated_gb"],
                "gpu_reserved_after_gb": gpu_after["reserved_gb"],
            },
        }
