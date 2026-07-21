from __future__ import annotations

import gc
import importlib.util
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from llm_benchmark.model_diagnostics import (
    collect_model_runtime,
    extract_quantization_metadata,
)
from llm_benchmark.reporting import bytes_to_gb, get_gpu_status


@dataclass
class GenerationSettings:
    enable_thinking: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    do_sample: bool = True
    repetition_penalty: float = 1.0


LOAD_PROFILES = ("bf16", "fp16", "int8", "int4")


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


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def build_model_load_kwargs(
    profile: str,
    *,
    has_bitsandbytes: bool | None = None,
) -> tuple[dict[str, Any], torch.dtype]:
    profile = validate_load_profile(profile)
    if profile == "bf16":
        return {"torch_dtype": torch.bfloat16, "device_map": "auto"}, torch.bfloat16
    if profile == "fp16":
        return {"torch_dtype": torch.float16, "device_map": "auto"}, torch.float16

    available = bitsandbytes_available() if has_bitsandbytes is None else has_bitsandbytes
    if not available:
        raise LoadProfileError(
            f"The {profile} profile requires bitsandbytes. "
            "Install it with: uv add bitsandbytes"
        )

    if profile == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        compute_dtype = torch.float16
    else:
        bf16_supported = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    return {
        "quantization_config": quantization_config,
        "device_map": "auto",
    }, compute_dtype


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
    ) -> None:
        self.model_name = model_name
        self.requested_load_profile = validate_load_profile(load_profile)
        self.effective_load_profile = self.requested_load_profile
        self.load_profile_metadata: dict[str, Any] | None = None
        self.settings = settings
        self.tokenizer: Any = None
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

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load_model(self, model_name: str, load_profile: str = "bf16") -> None:
        load_profile = validate_load_profile(load_profile)
        load_kwargs, requested_compute_dtype = build_model_load_kwargs(load_profile)
        if self.model is not None:
            print(f"\nUnloading {self.model_name}...")
            self.unload_model()

        load_started = time.perf_counter()
        print(f"\nLoading tokenizer: {model_name}")
        tokenizer_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_seconds = time.perf_counter() - tokenizer_started

        print(f"Loading model: {model_name}")
        model_started = time.perf_counter()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.model.eval()

        model_seconds = time.perf_counter() - model_started
        total_seconds = time.perf_counter() - load_started
        self.model_name = model_name
        self.requested_load_profile = load_profile
        self.load_profile_metadata = extract_quantization_metadata(
            self.model,
            load_profile,
            requested_compute_dtype,
        )
        self.effective_load_profile = self.load_profile_metadata[
            "effective_load_profile"
        ]
        self.model_runtime = collect_model_runtime(
            self.model,
            self.load_profile_metadata,
        )
        self.model_load_metrics = {
            "tokenizer_seconds": round(tokenizer_seconds, 3),
            "model_seconds": round(model_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }

        print(f"Tokenizer loaded in {tokenizer_seconds:.2f} seconds")
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
        print(f"Device: {next(self.model.parameters()).device}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory reserved:  {reserved:.2f} GB")

    def switch_model(self, model_name: str, load_profile: str = "bf16") -> None:
        load_profile = validate_load_profile(load_profile)
        build_model_load_kwargs(load_profile)
        previous_model = self.model_name
        previous_profile = self.requested_load_profile
        try:
            self.load_model(model_name, load_profile)
        except Exception:
            self.unload_model()
            print(f"Attempting to restore {previous_model} ({previous_profile})...")
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

    def generate(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        run_type: str = "interactive",
        benchmark_name: str | None = None,
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        prompt_preparation_started = request_started
        messages = [{"role": "user", "content": prompt}]

        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.settings.enable_thinking,
        )

        inputs = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
        )
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
            "compute_dtype": self.load_profile_metadata["compute_dtype"],
            "storage_dtype": self.load_profile_metadata["storage_dtype"],
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
