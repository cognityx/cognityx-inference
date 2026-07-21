from __future__ import annotations

import gc
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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
    def __init__(self, model_name: str, settings: GenerationSettings) -> None:
        self.model_name = model_name
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
        self.model_load_metrics: dict[str, float] | None = None
        self.load_model(model_name)

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

    def load_model(self, model_name: str) -> None:
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
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

        self.model_name = model_name
        model_seconds = time.perf_counter() - model_started
        total_seconds = time.perf_counter() - load_started
        self.model_load_metrics = {
            "tokenizer_seconds": round(tokenizer_seconds, 3),
            "model_seconds": round(model_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }

        print(f"Tokenizer loaded in {tokenizer_seconds:.2f} seconds")
        print(f"Model loaded in {model_seconds:.2f} seconds")
        print(f"Total load time: {total_seconds:.2f} seconds")
        print(f"Device: {next(self.model.parameters()).device}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory reserved:  {reserved:.2f} GB")

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
    ) -> dict[str, Any]:
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
        ).to(self.model.device)

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

        streamer = TextIteratorStreamer(
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

        gpu_after = get_gpu_status()
        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = bytes_to_gb(torch.cuda.max_memory_allocated())

        return {
            "model": self.model_name,
            "prompt": prompt,
            "settings": asdict(self.settings),
            "finish_reason": finish_reason,
            "generation_truncated": generation_truncated,
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
