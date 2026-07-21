from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationSettings:
    enable_thinking: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    do_sample: bool = True
    repetition_penalty: float = 1.0


class LocalLLM:
    def __init__(self, model_name: str, settings: GenerationSettings) -> None:
        self.model_name = model_name
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
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

        print(f"\nLoading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}")
        started = time.perf_counter()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

        self.model_name = model_name
        elapsed = time.perf_counter() - started

        print(f"Model loaded in {elapsed:.2f} seconds")
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

    def generate(self, prompt: str) -> dict[str, Any]:
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

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        started = time.perf_counter()

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

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - started

        generated_ids = outputs[0, prompt_tokens:]
        generated_tokens = generated_ids.shape[0]

        raw_output = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        thinking, answer = self.split_thinking_and_answer(raw_output)

        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        return {
            "model": self.model_name,
            "prompt": prompt,
            "settings": asdict(self.settings),
            "result": {
                "raw_output": raw_output,
                "thinking": thinking,
                "answer": answer,
            },
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generation_seconds": round(elapsed, 3),
                "tokens_per_second": (
                    round(generated_tokens / elapsed, 2)
                    if elapsed > 0
                    else None
                ),
                "peak_vram_gb": (
                    round(peak_vram_gb, 3)
                    if peak_vram_gb is not None
                    else None
                ),
            },
        }
