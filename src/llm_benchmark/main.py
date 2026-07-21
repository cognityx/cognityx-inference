from __future__ import annotations

import gc
import json
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


def parse_value(current_value: Any, new_value: str) -> Any:
    if isinstance(current_value, bool):
        normalized = new_value.lower()

        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False

        raise ValueError("Use true or false.")

    if isinstance(current_value, int):
        return int(new_value)

    if isinstance(current_value, float):
        return float(new_value)

    return new_value


def print_help() -> None:
    print(
        """
Commands
--------
/help
    Show this help.

 /settings
    Show the active model and generation parameters.

 /set PARAMETER VALUE
    Change a generation parameter without restarting.

    Examples:
        /set enable_thinking false
        /set max_new_tokens 500
        /set temperature 0.7
        /set top_p 0.8
        /set top_k 20
        /set do_sample true

 /model REPOSITORY_NAME
    Unload the current model and load another model.

    Example:
        /model Qwen/Qwen3-4B

 /quit
    Exit and release GPU memory.
"""
    )


def main() -> None:
    settings = GenerationSettings()
    llm = LocalLLM("Qwen/Qwen3-8B", settings)

    print_help()

    try:
        while True:
            user_input = input("\nYou> ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                break

            if user_input == "/help":
                print_help()
                continue

            if user_input == "/settings":
                print(
                    json.dumps(
                        {
                            "model": llm.model_name,
                            "settings": asdict(llm.settings),
                        },
                        indent=2,
                    )
                )
                continue

            if user_input.startswith("/set "):
                parts = user_input.split(maxsplit=2)

                if len(parts) != 3:
                    print("Usage: /set PARAMETER VALUE")
                    continue

                parameter, supplied_value = parts[1], parts[2]

                if not hasattr(llm.settings, parameter):
                    print(f"Unknown parameter: {parameter}")
                    print(f"Available: {', '.join(asdict(llm.settings))}")
                    continue

                try:
                    current_value = getattr(llm.settings, parameter)
                    parsed_value = parse_value(current_value, supplied_value)
                    setattr(llm.settings, parameter, parsed_value)
                    print(f"{parameter} = {parsed_value}")
                except ValueError as exc:
                    print(f"Invalid value: {exc}")

                continue

            if user_input.startswith("/model "):
                model_name = user_input.removeprefix("/model ").strip()

                if not model_name:
                    print("Usage: /model REPOSITORY_NAME")
                    continue

                try:
                    llm.load_model(model_name)
                except Exception as exc:
                    print(f"Could not load model: {exc}")
                    print("Attempting to restore Qwen/Qwen3-8B...")
                    llm.load_model("Qwen/Qwen3-8B")

                continue

            try:
                result = llm.generate(user_input)

                print("\nJSON")
                print(json.dumps(result, indent=2, ensure_ascii=False))

                print("\nThinking")
                print("--------")
                print(result["result"]["thinking"] or "[No thinking block returned]")

                print("\nAnswer")
                print("------")
                print(result["result"]["answer"] or "[No final answer returned]")

            except torch.cuda.OutOfMemoryError:
                print("\nCUDA out of memory.")
                print("Try lowering max_new_tokens or loading a smaller model.")
                torch.cuda.empty_cache()

            except Exception as exc:
                print(f"\nGeneration failed: {type(exc).__name__}: {exc}")

    finally:
        print("\nUnloading model...")
        llm.unload_model()


if __name__ == "__main__":
    main()
