from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch

from llm_benchmark.llm import GenerationSettings, LocalLLM
from llm_benchmark.reporting import (
    get_gpu_status,
    get_system_metadata,
    save_benchmark_result,
)


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

 /benchmark or /status
    Show model, settings, GPU status, and the most recent query metrics.

 /quit
    Exit and release GPU memory.
"""
    )


def print_metrics(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    time_to_first_token = metrics["time_to_first_token_seconds"]
    peak_vram = metrics["peak_vram_gb"]

    def format_gb(value: float | None) -> str:
        return f"{value:.3f} GB" if value is not None else "unavailable"

    print("\nMetrics")
    print("-------")
    print(
        "Time to first streamed output: "
        f"{time_to_first_token:.3f} seconds"
        if time_to_first_token is not None
        else "Time to first streamed output: unavailable"
    )
    print(f"Total generation time: {metrics['generation_seconds']:.3f} seconds")
    print(f"Prompt tokens: {metrics['prompt_tokens']}")
    print(f"Output tokens: {metrics['generated_tokens']}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Output tokens per second: {metrics['tokens_per_second']}")
    print(f"Finish reason: {result['finish_reason']}")
    print(f"Generation truncated: {result['generation_truncated']}")
    print(f"Peak VRAM: {format_gb(peak_vram)}")
    print(
        "GPU allocated before: "
        f"{format_gb(metrics['gpu_allocated_before_gb'])}"
    )
    print(
        "GPU reserved before: "
        f"{format_gb(metrics['gpu_reserved_before_gb'])}"
    )
    print(
        "GPU allocated after: "
        f"{format_gb(metrics['gpu_allocated_after_gb'])}"
    )
    print(
        "GPU reserved after: "
        f"{format_gb(metrics['gpu_reserved_after_gb'])}"
    )


def build_status(llm: LocalLLM, last_result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model": llm.model_name,
        "model_load_metrics": llm.model_load_metrics,
        "settings": asdict(llm.settings),
        "gpu": get_gpu_status(),
        "last_query_metrics": (
            last_result["metrics"] if last_result is not None else None
        ),
    }


def run() -> None:
    settings = GenerationSettings()
    llm = LocalLLM("Qwen/Qwen3-8B", settings)
    last_result: dict[str, Any] | None = None

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

            if user_input in {"/benchmark", "/status"}:
                print(json.dumps(build_status(llm, last_result), indent=2))
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
                print("\nResponse")
                print("--------")
                result = llm.generate(
                    user_input,
                    on_text=lambda text: print(text, end="", flush=True),
                )
                print()

                result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                result["metadata"] = get_system_metadata()
                last_result = result

                print_metrics(result)

                print("\nJSON")
                print(json.dumps(result, indent=2, ensure_ascii=False))

                print("\nThinking")
                print("--------")
                print(result["result"]["thinking"] or "[No thinking block returned]")

                print("\nAnswer")
                print("------")
                print(result["result"]["answer"] or "[No final answer returned]")

                saved_path = save_benchmark_result(result)
                print(f"\nSaved benchmark: {saved_path}")

            except torch.cuda.OutOfMemoryError:
                print("\nCUDA out of memory.")
                print("Try lowering max_new_tokens or loading a smaller model.")
                torch.cuda.empty_cache()

            except Exception as exc:
                print(f"\nGeneration failed: {type(exc).__name__}: {exc}")

    finally:
        print("\nUnloading model...")
        llm.unload_model()
