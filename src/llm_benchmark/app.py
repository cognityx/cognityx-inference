from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch

from llm_benchmark.comparison import (
    load_comparison_rows,
    parse_compare_command,
    render_table,
    save_comparison,
)
from llm_benchmark.cost_estimation import (
    CostConfig,
    configure_cost_model,
    estimate_api_equivalent_cost,
)
from llm_benchmark.llm import (
    LOAD_PROFILES,
    GenerationSettings,
    LocalLLM,
    validate_load_profile,
)
from llm_benchmark.reporting import (
    get_gpu_status,
    get_system_metadata,
    save_benchmark_result,
)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LOAD_PROFILE = "bf16"
BENCHMARK_PROMPT = "Can AI be applied to the travelling salesman problem?"
BENCHMARK_NAME = "travelling_salesman_problem"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a persistent local LLM benchmark chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-8B --profile bf16
  uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-14B --profile int4
""",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name")
    parser.add_argument(
        "--profile",
        choices=LOAD_PROFILES,
        default=DEFAULT_LOAD_PROFILE,
        help="Model-loading profile (default: bf16)",
    )
    return parser.parse_args(arguments)


def parse_model_command(user_input: str) -> tuple[str, str]:
    parts = user_input.split()
    if len(parts) not in {2, 3}:
        raise ValueError("Usage: /model REPOSITORY_NAME [PROFILE]")
    profile = parts[2] if len(parts) == 3 else DEFAULT_LOAD_PROFILE
    return parts[1], validate_load_profile(profile)


def benchmark_request() -> tuple[str, str, str]:
    return BENCHMARK_PROMPT, "benchmark", BENCHMARK_NAME


def parse_cost_model_command(user_input: str) -> CostConfig:
    parts = shlex.split(user_input)
    if len(parts) != 4:
        raise ValueError("Usage: /cost-model LABEL INPUT_PRICE_PER_MILLION OUTPUT_PRICE_PER_MILLION")
    try:
        input_price = float(parts[2])
        output_price = float(parts[3])
    except ValueError as exc:
        raise ValueError("Input and output prices must be numbers") from exc
    return configure_cost_model(parts[1], input_price, output_price)


def parse_cost_command(user_input: str) -> tuple[str, CostConfig | None]:
    if user_input == "/cost-off":
        return "off", None
    if user_input == "/cost-status":
        return "status", None
    if user_input.startswith("/cost-model"):
        return "model", parse_cost_model_command(user_input)
    raise ValueError("Unknown cost command")


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

 /model REPOSITORY_NAME [PROFILE]
    Unload the current model and load another model.

    Examples:
        /model Qwen/Qwen3-8B bf16
        /model Qwen/Qwen3-14B int8
        /model Qwen/Qwen3-14B int4

 /benchmark
    Run the travelling-salesman benchmark prompt through normal generation.

 /status
    Show model, settings, GPU status, and the most recent query metrics.

 /status detailed
    Include the complete model device map.

 /cost-model LABEL INPUT_PRICE OUTPUT_PRICE
    Enable a user-supplied API-equivalent token-cost estimate.

 /cost-off
    Disable API-equivalent cost estimation.

 /cost-status
    Show current cost-estimation assumptions.

 /compare [--all] [--model NAME] [--benchmark NAME] [--latest N]
          [--architecture] [--format table|markdown|csv]
    Compare saved benchmark runs; defaults to benchmark runs only.

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
    performance = result.get("performance", {})
    if performance:
        print("\nPerformance")
        print("-----------")
        print(f"Prompt tokenization: {performance['prompt_tokenization_seconds']:.3f} s")
        print(f"Prefill / first-token latency: {performance['prefill_seconds']} s")
        print(f"Prompt throughput: {performance['prompt_tokens_per_second']} tok/s")
        print(f"Decode time: {performance['decode_seconds']} s")
        print(f"Decode throughput: {performance['decode_tokens_per_second']} tok/s")
        print(f"End-to-end time: {performance['end_to_end_seconds']:.3f} s")


def build_status(
    llm: LocalLLM,
    last_result: dict[str, Any] | None,
    cost_config: CostConfig | None = None,
    detailed: bool = False,
) -> dict[str, Any]:
    runtime = llm.model_runtime or {}
    cache = runtime.get("kv_cache", {})
    model_size = runtime.get("model_size", {})
    model_family = runtime.get("model_family", {})
    runtime_status = {
        "model_dtype": runtime.get("model_dtype", "unknown"),
        "devices_used": runtime.get("devices_used", []),
        "uses_cpu_offload": runtime.get("uses_cpu_offload", False),
        "uses_disk_offload": runtime.get("uses_disk_offload", False),
        "attention_implementation": runtime.get(
            "attention_implementation", "unknown"
        ),
        "cache_enabled": cache.get("use_cache"),
        "cache_implementation": cache.get("cache_implementation", "unknown"),
        "logical_parameters": model_size.get("logical_parameters"),
        "logical_parameters_billions": model_size.get("logical_parameters_billions"),
        "estimated_raw_weight_storage_gib": model_size.get(
            "estimated_raw_weight_storage_gib"
        ),
        "parameter_count_source": model_size.get("parameter_count_source"),
        "architecture_type": model_family.get("architecture_type", "unknown"),
        "active_parameters_per_token": model_family.get(
            "active_parameters_per_token"
        ),
        "active_parameters_billions": model_family.get(
            "active_parameters_billions"
        ),
        "active_parameter_fraction": model_family.get("active_parameter_fraction"),
        "experts": (
            {
                "total": model_family.get("num_experts"),
                "selected_per_token": model_family.get("num_experts_per_token"),
                "shared": model_family.get("shared_experts"),
            }
            if model_family.get("architecture_type") == "moe"
            else None
        ),
    }
    if detailed:
        runtime_status["device_map"] = runtime.get("device_map")
        runtime_status["model_family"] = model_family

    return {
        "model": llm.model_name,
        "requested_load_profile": llm.requested_load_profile,
        "effective_load_profile": llm.effective_load_profile,
        "load_profile": llm.load_profile_metadata,
        "model_load_metrics": llm.model_load_metrics,
        "model_runtime": runtime_status,
        "settings": asdict(llm.settings),
        "gpu": get_gpu_status(),
        "last_query_metrics": (
            last_result["metrics"] if last_result is not None else None
        ),
        "cost_estimation": (cost_config or CostConfig()).to_dict(),
    }


def process_prompt(
    llm: LocalLLM,
    prompt: str,
    run_type: str = "interactive",
    benchmark_name: str | None = None,
    cost_config: CostConfig | None = None,
) -> dict[str, Any]:
    print("\nResponse")
    print("--------")
    result = llm.generate(
        prompt,
        on_text=lambda text: print(text, end="", flush=True),
        run_type=run_type,
        benchmark_name=benchmark_name,
    )
    print()

    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["metadata"] = get_system_metadata()
    result["schema_version"] = 2
    result["api_equivalent_cost_estimate"] = estimate_api_equivalent_cost(
        cost_config or CostConfig(),
        result["metrics"]["prompt_tokens"],
        result["metrics"]["generated_tokens"],
    )

    print_metrics(result)
    cost_estimate = result["api_equivalent_cost_estimate"]
    if cost_estimate["enabled"]:
        print(
            "API-equivalent estimate: "
            f"{cost_estimate['currency']} {cost_estimate['estimated_total_cost']:.12f} "
            f"({cost_estimate['pricing_label']}; hypothetical, not local cost)"
        )

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
    return result


def run(
    model_name: str = DEFAULT_MODEL,
    load_profile: str = DEFAULT_LOAD_PROFILE,
) -> None:
    settings = GenerationSettings()
    llm = LocalLLM(model_name, settings, load_profile)
    last_result: dict[str, Any] | None = None
    cost_config = CostConfig()

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
                            "requested_load_profile": llm.requested_load_profile,
                            "effective_load_profile": llm.effective_load_profile,
                            "load_profile": llm.load_profile_metadata,
                            "settings": asdict(llm.settings),
                        },
                        indent=2,
                    )
                )
                continue

            if user_input in {"/status", "/status detailed"}:
                print(
                    json.dumps(
                        build_status(
                            llm,
                            last_result,
                            cost_config,
                            detailed=user_input.endswith(" detailed"),
                        ),
                        indent=2,
                    )
                )
                continue

            if user_input.startswith("/cost-"):
                try:
                    cost_action, configured_cost = parse_cost_command(user_input)
                    if cost_action == "model":
                        cost_config = configured_cost or CostConfig()
                        print("API-equivalent cost estimation enabled.")
                        print(json.dumps(cost_config.to_dict(), indent=2))
                    elif cost_action == "off":
                        cost_config = CostConfig()
                        print("API-equivalent cost estimation disabled.")
                    else:
                        print(json.dumps(cost_config.to_dict(), indent=2))
                except ValueError as exc:
                    print(f"Invalid cost model: {exc}")
                continue

            if user_input.startswith("/compare"):
                try:
                    options = parse_compare_command(user_input)
                    rows, warnings = load_comparison_rows(options=options)
                    for warning in warnings:
                        print(f"Warning: {warning}")
                    if options.output_format == "table":
                        print(render_table(rows, options.include_architecture))
                    else:
                        path = save_comparison(
                            rows,
                            options.output_format,
                            include_architecture=options.include_architecture,
                        )
                        print(f"Saved comparison: {path}")
                except ValueError as exc:
                    print(f"Invalid compare command: {exc}")
                continue

            if user_input == "/benchmark":
                prompt, run_type, benchmark_name = benchmark_request()
                try:
                    last_result = process_prompt(
                        llm,
                        prompt,
                        run_type,
                        benchmark_name,
                        cost_config,
                    )
                except torch.cuda.OutOfMemoryError:
                    print("\nCUDA out of memory.")
                    print("Try lowering max_new_tokens or loading a smaller model.")
                    torch.cuda.empty_cache()
                except Exception as exc:
                    print(f"\nGeneration failed: {type(exc).__name__}: {exc}")
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
                try:
                    requested_model, requested_profile = parse_model_command(user_input)
                except ValueError as exc:
                    print(exc)
                    continue

                try:
                    llm.switch_model(requested_model, requested_profile)
                except Exception as exc:
                    print(f"Could not load model: {exc}")

                continue

            try:
                last_result = process_prompt(
                    llm,
                    user_input,
                    cost_config=cost_config,
                )

            except torch.cuda.OutOfMemoryError:
                print("\nCUDA out of memory.")
                print("Try lowering max_new_tokens or loading a smaller model.")
                torch.cuda.empty_cache()

            except Exception as exc:
                print(f"\nGeneration failed: {type(exc).__name__}: {exc}")

    finally:
        print("\nUnloading model...")
        llm.unload_model()


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    run(args.model, args.profile)
