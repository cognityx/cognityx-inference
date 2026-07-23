from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

from llm_benchmark.comparison import (
    load_comparison_rows,
    parse_compare_command,
    render_table,
    save_comparison,
)
from llm_benchmark.configuration import ConfigurationError, load_generation_settings
from llm_benchmark.cost_estimation import (
    CostConfig,
    configure_cost_model,
    estimate_api_equivalent_cost,
)
from llm_benchmark.llm import (
    GenerationCancelled,
    LOAD_PROFILES,
    LocalLLM,
    validate_load_profile,
)
from llm_benchmark.model_loading import ModelPlacementError, report_model_load_failure
from llm_benchmark.reporting import (
    get_gpu_status,
    get_system_metadata,
    save_benchmark_result,
)
from llm_benchmark.download_preflight import format_bytes, inspect_download
from llm_benchmark.contexts import (
    ContextError,
    build_context_prompt,
    prepare_folder_context,
)
from llm_benchmark.loading_decision import ModelLoadOptions, controlled_max_memory, estimate_capacity
from llm_benchmark.model_size import estimate_parameter_count_from_config
from llm_benchmark.vllm_engine import VLLMEngineError, VLLMLLM
from transformers import AutoConfig

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LOAD_PROFILE = "bf16"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"
BENCHMARK_PROMPT = "Can AI be applied to the travelling salesman problem?"
BENCHMARK_NAME = "travelling_salesman_problem"
CONTEXT_PROMPT_PREVIEW_WORDS = 24


def prompt_preview(prompt: str, word_limit: int = CONTEXT_PROMPT_PREVIEW_WORDS) -> str:
    words = prompt.split()
    if len(words) <= word_limit:
        return prompt
    return " ".join(words[:word_limit]) + " … [truncated]"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a persistent local LLM benchmark chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-8B --profile bf16
  uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-14B --profile int4
  uv run python src/llm_benchmark/main.py --config config.toml
  uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-32B --prepare-context wikitext --corpus wikitext-103-raw
  uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-32B --profile int4 --context wikitext --context-tokens 32000
  uv run python src/llm_benchmark/main.py --engine vllm --model Qwen/Qwen3-32B --profile int4 --context wikitext --context-tokens 10000 --vllm-max-model-len 29616
""",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name")
    parser.add_argument(
        "--engine", choices=("transformers", "vllm"), default="transformers",
        help="Inference engine (default: transformers)",
    )
    parser.add_argument(
        "--kv-cache-dtype", choices=("auto", "fp8"), default="auto",
        help="vLLM KV-cache dtype; auto uses the model dtype",
    )
    parser.add_argument(
        "--vllm-max-model-len", type=int,
        help="vLLM engine context capacity; must cover prompt plus max_new_tokens",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory vLLM may reserve (default: 0.90)",
    )
    parser.add_argument(
        "--vllm-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM automatic prefix caching (default: enabled)",
    )
    parser.add_argument(
        "--profile",
        choices=LOAD_PROFILES,
        default=DEFAULT_LOAD_PROFILE,
        help=(
            "bf16/fp16: explicit dtype; int4/int8: bitsandbytes for unquantized "
            "checkpoints; native: embedded checkpoint quantization; auto: native "
            "when present, otherwise bf16"
        ),
    )
    download = parser.add_mutually_exclusive_group()
    download.add_argument("--allow-download", action="store_true", help="Approve missing checkpoint downloads without prompting")
    download.add_argument("--no-download", action="store_true", help="Use cached files only; never download")
    parser.add_argument("--non-interactive", action="store_true", help="Disable all prompts; download permission must be explicit")
    parser.add_argument("--allow-cpu-offload", action="store_true", help="Allow controlled GPU-first CPU overflow placement")
    parser.add_argument("--context", help="Prepared context name under data/contexts")
    parser.add_argument("--context-tokens", type=int, help="Requested corpus-token budget")
    parser.add_argument("--prepare-context", metavar="NAME", help="Prepare a named folder context and exit")
    parser.add_argument("--corpus", help="Corpus path or wikitext-103-raw")
    parser.add_argument("--chunk-tokens", type=int, default=1024, help="Tokens per prepared chunk (default: 1024)")
    parser.add_argument("--chunk-overlap", type=int, default=128, help="Overlapping tokens between chunks (default: 128)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML configuration file containing generation defaults",
    )
    args = parser.parse_args(arguments)
    if bool(args.context) != bool(args.context_tokens):
        parser.error("--context and --context-tokens must be supplied together")
    if args.context_tokens is not None and args.context_tokens <= 0:
        parser.error("--context-tokens must be greater than zero")
    if args.prepare_context and not args.corpus:
        parser.error("--prepare-context requires --corpus")
    if args.vllm_max_model_len is not None and args.vllm_max_model_len <= 0:
        parser.error("--vllm-max-model-len must be greater than zero")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        parser.error("--vllm-gpu-memory-utilization must be greater than 0 and at most 1")
    return args


def ensure_engine_runtime(engine: str) -> None:
    """Re-execute vLLM runs in the isolated environment without touching .venv."""
    if engine != "vllm" or importlib.util.find_spec("vllm") is not None:
        return
    project_root = Path(__file__).resolve().parents[2]
    python = project_root / ".venv-vllm" / "bin" / "python"
    if not python.is_file():
        raise VLLMEngineError(
            "Missing isolated .venv-vllm. Create it with `uv venv .venv-vllm "
            "--python 3.12 --seed`, then install with `uv pip install --python "
            ".venv-vllm/bin/python vllm bitsandbytes --torch-backend=auto`."
        )
    environment = os.environ.copy()
    environment.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    cuda_home = next(
        (project_root / ".venv-vllm" / "lib").glob(
            "python*/site-packages/nvidia/cu13"
        ),
        None,
    )
    if cuda_home is not None:
        environment.setdefault("CUDA_HOME", str(cuda_home))
        cuda_lib = cuda_home / "lib"
        cuda_lib64 = cuda_home / "lib64"
        if cuda_lib.is_dir() and not cuda_lib64.exists():
            try:
                cuda_lib64.symlink_to(cuda_lib.name, target_is_directory=True)
            except OSError as exc:
                raise VLLMEngineError(
                    f"Could not create the CUDA lib64 compatibility link: {exc}"
                ) from exc
        cudart_versioned = cuda_lib / "libcudart.so.13"
        cudart_link = cuda_lib / "libcudart.so"
        if cudart_versioned.is_file() and not cudart_link.exists():
            try:
                cudart_link.symlink_to(cudart_versioned.name)
            except OSError as exc:
                raise VLLMEngineError(
                    f"Could not create the CUDA runtime linker name: {exc}"
                ) from exc
        cuda_library_paths = [str(cuda_home / "lib")]
        wsl_driver_library = Path("/usr/lib/wsl/lib")
        if wsl_driver_library.is_dir():
            cuda_library_paths.append(str(wsl_driver_library))
        existing_library_path = environment.get("LIBRARY_PATH")
        if existing_library_path:
            cuda_library_paths.append(existing_library_path)
        environment["LIBRARY_PATH"] = ":".join(cuda_library_paths)
        existing_runtime_path = environment.get("LD_LIBRARY_PATH")
        if existing_runtime_path:
            cuda_library_paths.append(existing_runtime_path)
        environment["LD_LIBRARY_PATH"] = ":".join(cuda_library_paths)
    path_entries = [str(python.parent)]
    if cuda_home is not None:
        path_entries.append(str(cuda_home / "bin"))
    environment["PATH"] = ":".join(
        [*path_entries, environment.get("PATH", "")]
    )
    os.execve(
        str(python),
        [str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        environment,
    )


def parse_model_command(user_input: str) -> tuple[str, str]:
    parts = user_input.split()
    if len(parts) not in {2, 3}:
        raise ValueError("Usage: /model REPOSITORY_NAME [PROFILE]")
    profile = parts[2] if len(parts) == 3 else DEFAULT_LOAD_PROFILE
    return parts[1], validate_load_profile(profile)


def benchmark_request() -> tuple[str, str, str]:
    return BENCHMARK_PROMPT, "benchmark", BENCHMARK_NAME


def approve_download(preflight: Any, *, allow: bool, deny: bool, non_interactive: bool, input_fn: Any = input) -> tuple[bool, dict[str, Any]]:
    print()
    if preflight.cached_checkpoint:
        print("Model checkpoint already cached")
        print(f"Cache path: {preflight.cache_location}")
        print(f"Approximate cached size: {format_bytes(preflight.cached_bytes)}")
        return False, {"download_approved": False, "checkpoint_already_cached": True}
    print("The selected model requires an additional download.\n")
    print(f"Model: {preflight.model}")
    print(f"Already cached: {format_bytes(preflight.cached_bytes)}")
    print(f"Additional download: {format_bytes(preflight.missing_download_bytes)}")
    print(f"Available disk space: {format_bytes(preflight.available_disk_bytes)}")
    print(f"Cache location: {preflight.cache_location}")
    if deny:
        return False, {"download_approved": False}
    if allow:
        return True, {"download_approved": True, "source": "--allow-download"}
    if non_interactive:
        raise RuntimeError("Download required. Non-interactive mode requires --allow-download (or pre-cache the checkpoint).")
    approved = input_fn("Proceed with download? [y/N] ").strip().lower() in {"y", "yes"}
    return approved, {"download_approved": approved, "source": "interactive_prompt"}


def print_capacity(estimate: dict[str, Any]) -> None:
    print("\nPre-load memory estimate")
    print("------------------------")
    print(f"Logical parameters: {estimate['logical_parameters'] or 'unknown'}")
    print(f"Requested profile: {estimate['requested_profile']}")
    print(f"Nominal bits/weight: {estimate['nominal_bits_per_weight'] or 'unknown'}")
    raw = estimate['theoretical_raw_weight_bytes']
    print(f"Theoretical raw weight storage: {format_bytes(raw) if raw else 'unknown'}")
    print(f"GPU total/free: {estimate['gpu_total_gib']} / {estimate['gpu_free_gib']} GiB")
    print(f"GPU-only placement: {estimate['gpu_only_estimate']}")
    print(f"Overhead warning: {estimate['overhead_warning']}")


def choose_loading_mode(profile: str, *, cpu_flag: bool, non_interactive: bool, input_fn: Any = input) -> tuple[str, str, str | None]:
    if non_interactive:
        return ("gpu_cpu_offload" if cpu_flag else "gpu_only"), profile, None
    print("\nSelect loading mode:\n")
    print("1. GPU only")
    if profile in {"int4", "int4-double"}:
        print("2. GPU only with INT4 nested/double quantization")
    print("3. GPU + CPU offload")
    print("4. Select a separate pre-quantized 3-bit or 2-bit checkpoint")
    print("5. Cancel")
    choice = input_fn("\nChoice: ").strip()
    if choice == "1": return "gpu_only", profile, None
    if choice == "2" and profile in {"int4", "int4-double"}: return "gpu_only", "int4-double", None
    if choice == "3": return "gpu_cpu_offload", profile, None
    if choice == "4":
        bits = input_fn("GPTQ bit width [3/2]: ").strip()
        if bits not in {"2", "3"}: raise ValueError("GPTQ bit width must be 2 or 3")
        checkpoint = input_fn("Enter the compatible pre-quantized model ID or local path: ").strip()
        if not checkpoint: raise ValueError("A pre-quantized checkpoint is required")
        return "gpu_only_low_bit", f"gptq{bits}", checkpoint
    raise RuntimeError("Model loading cancelled.")


def prepare_load(model: str, profile: str, *, allow_download: bool, no_download: bool, non_interactive: bool, allow_cpu_offload: bool, input_fn: Any = input) -> tuple[str, str, ModelLoadOptions]:
    original = model
    confirmations: dict[str, Any] = {}
    while True:
        preflight = inspect_download(model)
        can_download, confirmation = approve_download(preflight, allow=allow_download, deny=no_download, non_interactive=non_interactive, input_fn=input_fn)
        confirmations.update(confirmation)
        if not preflight.cached_checkpoint and not can_download:
            raise RuntimeError("Model download was not approved; loading stopped safely.")
        config = AutoConfig.from_pretrained(model, local_files_only=not can_download)
        estimate = estimate_capacity(estimate_parameter_count_from_config(config), profile)
        print_capacity(estimate)
        choice, selected_profile, alternate = choose_loading_mode(profile, cpu_flag=allow_cpu_offload, non_interactive=non_interactive, input_fn=input_fn)
        if alternate:
            model, profile = alternate, selected_profile
            continue
        options = ModelLoadOptions(
            loading_choice=choice,
            allow_download=can_download,
            cpu_offload_allowed=choice == "gpu_cpu_offload",
            strict_gpu_only=choice != "gpu_cpu_offload",
            max_memory=controlled_max_memory() if choice == "gpu_cpu_offload" else None,
            download_preflight=preflight.to_dict(),
            original_model_id=original,
            quantized_checkpoint_model_id=model if model != original else None,
            user_confirmations=confirmations,
        )
        return model, selected_profile, options


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


def parse_context_command(user_input: str) -> tuple[str | None, int | None]:
    parts = user_input.split()
    if parts == ["/context"]:
        raise ValueError("status")
    if parts == ["/context", "off"]:
        return None, None
    if len(parts) != 3:
        raise ValueError("Usage: /context CONTEXT_NAME TOKEN_COUNT, or /context off")
    try:
        token_count = int(parts[2])
    except ValueError as exc:
        raise ValueError("Context token count must be an integer") from exc
    if token_count <= 0:
        raise ValueError("Context token count must be greater than zero")
    return parts[1], token_count


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
        /model openai/gpt-oss-20b native

 /benchmark
    Run the travelling-salesman benchmark prompt through normal generation.

 /context CONTEXT_NAME TOKEN_COUNT
    Change the prepared context and token budget without reloading the model.

    Example:
        /context wikitext 32000

 /context
    Show the active context.

 /context off
    Disable context for subsequent benchmark runs.

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
    context_length = metrics.get("total_context_length_tokens")
    print(
        f"Total context length supported: {context_length} tokens"
        if context_length is not None
        else "Total context length supported: unknown"
    )
    print(f"Output tokens per second: {metrics['tokens_per_second']}")
    if "vllm:prefix_cache_queries" in metrics:
        print(
            "vLLM prefix-cache queried tokens (engine cumulative): "
            f"{metrics['vllm:prefix_cache_queries']}"
        )
        print(
            "vLLM prefix-cache hit tokens (engine cumulative): "
            f"{metrics['vllm:prefix_cache_hits']}"
        )
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
    breakdown = metrics.get("gpu_memory_breakdown")
    if breakdown:
        print("\nGPU memory after — measured/estimated breakdown")
        print("------------------------------------------------")
        print(
            "CUDA model parameter/buffer tensors (measured lower bound): "
            f"{format_gb(breakdown['cuda_model_parameter_and_buffer_tensors_gb'])}"
        )
        print(
            "Encoded prompt tensors: "
            f"{format_gb(breakdown['encoded_prompt_tensors_gb'])}"
        )
        print(
            "Estimated KV cache at final sequence length: "
            f"{format_gb(breakdown['estimated_kv_cache_at_final_sequence_gb'])}"
        )
        print(
            "Other allocated after generation: "
            f"{format_gb(breakdown['other_allocated_after_gb'])}"
        )
        print(
            "Unused reserved allocator memory: "
            f"{format_gb(breakdown['unused_reserved_allocator_gb'])}"
        )
        print(
            "Estimated peak transient runtime (activations/workspaces): "
            f"{format_gb(breakdown['estimated_peak_transient_runtime_gb'])}"
        )
        print("Prefill cache: not separate; see gpu_memory_breakdown.prefill_note in JSON")
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
    active_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = llm.model_runtime or {}
    cache = runtime.get("kv_cache", {})
    model_size = runtime.get("model_size", {})
    model_family = runtime.get("model_family", {})
    runtime_status = {
        "engine": runtime.get("engine", "transformers"),
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
        "quantization_resolution": getattr(llm, "quantization_resolution", None),
        "model_load_metrics": llm.model_load_metrics,
        "model_runtime": runtime_status,
        "loading": getattr(llm, "loading_diagnostics", None),
        "settings": asdict(llm.settings),
        "gpu": get_gpu_status(),
        "last_query_metrics": (
            last_result["metrics"] if last_result is not None else None
        ),
        "cost_estimation": (cost_config or CostConfig()).to_dict(),
        "context": active_context,
    }


def process_prompt(
    llm: LocalLLM,
    prompt: str,
    run_type: str = "interactive",
    benchmark_name: str | None = None,
    cost_config: CostConfig | None = None,
    context_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print("\nResponse")
    print("--------")
    print("Press Ctrl+C to stop this generation and keep the model loaded.\n")
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
    if context_metadata is not None:
        measured_prompt_tokens = result["metrics"]["prompt_tokens"]
        if measured_prompt_tokens != context_metadata["final_prompt_tokens"]:
            raise RuntimeError(
                "Context prompt token count changed between budgeting and generation; "
                "the run was not saved."
            )
        context_metadata = {
            **context_metadata,
            "peak_vram_gb": result["metrics"]["peak_vram_gb"],
            "uses_cpu_offload": (getattr(llm, "loading_diagnostics", None) or {}).get(
                "uses_cpu_offload", False
            ),
            "uses_disk_offload": (getattr(llm, "loading_diagnostics", None) or {}).get(
                "disk_offload_used", False
            ),
            "full_prompt_sha256": hashlib.sha256(
                result["prompt"].encode("utf-8")
            ).hexdigest(),
            "full_prompt_characters": len(result["prompt"]),
        }
        original_prompt = result["prompt"]
        result["prompt"] = prompt_preview(original_prompt)
        result["prompt_truncated_for_reporting"] = result["prompt"] != original_prompt
        result["prompt_preview_words"] = CONTEXT_PROMPT_PREVIEW_WORDS
    else:
        result["prompt_truncated_for_reporting"] = False
    result["context"] = context_metadata
    result["api_equivalent_cost_estimate"] = estimate_api_equivalent_cost(
        cost_config or CostConfig(),
        result["metrics"]["prompt_tokens"],
        result["metrics"]["generated_tokens"],
    )

    print_metrics(result)
    if context_metadata is not None:
        print(
            "Context: "
            f"{context_metadata['context_name']} ({context_metadata['context_provider']}), "
            f"{context_metadata['actual_corpus_tokens']} corpus tokens, "
            f"{context_metadata['final_prompt_tokens']} final prompt tokens, "
            f"{context_metadata['chunks_used_count']} chunks, "
            f"input {context_metadata['input_truncation_status']}"
        )
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
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    allow_download: bool = False,
    no_download: bool = False,
    non_interactive: bool = False,
    allow_cpu_offload: bool = False,
    context_name: str | None = None,
    context_tokens: int | None = None,
    engine: str = "transformers",
    kv_cache_dtype: str = "auto",
    vllm_max_model_len: int | None = None,
    vllm_gpu_memory_utilization: float = 0.90,
    vllm_prefix_caching: bool = True,
) -> None:
    settings = load_generation_settings(config_path)
    print(f"Generation settings loaded from: {config_path}")
    original_model = model_name
    while True:
        model_name, load_profile, load_options = prepare_load(model_name, load_profile, allow_download=allow_download, no_download=no_download, non_interactive=non_interactive, allow_cpu_offload=allow_cpu_offload)
        try:
            if engine == "vllm":
                configured_model_len = vllm_max_model_len or (
                    (context_tokens or 1024) + settings.max_new_tokens + 1024
                )
                llm = VLLMLLM(
                    model_name, settings, load_profile, load_options,
                    kv_cache_dtype=kv_cache_dtype,
                    max_model_len=configured_model_len,
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                    enable_prefix_caching=vllm_prefix_caching,
                )
            else:
                llm = LocalLLM(model_name, settings, load_profile, load_options)
            placement = llm.loading_diagnostics
            if placement["capacity_classification"] == "gpu_cpu_offloaded_inference":
                print("\nGPU + CPU offloaded inference success")
                print("CPU-resident modules:")
                for module in placement["cpu_resident_modules"]:
                    print(f"  - {module or '<root>'}")
            elif placement["capacity_classification"] == "gpu_only_low_bit_inference":
                print("\nGPU-only low-bit inference success")
                print("Note: aggressive quantization may affect response quality; retain the answer for human inspection.")
            else:
                print("\nGPU-only inference success")
            break
        except ModelPlacementError as exc:
            report_model_load_failure(exc)
            if non_interactive:
                raise
            print("\nThe model cannot fit entirely on the GPU with the current profile.")
            print("Choose another loading mode from the menu; the cached checkpoint will be reused.")
            model_name = original_model
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
                            "engine": engine,
                            "requested_load_profile": llm.requested_load_profile,
                            "effective_load_profile": llm.effective_load_profile,
                            "load_profile": llm.load_profile_metadata,
                            "quantization_resolution": getattr(
                                llm, "quantization_resolution", None
                            ),
                            "loading": getattr(llm, "loading_diagnostics", None),
                            "context": {
                                "name": context_name,
                                "requested_tokens": context_tokens,
                                "provider": "folder" if context_name else None,
                            },
                            "settings": asdict(llm.settings),
                        },
                        indent=2,
                    )
                )
                continue

            if user_input == "/context" or user_input.startswith("/context "):
                try:
                    requested_context, requested_tokens = parse_context_command(
                        user_input
                    )
                except ValueError as exc:
                    if str(exc) == "status":
                        print(
                            json.dumps(
                                {
                                    "context": context_name,
                                    "provider": "folder" if context_name else None,
                                    "requested_context_tokens": context_tokens,
                                    "model_remains_loaded": True,
                                },
                                indent=2,
                            )
                        )
                    else:
                        print(f"Invalid context command: {exc}")
                    continue
                context_name = requested_context
                context_tokens = requested_tokens
                if context_name is None:
                    print("Context disabled. The model remains loaded.")
                else:
                    print(
                        f"Context set to {context_name!r} with a "
                        f"{context_tokens}-token budget. The model remains loaded."
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
                            active_context={
                                "name": context_name,
                                "provider": "folder" if context_name else None,
                                "requested_tokens": context_tokens,
                            },
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
                    context_metadata = None
                    if context_name is not None and context_tokens is not None:
                        if llm.context_length_tokens is None:
                            raise ContextError(
                                "The model does not expose a reliable context limit"
                            )
                        contextual = build_context_prompt(
                            name=context_name,
                            requested_tokens=context_tokens,
                            question=prompt,
                            tokenizer=llm.tokenizer,
                            count_prompt_tokens=llm.count_prompt_tokens,
                            model_context_limit=llm.context_length_tokens,
                            max_new_tokens=llm.settings.max_new_tokens,
                        )
                        prompt = contextual.prompt
                        context_metadata = contextual.metadata
                    last_result = process_prompt(
                        llm,
                        prompt,
                        run_type,
                        benchmark_name,
                        cost_config,
                        context_metadata,
                    )
                except torch.cuda.OutOfMemoryError:
                    print("\nCUDA out of memory.")
                    print("Try lowering max_new_tokens or loading a smaller model.")
                    torch.cuda.empty_cache()
                except GenerationCancelled as exc:
                    print(f"\n{exc}")
                    print("Adjust /context, /set, or other settings and try again.")
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
                    requested_model, requested_profile, options = prepare_load(requested_model, requested_profile, allow_download=False, no_download=False, non_interactive=False, allow_cpu_offload=False)
                    llm.switch_model(requested_model, requested_profile, options)
                except ModelPlacementError as exc:
                    report_model_load_failure(exc)
                    if llm.model is None:
                        print("No working model remains; exiting.")
                        return
                except VLLMEngineError as exc:
                    print(f"Model switch unavailable: {exc}")

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

            except GenerationCancelled as exc:
                print(f"\n{exc}")
                print("Adjust /context, /set, or other settings and try again.")

            except Exception as exc:
                print(f"\nGeneration failed: {type(exc).__name__}: {exc}")

    finally:
        print("\nUnloading model...")
        llm.unload_model()


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    try:
        ensure_engine_runtime(args.engine)
        if args.prepare_context:
            local_only = args.no_download
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_only)
            except (OSError, ValueError):
                processor = AutoProcessor.from_pretrained(args.model, local_files_only=local_only)
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise ContextError("The selected processor does not expose a tokenizer")
            prepared = prepare_folder_context(
                args.prepare_context,
                args.corpus,
                tokenizer,
                chunk_tokens=args.chunk_tokens,
                overlap=args.chunk_overlap,
            )
            print(json.dumps(asdict(prepared), indent=2))
            return
        run(args.model, args.profile, args.config, allow_download=args.allow_download, no_download=args.no_download, non_interactive=args.non_interactive, allow_cpu_offload=args.allow_cpu_offload, context_name=args.context, context_tokens=args.context_tokens, engine=args.engine, kv_cache_dtype=args.kv_cache_dtype, vllm_max_model_len=args.vllm_max_model_len, vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, vllm_prefix_caching=args.vllm_prefix_caching)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
    except ModelPlacementError as exc:
        report_model_load_failure(exc)
    except RuntimeError as exc:
        print(f"Model loading stopped: {exc}")
    except ContextError as exc:
        print(f"Context error: {exc}")
    except VLLMEngineError as exc:
        print(f"vLLM error: {exc}")
