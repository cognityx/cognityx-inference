from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

BYTES_PER_GB = 1024**3
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/benchmarks")


def bytes_to_gb(value: int) -> float:
    return round(value / BYTES_PER_GB, 3)


def get_gpu_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "name": None,
            "allocated_gb": None,
            "reserved_gb": None,
            "total_memory_gb": None,
        }

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "allocated_gb": bytes_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved(device)),
        "total_memory_gb": bytes_to_gb(properties.total_memory),
    }


def cuda_model_tensor_bytes(model: Any) -> int | None:
    """Return a lower bound from unique CUDA parameter and buffer tensors."""
    if not torch.cuda.is_available():
        return None
    seen: set[int] = set()
    total = 0
    values = list(model.parameters())
    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        values.extend(buffers())
    for value in values:
        if id(value) in seen:
            continue
        seen.add(id(value))
        if getattr(value, "device", torch.device("cpu")).type != "cuda":
            continue
        try:
            total += int(value.numel()) * int(value.element_size())
        except (AttributeError, RuntimeError, TypeError):
            return None
    return total


def tensor_collection_bytes(values: Any) -> int:
    total = 0
    if hasattr(values, "values"):
        iterable = values.values()
    else:
        iterable = values
    for value in iterable:
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _dtype_bytes(dtype_name: str | None) -> int | None:
    normalized = str(dtype_name or "").lower()
    if "float64" in normalized:
        return 8
    if any(name in normalized for name in ("bfloat16", "float16", "half")):
        return 2
    if any(name in normalized for name in ("float32", "int32")):
        return 4
    if any(name in normalized for name in ("float8", "int8", "uint8")):
        return 1
    return None


def estimate_kv_cache_bytes(
    architecture: dict[str, Any],
    sequence_tokens: int,
    dtype_name: str | None,
) -> tuple[int | None, dict[str, Any]]:
    layers = architecture.get("num_hidden_layers")
    hidden = architecture.get("hidden_size")
    attention_heads = architecture.get("num_attention_heads")
    kv_heads = architecture.get("num_key_value_heads") or attention_heads
    bytes_per_element = _dtype_bytes(dtype_name)
    values = (layers, hidden, attention_heads, kv_heads, sequence_tokens, bytes_per_element)
    if not all(isinstance(value, int) and value > 0 for value in values):
        return None, {"available": False, "reason": "Required architecture or cache-dtype metadata is unavailable."}
    head_dim = hidden // attention_heads
    estimated = layers * 2 * kv_heads * head_dim * sequence_tokens * bytes_per_element
    return estimated, {
        "available": True,
        "formula": "layers * 2(K+V) * kv_heads * head_dim * sequence_tokens * bytes_per_element",
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "sequence_tokens": sequence_tokens,
        "bytes_per_element": bytes_per_element,
        "batch_size": 1,
    }


def build_gpu_memory_breakdown(
    *,
    allocated_before_gb: float | None,
    allocated_after_gb: float | None,
    reserved_after_gb: float | None,
    peak_allocated_gb: float | None,
    cuda_model_bytes: int | None,
    prompt_tensor_bytes: int,
    kv_cache_bytes: int | None,
    kv_estimate_details: dict[str, Any],
) -> dict[str, Any]:
    model_gb = bytes_to_gb(cuda_model_bytes) if cuda_model_bytes is not None else None
    prompt_gb = bytes_to_gb(prompt_tensor_bytes)
    other_after = (
        round(max(0.0, allocated_after_gb - model_gb - prompt_gb), 3)
        if allocated_after_gb is not None and model_gb is not None
        else None
    )
    kv_gb = bytes_to_gb(kv_cache_bytes) if kv_cache_bytes is not None else None
    transient = (
        round(max(0.0, peak_allocated_gb - allocated_before_gb - (kv_gb or 0.0)), 3)
        if peak_allocated_gb is not None and allocated_before_gb is not None
        else None
    )
    unused_reserved = (
        round(max(0.0, reserved_after_gb - allocated_after_gb), 3)
        if reserved_after_gb is not None and allocated_after_gb is not None
        else None
    )
    return {
        "allocated_after_gb": allocated_after_gb,
        "reserved_after_gb": reserved_after_gb,
        "cuda_model_parameter_and_buffer_tensors_gb": model_gb,
        "cuda_model_tensor_measurement": "lower_bound; fused/quantization state may not be registered as ordinary tensors",
        "encoded_prompt_tensors_gb": prompt_gb,
        "estimated_kv_cache_at_final_sequence_gb": kv_gb,
        "estimated_kv_cache_details": kv_estimate_details,
        "retained_kv_cache_after_generation_gb": None,
        "retained_kv_cache_note": "model.generate returned token IDs only; PyTorch cannot attribute whether any cache storage remains after return",
        "other_allocated_after_gb": other_after,
        "unused_reserved_allocator_gb": unused_reserved,
        "estimated_peak_transient_runtime_gb": transient,
        "prefill_note": "Prefill uses the KV cache plus temporary activations/workspaces; it is not a separate persistent cache and cannot be isolated by the PyTorch allocator.",
        "accuracy": "mixed measured and estimated; values are not additive proof of allocator ownership",
    }


def get_system_metadata() -> dict[str, Any]:
    gpu = get_gpu_status()
    return {
        "gpu_name": gpu["name"],
        "gpu_total_memory_gb": gpu["total_memory_gb"],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
    }


def benchmark_filename(timestamp: datetime | None = None) -> str:
    timestamp = timestamp or datetime.now(timezone.utc)
    utc_timestamp = timestamp.astimezone(timezone.utc)
    return f"benchmark_{utc_timestamp:%Y%m%dT%H%M%S_%fZ}.json"


def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.strip().replace("/", "--").replace("\\", "--")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", sanitized)
    sanitized = sanitized.strip(".-")
    return sanitized or "unknown-model"


def save_benchmark_result(
    result: dict[str, Any],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    model_directory = output_directory / sanitize_model_name(
        str(result.get("model", "unknown-model"))
    )
    profile = str(
        result.get("effective_load_profile")
        or result.get("requested_load_profile")
        or "unknown-profile"
    )
    profile_directory = model_directory / sanitize_model_name(profile)
    profile_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = profile_directory / benchmark_filename(timestamp)
    suffix = 1

    while path.exists():
        path = profile_directory / (
            f"{Path(benchmark_filename(timestamp)).stem}_{suffix}.json"
        )
        suffix += 1

    with path.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    return path
