from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_benchmark.reporting import get_gpu_status, get_system_metadata, sanitize_model_name

FAILED_LOAD_DIRECTORY = Path("outputs/failed_loads")

FAILURE_DESCRIPTIONS = {
    "unsupported_model_architecture": (
        "The model configuration is not supported by an available generation loader."
    ),
    "incorrect_auto_model_class": (
        "The selected AutoModel class does not support this model configuration."
    ),
    "cuda_out_of_memory": "CUDA ran out of memory while loading or placing the model.",
    "cpu_offload_required": (
        "Automatic device placement required one or more modules on CPU."
    ),
    "disk_offload_required": (
        "Automatic device placement required one or more modules on disk."
    ),
    "placement_failure": "Automatic device placement could not place the model.",
    "quantization_failure": "The requested quantized loading configuration was rejected.",
    "download_failure": "The tokenizer or model files could not be downloaded or resolved.",
}


class ModelPlacementError(RuntimeError):
    """Expected model-loading failure with user-facing diagnostic context."""

    def __init__(
        self,
        model_name: str,
        profile: str,
        original_exception: BaseException,
        failure_category: str,
        attempted_configuration: dict[str, Any],
        loader_diagnostic: dict[str, Any] | None = None,
        loading_stage: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.profile = profile
        self.original_exception = original_exception
        self.failure_category = failure_category
        self.attempted_configuration = attempted_configuration
        self.loader_diagnostic = loader_diagnostic
        self.loading_stage = loading_stage
        self.diagnostic_message = FAILURE_DESCRIPTIONS[failure_category]
        super().__init__(self.diagnostic_message)


def classify_model_load_failure(exception: BaseException) -> str | None:
    """Return a category only for expected placement/resource/download failures."""
    message = str(exception).casefold()
    class_name = type(exception).__name__.casefold()

    if "unsupportedmodelarchitectureerror" in class_name:
        return "unsupported_model_architecture"
    if (
        "unrecognized configuration class" in message
        and "for automodelfor" in message
    ):
        return "incorrect_auto_model_class"
    if "outofmemory" in class_name or "cuda out of memory" in message:
        return "cuda_out_of_memory"
    if "cpu" in message and any(
        phrase in message for phrase in ("dispatch", "offload", "device_map", "device map")
    ):
        return "cpu_offload_required"
    if "disk" in message and any(
        phrase in message for phrase in ("dispatch", "offload", "device_map", "device map")
    ):
        return "disk_offload_required"
    if any(
        phrase in message
        for phrase in (
            "bitsandbytes",
            "4-bit quantization",
            "8-bit quantization",
            "quantization config",
            "quantization_config",
        )
    ):
        return "quantization_failure"
    if any(
        phrase in message
        for phrase in (
            "device_map",
            "device map",
            "automatic device placement",
            "accelerate",
            "insufficient gpu memory",
            "insufficient memory",
            "not enough memory",
            "cannot fit the model",
            "can't fit the model",
        )
    ):
        return "placement_failure"
    if any(
        phrase in message
        for phrase in (
            "repository not found",
            "not a valid model identifier",
            "couldn't connect",
            "could not connect",
            "connection error",
            "connectionerror",
            "unauthorized",
            "gated repo",
            "localentrynotfounderror",
            "404 client error",
        )
    ):
        return "download_failure"
    return None


def attempted_load_configuration(
    profile: str,
    compute_dtype: Any = None,
    load_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    load_kwargs = load_kwargs or {}
    quantization_config = load_kwargs.get("quantization_config")
    quantization_bits = 4 if profile == "int4" else 8 if profile == "int8" else None
    storage_dtype = getattr(quantization_config, "bnb_4bit_quant_storage", None)
    return {
        "requested_load_profile": profile,
        "effective_load_profile": None,
        "quantization_enabled": quantization_bits is not None,
        "quantization_bits": quantization_bits,
        "quantization_method": "bitsandbytes" if quantization_bits is not None else None,
        "compute_dtype": str(compute_dtype) if compute_dtype is not None else None,
        "storage_dtype": str(storage_dtype) if storage_dtype is not None else None,
    }


def wrap_expected_model_load_failure(
    exception: BaseException,
    model_name: str,
    profile: str,
    attempted_configuration: dict[str, Any],
    loader_diagnostic: dict[str, Any] | None = None,
    loading_stage: str | None = None,
) -> ModelPlacementError | None:
    category = classify_model_load_failure(exception)
    if category is None:
        return None
    return ModelPlacementError(
        model_name,
        profile,
        exception,
        category,
        attempted_configuration,
        loader_diagnostic,
        loading_stage,
    )


def _suggestions(error: ModelPlacementError) -> list[str]:
    if error.failure_category in {
        "unsupported_model_architecture",
        "incorrect_auto_model_class",
    }:
        return [
            "Confirm that the installed Transformers version supports this architecture.",
            "Check the model card for its required Transformers task and processor.",
        ]
    suggestions = ["Use a smaller model."]
    if error.profile != "int4":
        suggestions.append("Try the INT4 loading profile.")
    suggestions.extend(
        [
            "Use multiple GPUs with sufficient combined memory.",
            "Free memory held by other GPU processes and try again.",
        ]
    )
    return suggestions


def format_model_load_failure(error: ModelPlacementError) -> str:
    gpu = get_gpu_status()
    attempted = error.attempted_configuration
    quantization = (
        f"{attempted['quantization_method']} {attempted['quantization_bits']}-bit"
        if attempted.get("quantization_enabled")
        else "disabled"
    )
    gpu_memory = (
        f"{gpu['total_memory_gb']:.2f} GB"
        if gpu["total_memory_gb"] is not None
        else "unavailable"
    )
    placement_summary = ""
    if error.failure_category in {
        "cpu_offload_required",
        "disk_offload_required",
        "placement_failure",
    }:
        placement_summary = (
            "\nDevice placement summary:\n\n"
            "Automatic placement determined that the model cannot be fully resident "
            "on the available GPU configuration.\n"
        )
    moe_note = ""
    if error.failure_category in {
        "cuda_out_of_memory",
        "cpu_offload_required",
        "disk_offload_required",
        "placement_failure",
    }:
        moe_note = (
            "\nPossible reason:\n\n"
            "For Mixture-of-Experts models, sparse expert activation can reduce per-token "
            "computation, but all expert weights may still need memory. Active parameters "
            "do not necessarily reduce the model's memory footprint.\n"
        )
    suggestions = "\n".join(f"• {suggestion}" for suggestion in _suggestions(error))
    architecture_summary = ""
    if error.failure_category in {
        "unsupported_model_architecture",
        "incorrect_auto_model_class",
    }:
        loader = error.loader_diagnostic or {}
        architecture_summary = (
            "\nModel architecture selection failed\n"
            "-----------------------------------\n"
            f"Configuration:       {loader.get('config_class', 'unknown')}\n"
            f"Attempted loader:    {loader.get('auto_model_class', 'none')}\n"
            f"Required task family: {loader.get('task', 'unknown')}\n\n"
            "This failure occurred before GPU memory placement was evaluated.\n"
        )
    return (
        "==================================================\n"
        "Model loading failed\n"
        "==================================================\n\n"
        f"Model:              {error.model_name}\n"
        f"Requested profile:  {error.profile}\n"
        f"Effective profile:  {attempted.get('effective_load_profile') or 'unavailable'}\n"
        f"Quantization:       {quantization}\n"
        f"Compute dtype:      {attempted.get('compute_dtype') or 'unavailable'}\n"
        f"Storage dtype:      {attempted.get('storage_dtype') or 'unavailable'}\n"
        f"CUDA available:     {'yes' if gpu['name'] is not None else 'no'}\n"
        f"GPU:                {gpu['name'] or 'unavailable'}\n"
        f"GPU total memory:   {gpu_memory}\n\n"
        f"Failure: {error.diagnostic_message}\n"
        f"{architecture_summary}"
        f"{placement_summary}"
        f"{moe_note}\n"
        "Suggestions\n"
        "-----------\n"
        f"{suggestions}\n\n"
        "No benchmark was executed."
    )


def save_failed_load(
    error: ModelPlacementError,
    directory: Path = FAILED_LOAD_DIRECTORY,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    stem = (
        f"failed_load_{timestamp:%Y%m%dT%H%M%S_%fZ}_"
        f"{sanitize_model_name(error.model_name)}_{error.profile}"
    )
    path = directory / f"{stem}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{stem}_{suffix}.json"
        suffix += 1

    metadata = get_system_metadata()
    payload = {
        "timestamp_utc": timestamp.isoformat(),
        "model": error.model_name,
        "profile": error.profile,
        "gpu": {
            "name": metadata["gpu_name"],
            "total_memory_gb": metadata["gpu_total_memory_gb"],
        },
        "cuda_version": metadata["cuda_version"],
        "pytorch_version": metadata["pytorch_version"],
        "failure_category": error.failure_category,
        "exception_message": str(error.original_exception),
        "diagnostic_summary": error.diagnostic_message,
        "attempted_configuration": error.attempted_configuration,
        "loading_stage": error.loading_stage,
        "model_loader": error.loader_diagnostic,
    }
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    return path


def report_model_load_failure(error: ModelPlacementError) -> Path | None:
    print(format_model_load_failure(error))
    try:
        path = save_failed_load(error)
    except OSError as save_error:
        print(f"\nCould not save failed-load diagnostics: {save_error}")
        return None
    print(f"\nFailed-load diagnostics: {path}")
    return path
