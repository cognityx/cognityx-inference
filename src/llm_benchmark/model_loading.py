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
    "conflicting_quantization_config": (
        "A runtime quantization configuration conflicted with checkpoint-native quantization."
    ),
    "checkpoint_not_quantized": (
        "The native profile was requested, but the checkpoint is not natively quantized."
    ),
    "unsupported_native_quantization": (
        "Transformers does not support the checkpoint's native quantization method."
    ),
    "native_quantization_dependency_missing": (
        "A dependency required by the checkpoint's native quantization backend is missing."
    ),
    "native_quantization_hardware_unsupported": (
        "The native quantization backend rejected the available hardware."
    ),
    "quantization_backend_rejected": "The selected quantization backend rejected the load.",
    "bitsandbytes_unavailable": "The requested profile requires bitsandbytes.",
    "gated_repository": (
        "The Hugging Face repository is gated and access has not been granted."
    ),
    "authentication_required": (
        "Hugging Face authentication is required or the supplied credentials are invalid."
    ),
    "repository_not_found": "The Hugging Face model repository could not be found.",
    "network_failure": "A network or Hugging Face Hub availability error occurred.",
    "missing_model_file": "A required model file could not be found.",
    "missing_processor_metadata": "Required tokenizer or processor metadata is missing.",
    "download_failure": "The tokenizer or model files could not be downloaded or resolved.",
    "model_load_runtime_error": "Model initialization failed.",
}

FAILURE_CODE_TO_CATEGORY = {
    "gated_repository": "authentication",
    "authentication_required": "authentication",
    "repository_not_found": "download",
    "network_failure": "download",
    "missing_model_file": "download",
    "download_failure": "download",
    "missing_processor_metadata": "preprocessing",
    "unsupported_model_architecture": "architecture",
    "incorrect_auto_model_class": "architecture",
    "conflicting_quantization_config": "quantization",
    "checkpoint_not_quantized": "quantization",
    "unsupported_native_quantization": "quantization",
    "native_quantization_dependency_missing": "quantization",
    "native_quantization_hardware_unsupported": "quantization",
    "quantization_backend_rejected": "quantization",
    "bitsandbytes_unavailable": "quantization",
    "cpu_offload_required": "placement",
    "disk_offload_required": "placement",
    "placement_failure": "placement",
    "cuda_out_of_memory": "memory",
    "model_load_runtime_error": "unknown",
}

HUB_FAILURE_CATEGORIES = {
    "gated_repository",
    "authentication_required",
    "repository_not_found",
    "network_failure",
    "download_failure",
}


class ModelPlacementError(RuntimeError):
    """Expected model-loading failure with user-facing diagnostic context."""

    def __init__(
        self,
        model_name: str,
        profile: str,
        original_exception: BaseException,
        failure_code: str,
        attempted_configuration: dict[str, Any],
        loader_diagnostic: dict[str, Any] | None = None,
        loading_stage: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.profile = profile
        self.original_exception = original_exception
        self.failure_code = failure_code
        self.failure_category = FAILURE_CODE_TO_CATEGORY[failure_code]
        self.attempted_configuration = attempted_configuration
        self.loader_diagnostic = loader_diagnostic
        self.loading_stage = loading_stage
        self.diagnostic_message = FAILURE_DESCRIPTIONS[failure_code]
        super().__init__(self.diagnostic_message)


def _exception_chain(exception: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _http_status(exception: BaseException) -> int | None:
    response = getattr(exception, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def classify_model_load_failure(exception: BaseException) -> str | None:
    """Return a category only for expected placement/resource/download failures."""
    chain = _exception_chain(exception)
    message = "\n".join(str(item) for item in chain).casefold()
    class_names = {type(item).__name__.casefold() for item in chain}
    statuses = {_http_status(item) for item in chain}

    if "unsupportedmodelarchitectureerror" in class_names:
        return "unsupported_model_architecture"
    if (
        "unrecognized configuration class" in message
        and "for automodelfor" in message
    ):
        return "incorrect_auto_model_class"
    if any("outofmemory" in name for name in class_names) or "cuda out of memory" in message:
        return "cuda_out_of_memory"
    if "cpu" in message and any(
        phrase in message for phrase in ("dispatch", "offload", "device_map", "device map")
    ):
        return "cpu_offload_required"
    if "disk" in message and any(
        phrase in message for phrase in ("dispatch", "offload", "device_map", "device map")
    ):
        return "disk_offload_required"
    if "nativequantizationrequirederror" in class_names:
        return "checkpoint_not_quantized"
    if "bitsandbytesunavailableerror" in class_names:
        return "bitsandbytes_unavailable"
    if "triton" in message and any(
        phrase in message
        for phrase in ("no module named", "not installed", "is required", "missing")
    ):
        return "native_quantization_dependency_missing"
    if any(
        phrase in message
        for phrase in (
            "compute capability",
            "unsupported gpu",
            "gpu is not supported",
            "hardware is not supported",
            "sm version",
        )
    ) and any(phrase in message for phrase in ("mxfp", "quant", "triton")):
        return "native_quantization_hardware_unsupported"
    if (
        "quantization_config" in message
        and any(phrase in message for phrase in ("already quantized", "conflict", "pre-quantized"))
    ):
        return "conflicting_quantization_config"
    if any(
        phrase in message
        for phrase in (
            "unsupported quantization method",
            "unknown quantization type",
            "unknown quantization method",
            "no quantizer found",
        )
    ):
        return "unsupported_native_quantization"
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
        return "quantization_backend_rejected"
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
    if "gatedrepoerror" in class_names or any(
        phrase in message
        for phrase in (
            "gated repo",
            "gated repository",
            "cannot access gated",
            "access to model is restricted",
            "accept the license",
            "accept the licence",
        )
    ):
        return "gated_repository"
    if statuses.intersection({401, 403}) or any(
        phrase in message
        for phrase in (
            "401 client error",
            "401 unauthorized",
            "unauthorized",
            "authentication required",
            "invalid token",
            "invalid credentials",
        )
    ):
        return "authentication_required"
    if "repositorynotfounderror" in class_names or 404 in statuses or any(
        phrase in message
        for phrase in (
            "repository not found",
            "not a valid model identifier",
            "404 client error",
        )
    ):
        return "repository_not_found"
    if any(
        name in class_names
        for name in (
            "connectionerror",
            "connecterror",
            "connecttimeouterror",
            "readtimeout",
            "proxyerror",
        )
    ) or any(
        phrase in message
        for phrase in (
            "couldn't connect",
            "could not connect",
            "connection error",
            "temporary failure in name resolution",
            "name or service not known",
            "network is unreachable",
            "connection timed out",
            "hub is currently unavailable",
        )
    ):
        return "network_failure"
    if any(
        phrase in message
        for phrase in (
            "preprocessor_config.json",
            "can't load image processor",
            "can't load tokenizer",
        )
    ):
        return "missing_processor_metadata"
    if "localentrynotfounderror" in class_names or any(
        phrase in message
        for phrase in (
            "can't load the configuration",
            "could not locate the requested files",
            "error while downloading",
        )
    ):
        return "download_failure"
    if any(
        phrase in message
        for phrase in (
            "model.safetensors not found",
            "pytorch_model.bin not found",
            "no weight file",
            "does not appear to have a file named",
        )
    ):
        return "missing_model_file"
    return None


def attempted_load_configuration(
    profile: str,
    compute_dtype: Any = None,
    load_kwargs: dict[str, Any] | None = None,
    resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    load_kwargs = load_kwargs or {}
    quantization_config = load_kwargs.get("quantization_config")
    quantization_bits = 4 if profile == "int4" else 8 if profile == "int8" else None
    storage_dtype = getattr(quantization_config, "bnb_4bit_quant_storage", None)
    attempted = {
        "requested_load_profile": profile,
        "effective_load_profile": None,
        "quantization_enabled": quantization_bits is not None,
        "quantization_bits": quantization_bits,
        "quantization_method": "bitsandbytes" if quantization_bits is not None else None,
        "compute_dtype": str(compute_dtype) if compute_dtype is not None else None,
        "storage_dtype": str(storage_dtype) if storage_dtype is not None else None,
    }
    if resolution is not None:
        attempted["quantization_resolution"] = resolution
        attempted["effective_load_profile"] = resolution.get("effective_profile")
        attempted["quantization_enabled"] = bool(
            resolution.get("checkpoint_quantized")
            or resolution.get("runtime_quantization_method")
        )
        attempted["quantization_method"] = (
            resolution.get("checkpoint_quantization_method")
            or resolution.get("runtime_quantization_method")
        )
        if attempted["quantization_method"] == "mxfp4":
            attempted["quantization_bits"] = 4
    return attempted


def wrap_expected_model_load_failure(
    exception: BaseException,
    model_name: str,
    profile: str,
    attempted_configuration: dict[str, Any],
    loader_diagnostic: dict[str, Any] | None = None,
    loading_stage: str | None = None,
) -> ModelPlacementError | None:
    category = classify_model_load_failure(exception)
    if category is None and loading_stage == "model_initialization":
        category = "model_load_runtime_error"
    if category is None:
        return None
    effective_stage = loading_stage
    failure_category = FAILURE_CODE_TO_CATEGORY[category]
    if loading_stage == "model_initialization" and failure_category in {
        "authentication",
        "download",
    }:
        effective_stage = "model_download"
    elif loading_stage == "model_initialization" and failure_category == "placement":
        effective_stage = "device_mapping"
    return ModelPlacementError(
        model_name,
        profile,
        exception,
        category,
        attempted_configuration,
        loader_diagnostic,
        effective_stage,
    )


def _suggestions(error: ModelPlacementError) -> list[str]:
    if error.failure_code in {
        "unsupported_model_architecture",
        "incorrect_auto_model_class",
    }:
        return [
            "Confirm that the installed Transformers version supports this architecture.",
            "Check the model card for its required Transformers task and processor.",
        ]
    if error.failure_code == "gated_repository":
        return [
            "Open the model page, accept its licence or access conditions, and wait for approval if required.",
            "Authenticate with `hf auth login` or set an HF_TOKEN that has access to the repository.",
        ]
    if error.failure_code == "authentication_required":
        return [
            "Authenticate with `hf auth login` or set HF_TOKEN.",
            "Verify that the token is valid and has permission to read this repository.",
        ]
    if error.failure_code == "repository_not_found":
        return [
            "Verify the model ID, spelling, and requested revision.",
            "If the repository is private, authenticate with an account that can access it.",
        ]
    if error.failure_code == "network_failure":
        return [
            "Check DNS, proxy, firewall, and internet connectivity.",
            "Check Hugging Face Hub availability and retry the request.",
        ]
    if error.failure_code == "download_failure":
        return [
            "The repository may be gated and its licence may not have been accepted.",
            "Hugging Face authentication may be missing or invalid.",
            "The model ID or revision may be incorrect.",
            "A network or Hugging Face Hub availability problem may have occurred.",
        ]
    if error.failure_category == "quantization":
        return [
            "Review the technical detail and quantization-resolution fields below.",
            "Verify backend dependencies and hardware support for the effective profile.",
        ]
    if error.failure_category in {
        "authentication",
        "download",
        "preprocessing",
        "architecture",
        "unknown",
    }:
        return ["Review the technical detail and failure stage above."]
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
    quantization = "disabled"
    if attempted.get("quantization_enabled"):
        method = attempted.get("quantization_method") or "unknown"
        bits = attempted.get("quantization_bits")
        quantization = f"{method} {bits}-bit" if bits is not None else method
    gpu_memory = (
        f"{gpu['total_memory_gb']:.2f} GB"
        if gpu["total_memory_gb"] is not None
        else "unavailable"
    )
    placement_summary = ""
    if error.failure_code in {
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
    if error.failure_code in {
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
    advice_heading = "Possible causes" if error.failure_code == "download_failure" else "Suggestions"
    architecture_summary = ""
    if error.failure_code in {
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
    hub_summary = ""
    if error.failure_code in HUB_FAILURE_CATEGORIES:
        hub_summary = (
            "\nThis failure occurred while resolving or downloading model artifacts, "
            "before GPU memory placement was evaluated.\n"
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
        f"Failure stage:      {error.loading_stage or 'unknown'}\n"
        f"Failure category:   {error.failure_category}\n"
        f"Failure code:       {error.failure_code}\n\n"
        "Summary\n"
        "-------\n"
        f"{error.diagnostic_message}\n"
        "\nTechnical detail\n"
        "----------------\n"
        f"{error.original_exception}\n"
        f"{architecture_summary}"
        f"{hub_summary}"
        f"{placement_summary}"
        f"{moe_note}\n"
        f"{advice_heading}\n"
        f"{'-' * len(advice_heading)}\n"
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
    exception_chain = [
        {"exception_type": type(item).__name__, "exception_message": str(item)}
        for item in _exception_chain(error.original_exception)
    ]
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
        "exception_type": type(error.original_exception).__name__,
        "exception_message": str(error.original_exception),
        "exception_chain": exception_chain,
        "failure_stage": error.loading_stage,
        "failure_category": error.failure_category,
        "failure_code": error.failure_code,
        "bf16_dequantization_fallback_attempted": any(
            phrase in str(error.original_exception).casefold()
            for phrase in ("dequantize to bfloat16", "bf16 dequantization", "bfloat16 fallback")
        ),
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
