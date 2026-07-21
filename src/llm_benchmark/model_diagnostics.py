from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from llm_benchmark.model_size import calculate_model_size


def _attribute(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    try:
        return getattr(source, name, None)
    except (AttributeError, RuntimeError):
        return None


def _config_value(config: Any, *names: str) -> Any:
    for source in (config, _attribute(config, "text_config")):
        for name in names:
            value = _attribute(source, name)
            if value is not None:
                return value
    return None


def normalize_device(device: Any) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, torch.device):
        return str(device)
    return str(device).lower()


def analyze_device_map(
    device_map: Mapping[str, Any] | None,
    parameter_devices: Iterable[Any] = (),
) -> dict[str, Any]:
    normalized_map = (
        {str(module): normalize_device(device) for module, device in device_map.items()}
        if device_map
        else None
    )
    mapped_devices = set(normalized_map.values()) if normalized_map else set()
    observed_devices = {normalize_device(device) for device in parameter_devices}
    devices = mapped_devices or observed_devices

    uses_cpu_offload = "cpu" in mapped_devices
    uses_disk_offload = "disk" in mapped_devices
    non_cuda_devices = sorted(
        device
        for device in devices
        if not device.startswith("cuda") and device != "disk"
    )

    return {
        "device_map": normalized_map,
        "devices_used": sorted(devices),
        "uses_cpu_offload": uses_cpu_offload,
        "uses_disk_offload": uses_disk_offload,
        "uses_non_cuda_device": bool(non_cuda_devices or uses_disk_offload),
        "non_cuda_devices": non_cuda_devices,
    }


def detect_attention_implementation(model: Any) -> str:
    config = _attribute(model, "config")
    for source, name in (
        (config, "_attn_implementation"),
        (config, "attn_implementation"),
        (model, "_attn_implementation"),
        (model, "attn_implementation"),
    ):
        value = _attribute(source, name)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def extract_cache_configuration(model: Any) -> dict[str, Any]:
    generation_config = _attribute(model, "generation_config")
    model_config = _attribute(model, "config")

    use_cache = _attribute(generation_config, "use_cache")
    cache_source = "generation_config"
    if use_cache is None:
        use_cache = _config_value(model_config, "use_cache")
        cache_source = "model_config" if use_cache is not None else "unknown"

    cache_implementation = _attribute(generation_config, "cache_implementation")
    implementation_source = "generation_config"
    if cache_implementation is None:
        cache_implementation = _config_value(model_config, "cache_implementation")
        implementation_source = (
            "model_config" if cache_implementation is not None else "default"
        )

    implementation_explicit = cache_implementation is not None
    if cache_implementation is None:
        if use_cache is True:
            cache_implementation = "default"
        elif use_cache is False:
            cache_implementation = "disabled"
        else:
            cache_implementation = "unknown"

    return {
        "use_cache": use_cache,
        "use_cache_source": cache_source,
        "cache_implementation": str(cache_implementation),
        "cache_implementation_explicit": implementation_explicit,
        "cache_implementation_source": implementation_source,
    }


def extract_architecture_metadata(
    config: Any,
    total_parameters: int | None,
) -> dict[str, Any]:
    architectures = _config_value(config, "architectures")
    if isinstance(architectures, str):
        architecture = architectures
    else:
        architecture = architectures[0] if architectures else None
    rope_scaling = _config_value(config, "rope_scaling")
    rope_theta = _config_value(config, "rope_theta")

    return {
        "model_type": _config_value(config, "model_type"),
        "architecture": architecture,
        "total_parameters": total_parameters,
        "num_hidden_layers": _config_value(
            config, "num_hidden_layers", "n_layer", "num_layers"
        ),
        "hidden_size": _config_value(config, "hidden_size", "n_embd", "d_model"),
        "intermediate_size": _config_value(
            config, "intermediate_size", "ffn_dim", "d_ff"
        ),
        "num_attention_heads": _config_value(
            config, "num_attention_heads", "n_head"
        ),
        "num_key_value_heads": _config_value(config, "num_key_value_heads"),
        "vocab_size": _config_value(config, "vocab_size"),
        "max_position_embeddings": _config_value(
            config, "max_position_embeddings", "n_positions", "max_sequence_length"
        ),
        "rope": {
            "theta": rope_theta,
            "scaling": rope_scaling,
        },
    }


def extract_quantization_metadata(
    model: Any,
    requested_profile: str,
    requested_compute_dtype: Any = None,
) -> dict[str, Any]:
    config = _attribute(model, "config")
    quantization_config = _attribute(config, "quantization_config")
    if quantization_config is None:
        quantization_config = _attribute(model, "quantization_config")

    loaded_in_4bit = bool(
        _attribute(model, "is_loaded_in_4bit")
        or _attribute(quantization_config, "load_in_4bit")
    )
    loaded_in_8bit = bool(
        _attribute(model, "is_loaded_in_8bit")
        or _attribute(quantization_config, "load_in_8bit")
    )

    if loaded_in_4bit:
        effective_profile = "int4"
        quantization_bits = 4
    elif loaded_in_8bit:
        effective_profile = "int8"
        quantization_bits = 8
    else:
        effective_profile = requested_profile
        quantization_bits = None

    quantization_enabled = quantization_bits is not None
    quantization_method = _attribute(quantization_config, "quant_method")
    if quantization_enabled and quantization_method is None:
        quantization_method = "bitsandbytes"
    quantization_method = _attribute(quantization_method, "value") or quantization_method

    compute_dtype = _attribute(quantization_config, "bnb_4bit_compute_dtype")
    if compute_dtype is None:
        compute_dtype = requested_compute_dtype

    storage_dtype = _attribute(quantization_config, "bnb_4bit_quant_storage")
    if storage_dtype is None and not quantization_enabled:
        storage_dtype = requested_compute_dtype

    return {
        "requested_load_profile": requested_profile,
        "effective_load_profile": effective_profile,
        "quantization_enabled": quantization_enabled,
        "quantization_bits": quantization_bits,
        "quantization_method": (
            str(quantization_method) if quantization_method is not None else None
        ),
        "compute_dtype": str(compute_dtype) if compute_dtype is not None else None,
        "storage_dtype": str(storage_dtype) if storage_dtype is not None else None,
    }


def collect_model_runtime(
    model: Any,
    load_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dtype_values: set[str] = set()
    parameter_devices: set[Any] = set()
    for parameter in model.parameters():
        dtype_values.add(str(parameter.dtype))
        parameter_devices.add(parameter.device)

    parameter_dtypes = sorted(dtype_values)

    if len(parameter_dtypes) == 1:
        model_dtype = parameter_dtypes[0]
    elif parameter_dtypes:
        model_dtype = "mixed"
    else:
        configured_dtype = _config_value(_attribute(model, "config"), "torch_dtype")
        model_dtype = str(configured_dtype) if configured_dtype is not None else "unknown"

    device_placement = analyze_device_map(
        _attribute(model, "hf_device_map"),
        parameter_devices,
    )
    model_size = calculate_model_size(model, load_profile)
    model_size["uses_mixed_dtypes"] = len(parameter_dtypes) > 1

    return {
        **device_placement,
        "model_dtype": model_dtype,
        "parameter_dtypes": parameter_dtypes,
        "uses_mixed_dtypes": len(parameter_dtypes) > 1,
        "load_profile": load_profile,
        "model_size": model_size,
        "attention_implementation": detect_attention_implementation(model),
        "kv_cache": extract_cache_configuration(model),
        "architecture": extract_architecture_metadata(
            _attribute(model, "config"),
            model_size["logical_parameters"],
        ),
    }
