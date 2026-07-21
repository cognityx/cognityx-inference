from __future__ import annotations

from collections.abc import Iterable
from typing import Any

BYTES_PER_GIB = 1024**3


def bytes_to_gib(value: int | float) -> float:
    return round(value / BYTES_PER_GIB, 3)


def nominal_bits_for_profile(profile: str) -> int | None:
    return {"bf16": 16, "fp16": 16, "int8": 8, "int4": 4}.get(profile)


def _attribute(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    try:
        return getattr(source, name, None)
    except (AttributeError, RuntimeError):
        return None


def _unique(values: Iterable[Any]) -> list[Any]:
    seen: set[int] = set()
    unique_values: list[Any] = []
    for value in values:
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            unique_values.append(value)
    return unique_values


def _is_quantized_parameter(parameter: Any, quantization_bits: int | None) -> bool:
    if quantization_bits is None:
        return False
    class_name = type(parameter).__name__.lower()
    return bool(
        "params4bit" in class_name
        or "int8params" in class_name
        or _attribute(parameter, "quant_state") is not None
        or _attribute(parameter, "quant_type") is not None
    )


def _logical_numel(parameter: Any, quantization_bits: int | None) -> int:
    stored_elements = int(parameter.numel())
    if not _is_quantized_parameter(parameter, quantization_bits):
        return stored_elements

    element_size = _attribute(parameter, "element_size")
    stored_bits = int(element_size() * 8) if callable(element_size) else 8
    packing_ratio = max(1, stored_bits // int(quantization_bits or stored_bits))
    return stored_elements * packing_ratio


def _metadata_parameter_count(model: Any) -> int | None:
    config = _attribute(model, "config")
    for source in (model, config):
        for name in ("logical_parameter_count", "num_parameters", "parameter_count"):
            value = _attribute(source, name)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _config_value(config: Any, *names: str) -> Any:
    text_config = _attribute(config, "text_config")
    for source in (config, text_config):
        for name in names:
            value = _attribute(source, name)
            if value is not None:
                return value
    return None


def _config_derived_parameter_count(model: Any) -> int | None:
    """Approximate a gated decoder-only transformer from common config fields."""
    config = _attribute(model, "config")
    hidden = _config_value(config, "hidden_size", "n_embd", "d_model")
    layers = _config_value(config, "num_hidden_layers", "n_layer", "num_layers")
    intermediate = _config_value(config, "intermediate_size", "ffn_dim", "d_ff")
    heads = _config_value(config, "num_attention_heads", "n_head")
    vocab = _config_value(config, "vocab_size")
    if not all(isinstance(value, int) and value > 0 for value in (
        hidden, layers, intermediate, heads, vocab
    )):
        return None

    key_value_heads = _config_value(config, "num_key_value_heads") or heads
    head_dim = hidden // heads
    embedding = vocab * hidden
    attention_per_layer = hidden * (
        hidden + 2 * key_value_heads * head_dim + hidden
    )
    gated_mlp_per_layer = 3 * hidden * intermediate
    normalization = (2 * layers + 1) * hidden
    output_head = 0 if _config_value(config, "tie_word_embeddings") else embedding
    return embedding + output_head + layers * (
        attention_per_layer + gated_mlp_per_layer
    ) + normalization


def _tensor_storage_bytes(values: Iterable[Any]) -> int | None:
    total = 0
    for value in values:
        element_size = _attribute(value, "element_size")
        device = str(_attribute(value, "device") or "")
        if not callable(element_size) or device == "meta":
            return None
        total += int(value.numel()) * int(element_size())
    return total


def calculate_model_size(
    model: Any,
    load_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate logical weight count from physical tensor storage."""
    load_profile = load_profile or {}
    requested_profile = str(load_profile.get("requested_load_profile") or "unknown")
    effective_profile = str(load_profile.get("effective_load_profile") or requested_profile)
    quantization_bits = load_profile.get("quantization_bits")
    quantization_enabled = bool(load_profile.get("quantization_enabled"))
    parameters = _unique(model.parameters())

    metadata_count = _metadata_parameter_count(model)
    quantized_parameters = [
        parameter
        for parameter in parameters
        if _is_quantized_parameter(parameter, quantization_bits)
    ]

    if not quantization_enabled:
        native_count = _attribute(model, "num_parameters")
        if callable(native_count):
            logical_parameters = int(native_count())
            count_source = "model_native_num_parameters"
        else:
            logical_parameters = sum(int(parameter.numel()) for parameter in parameters)
            count_source = "unique_parameter_tensors"
        count_reliable = True
    elif metadata_count is not None:
        logical_parameters = metadata_count
        count_source = "model_metadata"
        count_reliable = True
    elif quantized_parameters:
        logical_parameters = sum(
            _logical_numel(parameter, quantization_bits) for parameter in parameters
        )
        count_source = "quantization_aware_parameter_tensors"
        count_reliable = True
    else:
        config_count = _config_derived_parameter_count(model)
        logical_parameters = (
            config_count
            if config_count is not None
            else sum(int(parameter.numel()) for parameter in parameters)
        )
        count_source = (
            "config_derived_causal_lm_approximation"
            if config_count is not None
            else "packed_parameter_fallback"
        )
        count_reliable = False

    nominal_bits = nominal_bits_for_profile(effective_profile)
    estimated_bytes = (
        round(logical_parameters * nominal_bits / 8)
        if nominal_bits is not None
        else None
    )

    actual_storage_bytes: int | None
    if quantization_enabled:
        actual_storage_bytes = None
        actual_storage_note = (
            "Quantized wrapper storage is not reported as actual physical size because "
            "packed weights and quantization state are not exposed uniformly."
        )
    else:
        buffers = _unique(model.buffers()) if callable(_attribute(model, "buffers")) else []
        actual_storage_bytes = _tensor_storage_bytes(_unique([*parameters, *buffers]))
        actual_storage_note = (
            "Sum of unique parameter and persistent buffer tensor bytes."
            if actual_storage_bytes is not None
            else "Tensor byte size was not reliably available for every parameter or buffer."
        )

    estimate_note = (
        "Nominal weight-only estimate. It excludes quantization scales and zero points, "
        "packing metadata, non-quantized layers, allocator overhead, KV cache, activations, "
        "and runtime buffers; it is not an estimate of total VRAM usage."
    )

    return {
        "logical_parameters": logical_parameters,
        "logical_parameters_billions": round(logical_parameters / 1_000_000_000, 3),
        "parameter_count_source": count_source,
        "parameter_count_reliable": count_reliable,
        "requested_profile": requested_profile,
        "effective_profile": effective_profile,
        "nominal_bits_per_weight": nominal_bits,
        "estimated_raw_weight_storage_bytes": estimated_bytes,
        "estimated_raw_weight_storage_gib": (
            bytes_to_gib(estimated_bytes) if estimated_bytes is not None else None
        ),
        "actual_parameter_storage_bytes": actual_storage_bytes,
        "actual_parameter_storage_gib": (
            bytes_to_gib(actual_storage_bytes)
            if actual_storage_bytes is not None
            else None
        ),
        "actual_storage_notes": actual_storage_note,
        "storage_estimate_notes": estimate_note,
        "compute_dtype": load_profile.get("compute_dtype"),
        "storage_dtype": load_profile.get("storage_dtype"),
        "uses_mixed_dtypes": None,
        "quantization_method": load_profile.get("quantization_method"),
        "quantization_bits": quantization_bits,
    }
