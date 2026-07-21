from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

MOE_IDENTIFIER_PARTS = (
    "moe",
    "mixtral",
    "dbrx",
    "switchtransformers",
    "jetmoe",
    "phimoe",
)


def _attribute(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    try:
        return getattr(source, name, None)
    except (AttributeError, RuntimeError):
        return None


def _value(config: Any, *names: str) -> Any:
    for source in (config, _attribute(config, "text_config")):
        for name in names:
            value = _attribute(source, name)
            if value is not None:
                return value
    return None


def _positive_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and value > 0 else None


def extract_moe_config(config: Any) -> dict[str, Any]:
    """Collect common MoE fields without assuming a specific model family."""
    return {
        "num_experts": _positive_int(
            _value(config, "num_experts", "num_local_experts", "n_routed_experts")
        ),
        "num_experts_per_token": _positive_int(
            _value(
                config,
                "num_experts_per_tok",
                "num_experts_per_token",
                "experts_per_token",
                "num_selected_experts",
                "top_k",
            )
        ),
        "moe_intermediate_size": _positive_int(
            _value(config, "moe_intermediate_size", "expert_intermediate_size")
        ),
        "shared_expert_intermediate_size": _positive_int(
            _value(config, "shared_expert_intermediate_size")
        ),
        "num_shared_experts": _positive_int(
            _value(config, "num_shared_experts", "n_shared_experts")
        ),
        "decoder_sparse_step": _positive_int(
            _value(config, "decoder_sparse_step", "moe_layer_frequency")
        ),
        "num_dense_layers": _positive_int(
            _value(config, "num_dense_layers", "first_k_dense_replace")
        ),
        "router_aux_loss_coef": _value(
            config, "router_aux_loss_coef", "router_aux_loss_coefficient"
        ),
    }


def _architecture_identifiers(config: Any) -> list[str]:
    architectures = _value(config, "architectures")
    if isinstance(architectures, str):
        values = [architectures]
    elif isinstance(architectures, (list, tuple)):
        values = [str(value) for value in architectures]
    else:
        values = []
    model_type = _value(config, "model_type")
    if model_type:
        values.append(str(model_type))
    return values


def _family_name(config: Any) -> str | None:
    identifiers = _architecture_identifiers(config)
    if not identifiers:
        return None
    family = re.sub(r"(ForCausalLM|ForConditionalGeneration|Model)$", "", identifiers[0])
    family = re.sub(r"(?i)moe", "", family).replace("_", "-").strip("-")
    return family or identifiers[0]


def classify_architecture(config: Any) -> str:
    moe = extract_moe_config(config)
    identifiers = " ".join(_architecture_identifiers(config)).casefold()
    if (
        (moe["num_experts"] is not None and moe["num_experts"] > 1)
        or moe["num_experts_per_token"] is not None
        or moe["moe_intermediate_size"] is not None
        or moe["num_shared_experts"] is not None
        or moe["decoder_sparse_step"] is not None
        or moe["router_aux_loss_coef"] is not None
        or any(part in identifiers for part in MOE_IDENTIFIER_PARTS)
    ):
        return "moe"

    dense_fields = (
        _positive_int(_value(config, "hidden_size", "n_embd", "d_model")),
        _positive_int(_value(config, "num_hidden_layers", "n_layer", "num_layers")),
        _positive_int(_value(config, "intermediate_size", "ffn_dim", "d_ff")),
    )
    if all(value is not None for value in dense_fields) and not any(moe.values()):
        return "dense"
    if moe["num_experts"] == 1:
        return "dense"
    return "unknown"


def _explicit_active_parameters(config: Any) -> int | None:
    value = _value(
        config,
        "active_parameters_per_token",
        "active_parameter_count",
        "num_active_parameters",
        "active_parameters",
        "num_active_params",
    )
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def estimate_moe_active_parameters(config: Any) -> tuple[int | None, str]:
    """Approximate active MoE weights without scaling the whole model by top-k."""
    moe = extract_moe_config(config)
    hidden = _positive_int(_value(config, "hidden_size", "n_embd", "d_model"))
    layers = _positive_int(
        _value(config, "num_hidden_layers", "n_layer", "num_layers")
    )
    heads = _positive_int(_value(config, "num_attention_heads", "n_head"))
    vocab = _positive_int(_value(config, "vocab_size"))
    selected = moe["num_experts_per_token"]
    expert_intermediate = moe["moe_intermediate_size"]
    num_experts = moe["num_experts"]
    if None in (
        hidden,
        layers,
        heads,
        vocab,
        selected,
        expert_intermediate,
        num_experts,
    ):
        return None, "Insufficient config metadata for an active-parameter estimate."

    key_value_heads = _positive_int(_value(config, "num_key_value_heads")) or heads
    head_dim = hidden // heads
    embedding = vocab * hidden
    output_head = 0 if _value(config, "tie_word_embeddings") else embedding
    attention_per_layer = hidden * (
        hidden + 2 * key_value_heads * head_dim + hidden
    )
    normalization_per_layer = 2 * hidden
    router_per_sparse_layer = hidden * num_experts
    routed_experts_per_sparse_layer = selected * 3 * hidden * expert_intermediate

    sparse_step = moe["decoder_sparse_step"] or 1
    configured_dense_layers = moe["num_dense_layers"] or 0
    sparse_layers = min(layers, max(0, (layers - configured_dense_layers) // sparse_step))
    dense_layers = layers - sparse_layers
    dense_intermediate = _positive_int(
        _value(config, "intermediate_size", "ffn_dim", "d_ff")
    )
    if dense_layers and dense_intermediate is None:
        return None, "Dense-layer intermediate size is unavailable."

    shared_expert_count = moe["num_shared_experts"] or 0
    shared_intermediate = moe["shared_expert_intermediate_size"]
    if shared_expert_count and shared_intermediate is None:
        return None, "Shared-expert count is known but its intermediate size is unavailable."
    shared_per_sparse_layer = (
        shared_expert_count * 3 * hidden * int(shared_intermediate or 0)
    )
    dense_mlp = dense_layers * 3 * hidden * int(dense_intermediate or 0)

    active = (
        embedding
        + output_head
        + layers * (attention_per_layer + normalization_per_layer)
        + hidden
        + sparse_layers
        * (
            router_per_sparse_layer
            + routed_experts_per_sparse_layer
            + shared_per_sparse_layer
        )
        + dense_mlp
    )
    notes = (
        "Approximation from config: embeddings, output head, attention and norms are "
        "always active; selected routed experts and all configured shared experts are "
        "counted per sparse layer. Gated expert MLPs are approximated as three matrices. "
        "Parameter activation does not imply equal computational cost."
    )
    return active, notes


def classify_model_family(
    config: Any,
    logical_parameters: int | None,
    logical_count_reliable: bool,
    model: Any = None,
) -> dict[str, Any]:
    architecture_type = classify_architecture(config)
    moe_config = extract_moe_config(config)
    explicit_active = _explicit_active_parameters(model)
    explicit_source = "model_metadata"
    if explicit_active is None:
        explicit_active = _explicit_active_parameters(config)
        explicit_source = "config"

    if architecture_type == "dense" and logical_parameters is not None:
        active = logical_parameters
        source = "logical_parameter_count"
        reliable = logical_count_reliable
        notes = "Dense model: all logical model parameters are treated as active per token."
    elif explicit_active is not None:
        active = explicit_active
        source = explicit_source
        reliable = True
        notes = "Active parameter count is explicitly supplied by model configuration."
    elif architecture_type == "moe":
        active, notes = estimate_moe_active_parameters(config)
        source = "config_derived_estimate" if active is not None else "unavailable"
        reliable = False
    else:
        active = None
        source = "unavailable"
        reliable = False
        notes = "Architecture or active-parameter metadata is insufficient."

    fraction = (
        active / logical_parameters
        if active is not None and logical_parameters is not None and logical_parameters > 0
        else None
    )
    return {
        "family": _family_name(config),
        "architecture_type": architecture_type,
        "dense_or_moe": architecture_type,
        "logical_parameters": logical_parameters,
        "active_parameters_per_token": active,
        "active_parameters_billions": (
            round(active / 1_000_000_000, 3) if active is not None else None
        ),
        "active_parameter_fraction": round(fraction, 6) if fraction is not None else None,
        "active_parameter_count_source": source,
        "active_parameter_count_reliable": reliable,
        "active_parameter_count_notes": notes,
        "num_experts": moe_config["num_experts"],
        "num_experts_per_token": moe_config["num_experts_per_token"],
        "shared_experts": moe_config["num_shared_experts"],
        "moe_config": moe_config,
    }
