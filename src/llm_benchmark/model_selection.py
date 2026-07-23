"""Choose the Transformers AutoModel loader supported by a model config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText


class UnsupportedModelArchitectureError(ValueError):
    """Raised when no supported generation AutoModel accepts a configuration."""

    def __init__(self, config: Any) -> None:
        self.config_class = type(config).__name__
        super().__init__(
            f"Unsupported configuration class {self.config_class}: no supported "
            "causal-LM or image-text-to-text AutoModel mapping was found."
        )


@dataclass(frozen=True)
class ModelLoaderSelection:
    """Selected Transformers task, loader class, and tokenizer/processor mode."""
    task: str
    auto_model_class: Any
    selection_source: str
    uses_processor: bool

    def diagnostic(self, config: Any) -> dict[str, Any]:
        """Describe the selection in a JSON-ready form."""
        return {
            "config_class": type(config).__name__,
            "task": self.task,
            "auto_model_class": self.auto_model_class.__name__,
            "selection_source": self.selection_source,
            "text_only_execution": True,
        }


def _mapping_supports(auto_model_class: Any, config: Any) -> bool:
    mapping = getattr(auto_model_class, "_model_mapping", None)
    if mapping is None:
        return False
    try:
        return type(config) in mapping
    except (KeyError, TypeError):
        return False


def select_model_loader(config: Any) -> ModelLoaderSelection:
    """Select a generation loader from Transformers' configuration mappings.

    Args:
        config: Loaded Transformers configuration object.

    Returns:
        A causal-LM or image-text-to-text loader selection.

    Raises:
        UnsupportedModelArchitectureError: If no supported mapping exists.
    """
    if _mapping_supports(AutoModelForCausalLM, config):
        return ModelLoaderSelection(
            task="causal-lm",
            auto_model_class=AutoModelForCausalLM,
            selection_source="transformers_auto_mapping",
            uses_processor=False,
        )
    if _mapping_supports(AutoModelForImageTextToText, config):
        return ModelLoaderSelection(
            task="image-text-to-text",
            auto_model_class=AutoModelForImageTextToText,
            selection_source="transformers_auto_mapping",
            uses_processor=True,
        )

    # Compatibility fallback for Transformers releases where Mistral 3 exists but
    # is not yet exposed through the lazy image-text AutoModel mapping.
    if type(config).__name__ == "Mistral3Config":
        return ModelLoaderSelection(
            task="image-text-to-text",
            auto_model_class=AutoModelForImageTextToText,
            selection_source="explicit_config_class_fallback",
            uses_processor=True,
        )
    raise UnsupportedModelArchitectureError(config)
