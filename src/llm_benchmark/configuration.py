from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

from llm_benchmark.llm import GenerationSettings


class ConfigurationError(ValueError):
    """Raised when the application configuration is invalid."""


def _validate_setting(name: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigurationError(f"generation.{name} must be true or false")
        return value
    if isinstance(default, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigurationError(f"generation.{name} must be an integer")
        return value
    if isinstance(default, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigurationError(f"generation.{name} must be a number")
        return float(value)
    raise ConfigurationError(f"generation.{name} has an unsupported value type")


def load_generation_settings(path: Path) -> GenerationSettings:
    """Load generation defaults from the ``[generation]`` TOML table."""
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"Invalid TOML in {path}: {exc}") from exc

    generation = document.get("generation", {})
    if not isinstance(generation, dict):
        raise ConfigurationError("[generation] must be a TOML table")

    defaults = GenerationSettings()
    known_fields = {field.name for field in fields(GenerationSettings)}
    unknown = sorted(set(generation) - known_fields)
    if unknown:
        raise ConfigurationError(
            f"Unknown generation setting(s): {', '.join(unknown)}"
        )

    values = {
        name: _validate_setting(name, generation[name], getattr(defaults, name))
        for name in generation
    }
    settings = GenerationSettings(**values)
    if settings.max_new_tokens <= 0:
        raise ConfigurationError("generation.max_new_tokens must be greater than zero")
    if settings.temperature < 0:
        raise ConfigurationError("generation.temperature cannot be negative")
    if not 0 < settings.top_p <= 1:
        raise ConfigurationError("generation.top_p must be greater than 0 and at most 1")
    if settings.top_k < 0:
        raise ConfigurationError("generation.top_k cannot be negative")
    if settings.repetition_penalty <= 0:
        raise ConfigurationError("generation.repetition_penalty must be greater than zero")
    return settings
