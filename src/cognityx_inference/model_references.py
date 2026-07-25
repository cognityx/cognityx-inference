"""Resolve friendly local model references against the Hugging Face cache."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from cognityx_inference.environment import DEFAULT_HUGGINGFACE_HOME

_SIZE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?[bBmM])(?![A-Za-z0-9])"
)


@dataclass(frozen=True, slots=True)
class ModelReferenceResolution:
    """Requested and canonical model identifiers."""

    requested: str
    resolved: str
    source: str
    candidates: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.requested != self.resolved


def cached_model_ids(cache_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return model IDs represented by Hugging Face cache directories."""
    hub = Path(
        cache_dir
        or os.environ.get("HF_HUB_CACHE")
        or DEFAULT_HUGGINGFACE_HOME / "hub"
    )
    if not hub.is_dir():
        return ()
    values = []
    for path in hub.glob("models--*"):
        if not path.is_dir():
            continue
        encoded = path.name.removeprefix("models--")
        parts = encoded.split("--", 1)
        if len(parts) == 2:
            values.append(f"{parts[0]}/{parts[1]}")
    return tuple(sorted(set(values)))


def resolve_local_model_reference(
    model: str,
    cache_dir: str | Path | None = None,
) -> ModelReferenceResolution:
    """Resolve a unique organization-and-size shorthand from local cache.

    Exact IDs always win. A shorthand is changed only when exactly one cached
    model has the same organization and size marker, such as ``8B``. Ambiguous
    or unrelated names remain unchanged.
    """
    requested = model.strip()
    available = cached_model_ids(cache_dir)
    if requested in available:
        return ModelReferenceResolution(requested, requested, "exact_cache")
    organization, separator, _name = requested.partition("/")
    size_match = _SIZE_PATTERN.search(requested)
    if separator and size_match:
        size = size_match.group(1).lower()
        matches = tuple(
            candidate
            for candidate in available
            if candidate.partition("/")[0].lower() == organization.lower()
            and any(
                match.group(1).lower() == size
                for match in _SIZE_PATTERN.finditer(candidate)
            )
        )
        if len(matches) == 1:
            return ModelReferenceResolution(
                requested,
                matches[0],
                "unique_cached_organization_and_size",
                matches,
            )
        if matches:
            return ModelReferenceResolution(
                requested, requested, "ambiguous_cache_match", matches
            )
    return ModelReferenceResolution(requested, requested, "unresolved")
