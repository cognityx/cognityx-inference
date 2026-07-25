"""Runtime environment defaults for local inference."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HUGGINGFACE_HOME = Path("/mnt/d/AI/models/huggingface")


def configure_huggingface_cache(
    root: str | Path = DEFAULT_HUGGINGFACE_HOME,
) -> Path | None:
    """Use the shared Windows-mounted cache when it is available.

    Explicit ``HF_HOME`` or ``HF_HUB_CACHE`` values always win. The function
    returns the selected root and is intentionally side-effect-free when the
    shared cache does not exist.
    """
    cache_root = Path(root)
    if not cache_root.is_dir():
        return None
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    return cache_root
