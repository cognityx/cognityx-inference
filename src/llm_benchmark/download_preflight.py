"""Inspect Hugging Face checkpoint downloads before loading a model."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download


ALLOW_PATTERNS = (
    "*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.tiktoken",
    "tokenizer.*", "merges.txt", "vocab.*", "chat_template*",
)
IGNORE_PATTERNS = (
    "*.onnx", "onnx/*", "*.msgpack", "flax_model*", "tf_model*", "*.h5",
)


@dataclass(frozen=True)
class DownloadPreflight:
    """Cached and missing checkpoint artifacts plus local capacity information."""
    model: str
    cached_checkpoint: bool
    cached_bytes: int
    missing_download_bytes: int
    cached_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    cache_location: str
    available_disk_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable preflight result as a serializable mapping."""
        return asdict(self)


def _disk_space(path: Path) -> int:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free


def inspect_download(
    model: str,
    *,
    cache_dir: str | Path | None = None,
    dry_runner: Callable[..., list[Any]] = snapshot_download,
) -> DownloadPreflight:
    """Inspect a local path or perform a Hugging Face Hub dry run.

    Args:
        model: Local checkpoint path or Hugging Face model identifier.
        cache_dir: Optional Hub cache directory.
        dry_runner: Injectable Hub dry-run collaborator used by tests.

    Returns:
        File, byte, cache-location, and disk-capacity details.

    Raises:
        OSError: If local files or disk information cannot be inspected.
        huggingface_hub.errors.HfHubHTTPError: If Hub metadata is unavailable.

    Side Effects:
        Reads local metadata and may make a Hub metadata request, but does not
        download checkpoint content.
    """
    local = Path(model).expanduser()
    if local.exists():
        files = tuple(str(path.relative_to(local)) for path in local.rglob("*") if path.is_file())
        size = sum(path.stat().st_size for path in local.rglob("*") if path.is_file())
        return DownloadPreflight(model, True, size, 0, files, (), str(local.resolve()), _disk_space(local))

    infos = dry_runner(
        repo_id=model,
        cache_dir=str(cache_dir) if cache_dir else None,
        dry_run=True,
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    if any(info.filename.endswith(".safetensors") for info in infos):
        infos = [info for info in infos if not (info.filename.endswith(".bin") and "pytorch_model" in info.filename)]
    cached = tuple(info.filename for info in infos if info.is_cached)
    missing = tuple(info.filename for info in infos if info.will_download)
    cached_bytes = sum(info.file_size for info in infos if info.is_cached)
    missing_bytes = sum(info.file_size for info in infos if info.will_download)
    paths = [Path(info.local_path) for info in infos]
    if paths:
        common = Path(os.path.commonpath([str(path) for path in paths]))
        cache_location = common if common.is_dir() else common.parent
    else:
        cache_location = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "hub")
    return DownloadPreflight(
        model, not missing, cached_bytes, missing_bytes, cached, missing,
        str(cache_location), _disk_space(cache_location),
    )


def format_bytes(value: int) -> str:
    """Format a byte count as a two-decimal GiB string."""
    return f"{value / 1024**3:.2f} GiB"
