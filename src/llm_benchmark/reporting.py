from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

BYTES_PER_GB = 1024**3
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/benchmarks")


def bytes_to_gb(value: int) -> float:
    return round(value / BYTES_PER_GB, 3)


def get_gpu_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "name": None,
            "allocated_gb": None,
            "reserved_gb": None,
            "total_memory_gb": None,
        }

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "allocated_gb": bytes_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved(device)),
        "total_memory_gb": bytes_to_gb(properties.total_memory),
    }


def get_system_metadata() -> dict[str, Any]:
    gpu = get_gpu_status()
    return {
        "gpu_name": gpu["name"],
        "gpu_total_memory_gb": gpu["total_memory_gb"],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
    }


def benchmark_filename(timestamp: datetime | None = None) -> str:
    timestamp = timestamp or datetime.now(timezone.utc)
    utc_timestamp = timestamp.astimezone(timezone.utc)
    return f"benchmark_{utc_timestamp:%Y%m%dT%H%M%S_%fZ}.json"


def save_benchmark_result(
    result: dict[str, Any],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = output_directory / benchmark_filename(timestamp)
    suffix = 1

    while path.exists():
        path = output_directory / (
            f"{Path(benchmark_filename(timestamp)).stem}_{suffix}.json"
        )
        suffix += 1

    with path.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    return path
