from __future__ import annotations

import os
import resource
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from llm_benchmark.model_size import nominal_bits_for_profile


@dataclass(frozen=True)
class ModelLoadOptions:
    loading_choice: str = "gpu_only"
    allow_download: bool = True
    cpu_offload_allowed: bool = False
    strict_gpu_only: bool = False
    disk_offload_allowed: bool = False
    max_memory: dict[Any, str] | None = None
    download_preflight: dict[str, Any] | None = None
    original_model_id: str | None = None
    quantized_checkpoint_model_id: str | None = None
    user_confirmations: dict[str, Any] = field(default_factory=dict)


def process_ram_gib() -> float:
    try:
        with open("/proc/self/statm", encoding="utf-8") as status:
            pages = int(status.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1024**3, 3)
    except (OSError, ValueError, IndexError):
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def system_ram_gib() -> float:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3


def gpu_memory() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    free, total = torch.cuda.mem_get_info()
    return total / 1024**3, free / 1024**3


def controlled_max_memory(gpu_headroom_fraction: float = 0.15) -> dict[Any, str]:
    total, free = gpu_memory()
    if free is None:
        return {"cpu": f"{max(1, int(system_ram_gib() * 0.8))}GiB"}
    usable = max(1, int(free * (1 - gpu_headroom_fraction)))
    return {0: f"{usable}GiB", "cpu": f"{max(1, int(system_ram_gib() * 0.8))}GiB"}


def estimate_capacity(parameter_count: int | None, profile: str) -> dict[str, Any]:
    bits = nominal_bits_for_profile(profile)
    raw = round(parameter_count * bits / 8) if parameter_count and bits else None
    total, free = gpu_memory()
    if raw is None or free is None:
        classification = "unknown"
    else:
        raw_gib = raw / 1024**3
        classification = "likely" if raw_gib <= free * 0.8 else "borderline" if raw_gib <= free else "impossible"
    return {
        "logical_parameters": parameter_count,
        "requested_profile": profile,
        "nominal_bits_per_weight": bits,
        "theoretical_raw_weight_bytes": raw,
        "gpu_total_gib": round(total, 2) if total is not None else None,
        "gpu_free_gib": round(free, 2) if free is not None else None,
        "gpu_only_estimate": classification,
        "overhead_warning": "Raw weights exclude quantization metadata, CUDA workspaces, activations, and KV cache.",
    }


def load_diagnostics(options: ModelLoadOptions, before_ram: float, after_ram: float, device_map: dict[str, Any]) -> dict[str, Any]:
    cpu = [name for name, device in device_map.items() if str(device).lower() == "cpu"]
    disk = [name for name, device in device_map.items() if str(device).lower() == "disk"]
    gpu = [name for name, device in device_map.items() if str(device).lower() not in {"cpu", "disk"}]
    offloaded = bool(cpu)
    low_bit = options.loading_choice == "gpu_only_low_bit"
    return {
        **asdict(options),
        "capacity_classification": "gpu_cpu_offloaded_inference" if offloaded else "gpu_only_low_bit_inference" if low_bit else "gpu_only_inference",
        "gpu_only_capacity_pass": not offloaded and not disk,
        "uses_cpu_offload": offloaded,
        "cpu_resident_modules": cpu,
        "gpu_resident_modules": gpu,
        "disk_offload_used": bool(disk),
        "cpu_memory_before_gib": before_ram,
        "cpu_memory_after_gib": after_ram,
        "cpu_memory_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
    }
