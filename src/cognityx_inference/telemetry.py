"""Local process, host, NVIDIA, and optional Windows-bridge telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Callable


def query_nvidia_gpu(executable: str = "nvidia-smi") -> dict[str, Any] | None:
    """Return one NVIDIA GPU sample, or ``None`` when unavailable."""
    fields = (
        "index,name,memory.used,memory.total,utilization.gpu,"
        "temperature.gpu,power.draw,power.limit"
    )
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        line = result.stdout.splitlines()[0]
        values = [item.strip() for item in line.split(",")]
        if len(values) != 8:
            return None
        numeric = [
            int(values[0]),
            values[1],
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
            float(values[6]),
            float(values[7]),
        ]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return {
        "device_index": numeric[0],
        "device": numeric[1],
        "dedicated_memory_used_bytes": round(numeric[2] * 1024 * 1024),
        "dedicated_memory_total_bytes": round(numeric[3] * 1024 * 1024),
        "utilization_percent": numeric[4],
        "temperature_celsius": numeric[5],
        "power_watts": numeric[6],
        "power_limit_watts": numeric[7],
        "shared_memory_used_bytes": None,
        "source": "nvidia-smi",
    }


def read_windows_bridge(
    path: str | Path,
    *,
    max_age_seconds: float = 5,
) -> dict[str, Any] | None:
    """Read a Windows-produced JSON sample without inventing missing fields."""
    bridge = Path(path)
    try:
        if time.time() - bridge.stat().st_mtime > max_age_seconds:
            return None
        value = json.loads(bridge.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value["source"] = "windows_bridge"
    value["scope"] = "windows_host"
    return value


def query_host() -> dict[str, Any]:
    """Sample the visible host/WSL scope using psutil."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "Local telemetry requires the telemetry extra: uv sync --extra telemetry"
        ) from exc
    memory = psutil.virtual_memory()
    return {
        "source": "psutil",
        "scope": "wsl_vm",
        "cpu_percent": psutil.cpu_percent(interval=None),
        "total_bytes": memory.total,
        "used_bytes": memory.used,
        "used_percent": memory.percent,
    }


@dataclass(slots=True)
class ResourceMonitor:
    """Sample resources in the background and aggregate compatible fields."""

    gpu_sampler: Callable[[], dict[str, Any] | None] = query_nvidia_gpu
    host_sampler: Callable[[], dict[str, Any]] = query_host
    windows_sampler: Callable[[], dict[str, Any] | None] | None = None
    interval_seconds: float = 0.25
    _samples: list[dict[str, Any]] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _process: Any = field(default=None, init=False)

    def start(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "Local telemetry requires the telemetry extra: "
                "uv sync --extra telemetry"
            ) from exc
        self._stop.clear()
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        return self._aggregate()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        windows = self.windows_sampler() if self.windows_sampler else None
        try:
            gpu = self.gpu_sampler()
        except Exception:
            gpu = None
        try:
            host = windows or self.host_sampler()
        except Exception:
            host = {
                "source": "unavailable",
                "scope": "unknown",
                "cpu_percent": None,
                "total_bytes": None,
                "used_bytes": None,
                "used_percent": None,
            }
        self._samples.append(
            {
                "sampled_at": time.time(),
                "process_cpu_percent": self._process.cpu_percent(interval=None),
                "process_ram_bytes": self._process.memory_info().rss,
                "host": host,
                "gpu": gpu,
            }
        )

    def _aggregate(self) -> dict[str, Any]:
        process_cpu = [item["process_cpu_percent"] for item in self._samples]
        process_ram = [item["process_ram_bytes"] for item in self._samples]
        hosts = [item["host"] for item in self._samples]
        gpus = [item["gpu"] for item in self._samples if item["gpu"]]
        return {
            "sample_count": len(self._samples),
            "process_cpu_average_percent": _mean(process_cpu),
            "process_cpu_peak_percent": max(process_cpu, default=None),
            "process_ram_average_bytes": _mean(process_ram, rounded=True),
            "process_ram_peak_bytes": max(process_ram, default=None),
            "host_source": hosts[-1].get("source") if hosts else None,
            "host_scope": hosts[-1].get("scope") if hosts else None,
            "host_cpu_average_percent": _mean_values(hosts, "cpu_percent"),
            "host_cpu_peak_percent": _max_values(hosts, "cpu_percent"),
            "host_ram_total_bytes": (
                hosts[-1].get("total_bytes") if hosts else None
            ),
            "host_ram_average_used_bytes": _mean_values(
                hosts, "used_bytes", rounded=True
            ),
            "host_ram_peak_used_bytes": _max_values(hosts, "used_bytes"),
            "host_ram_average_percent": _mean_values(hosts, "used_percent"),
            "host_ram_peak_percent": _max_values(hosts, "used_percent"),
            "windows_host": (
                hosts[-1]
                if hosts and hosts[-1].get("scope") == "windows_host"
                else None
            ),
            "gpu_usage": _aggregate_gpu(gpus),
        }


def _mean(values: list[Any], *, rounded: bool = False) -> Any:
    usable = [value for value in values if isinstance(value, (int, float))]
    if not usable:
        return None
    result = statistics.fmean(usable)
    return round(result) if rounded else result


def _mean_values(
    values: list[dict[str, Any]], key: str, *, rounded: bool = False
) -> Any:
    return _mean([value.get(key) for value in values], rounded=rounded)


def _max_values(values: list[dict[str, Any]], key: str) -> Any:
    usable = [
        value[key]
        for value in values
        if isinstance(value.get(key), (int, float))
    ]
    return max(usable, default=None)


def _aggregate_gpu(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not samples:
        return None
    fields = (
        "dedicated_memory_used_bytes",
        "shared_memory_used_bytes",
        "utilization_percent",
        "temperature_celsius",
        "power_watts",
    )
    result = {
        "device": samples[-1].get("device"),
        "device_index": samples[-1].get("device_index"),
        "source": samples[-1].get("source"),
        "sample_count": len(samples),
    }
    for field_name in fields:
        result[f"{field_name}_average"] = _mean_values(samples, field_name)
        result[f"{field_name}_peak"] = _max_values(samples, field_name)
    result["dedicated_memory_total_bytes"] = samples[-1].get(
        "dedicated_memory_total_bytes"
    )
    result["power_limit_watts"] = samples[-1].get("power_limit_watts")
    return result
