from __future__ import annotations

import json
import time

from cognityx_inference.telemetry import ResourceMonitor, read_windows_bridge


def test_windows_bridge_marks_source_and_scope(tmp_path) -> None:
    path = tmp_path / "telemetry.json"
    path.write_text(
        json.dumps(
            {
                "cpu_percent": 25,
                "shared_memory_used_bytes": 1024,
            }
        ),
        encoding="utf-8",
    )

    sample = read_windows_bridge(path)

    assert sample["source"] == "windows_bridge"
    assert sample["scope"] == "windows_host"
    assert sample["shared_memory_used_bytes"] == 1024


def test_stale_windows_bridge_is_unavailable(tmp_path) -> None:
    path = tmp_path / "telemetry.json"
    path.write_text("{}", encoding="utf-8")

    assert read_windows_bridge(path, max_age_seconds=0.000001) is None


def test_resource_summary_reports_windows_shared_gpu_and_peaks() -> None:
    monitor = ResourceMonitor()
    monitor._samples = [
        {
            "phase": "inference", "process_cpu_percent": 10, "process_ram_bytes": 100,
            "host": {"source": "windows_bridge", "scope": "windows_host", "cpu_percent": 20, "total_bytes": 1000, "used_bytes": 400, "used_percent": 40},
            "gpu": {"device": "GPU", "dedicated_memory_used_bytes": 100, "dedicated_memory_total_bytes": 500, "shared_memory_used_bytes": 20, "shared_memory_total_bytes": 200, "utilization_percent": 30, "temperature_celsius": 60, "power_watts": 200, "power_limit_watts": 300},
        },
        {
            "phase": "inference", "process_cpu_percent": 30, "process_ram_bytes": 200,
            "host": {"source": "windows_bridge", "scope": "windows_host", "cpu_percent": 40, "total_bytes": 1000, "used_bytes": 600, "used_percent": 60},
            "gpu": {"device": "GPU", "dedicated_memory_used_bytes": 200, "dedicated_memory_total_bytes": 500, "shared_memory_used_bytes": 80, "shared_memory_total_bytes": 200, "utilization_percent": 80, "temperature_celsius": 70, "power_watts": 250, "power_limit_watts": 300},
        },
    ]

    summary = monitor._aggregate()

    assert summary["host_cpu_average_percent"] == 30
    assert summary["host_ram_peak_used_bytes"] == 600
    assert summary["gpu_usage"]["dedicated_memory_used_bytes_peak"] == 200
    assert summary["gpu_usage"]["shared_memory_used_bytes_average"] == 50
    assert summary["gpu_usage"]["temperature_celsius_peak"] == 70
    assert summary["gpu_usage"]["power_watts_peak"] == 250
