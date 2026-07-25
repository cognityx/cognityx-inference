from __future__ import annotations

import json
import time

from cognityx_inference.telemetry import read_windows_bridge


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
