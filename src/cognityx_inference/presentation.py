"""Compact terminal views for inference lifecycle and discovery operations."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


def render(
    value: Any,
    *,
    kind: str,
    output_format: str = "auto",
) -> str:
    """Render a control-plane payload without changing its machine API shape."""
    if output_format == "json":
        return json.dumps(value, indent=2, ensure_ascii=False)
    if kind == "discovery_event":
        return _discovery_event(value)
    if kind == "discovery_list":
        return _discovery_list(value)
    if kind == "certified_profile_list":
        return _certified_profile_list(value)
    if kind == "certified_profile_show":
        return _certified_profile_show(value)
    if kind == "model_status":
        return _model_status(value)
    return _detail(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    values = [[_text(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in values:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join("-" * width for width in widths)
    header = "  ".join(name.ljust(widths[index]) for index, name in enumerate(headers))
    body = ["  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in values]
    return "\n".join([header, line, *body]) if body else "\n".join([header, line, "(none)"])


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _get(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _model_status(value: Any) -> str:
    items = value if isinstance(value, list) else [value]
    return _table(
        ["Model", "Backend", "State", "Context", "KV cache", "Profile", "Leases", "Load (s)"],
        [
            (
                _get(item, "identity", "model"),
                _get(item, "identity", "backend"),
                item.get("state"),
                _runtime(item, "context_length"),
                _runtime(item, "kv_cache_precision"),
                _runtime(item, "quantization"),
                item.get("active_leases"),
                item.get("load_seconds"),
            )
            for item in items
            if isinstance(item, Mapping)
        ],
    )


def _runtime(item: Mapping[str, Any], key: str) -> Any:
    runtime = _get(item, "identity", "runtime")
    if isinstance(runtime, Mapping):
        return runtime.get(key)
    return None


def _discovery_list(value: Any) -> str:
    items = value if isinstance(value, list) else [value]
    return _table(
        ["Job ID", "State", "Model", "Backend", "Profile", "Trials", "Certified profile", "Started"],
        [
            (
                item.get("job_id"), item.get("state"), item.get("model"), item.get("backend"),
                item.get("profile"), f"{len(item.get('trials') or [])}",
                item.get("certified_profile_id"), item.get("started_at"),
            )
            for item in items
            if isinstance(item, Mapping)
        ],
    )


def _certified_profile_list(value: Any) -> str:
    items = value if isinstance(value, list) else [value]
    return _table(
        ["Profile ID", "Model", "Backend", "Quant.", "KV cache", "Context", "Generation", "tok/s", "TTFT (s)", "Created"],
        [
            (
                item.get("profile_id"), _get(item, "compatibility", "model"),
                _get(item, "compatibility", "backend"), _get(item, "compatibility", "profile"),
                _get(item, "compatibility", "kv_cache_precision"), item.get("maximum_certified_context_length"),
                _get(item, "certified_configuration", "generation_length"),
                item.get("minimum_observed_tokens_per_second"), item.get("maximum_time_to_first_token_seconds"),
                item.get("created_at"),
            )
            for item in items
            if isinstance(item, Mapping)
        ],
    )


def _certified_profile_show(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _detail(value)
    compatibility = value.get("compatibility") or {}
    configuration = value.get("certified_configuration") or {}
    performance = value.get("performance") or {}
    resources = value.get("resource_summary") or {}
    gpu = resources.get("gpu_usage") or {}
    lines = [
        "Certified profile",
        "-----------------",
        f"Profile ID: {value.get('profile_id')}",
        f"Created: {value.get('created_at')}",
        f"Model/backend: {compatibility.get('model')} / {compatibility.get('backend')}",
        f"Quantization / KV cache: {compatibility.get('profile')} / {compatibility.get('kv_cache_precision')}",
        f"Certified context / generation: {value.get('maximum_certified_context_length')} / {configuration.get('generation_length')}",
        f"Evidence job: {value.get('evidence_job_id')}",
        "",
        "Performance",
        "-----------",
        f"Load: {_text(performance.get('model_loading_seconds'))} s   Prompt: {_text(performance.get('prompt_processing_seconds'))} s",
        f"TTFT: {_text(performance.get('time_to_first_token_seconds'))} s   Generation: {_text(performance.get('generation_seconds'))} s   Throughput: {_text(performance.get('tokens_per_second'))} tok/s",
        "",
        "Resources (overall average / peak)",
        "----------------------------------",
        f"Host CPU: {_text(resources.get('host_cpu_average_percent'))}% / {_text(resources.get('host_cpu_peak_percent'))}%",
        f"Host RAM: {_bytes(resources.get('host_ram_average_used_bytes'))} / {_bytes(resources.get('host_ram_peak_used_bytes'))}",
        f"GPU memory: {_bytes(gpu.get('dedicated_memory_used_bytes_average'))} / {_bytes(gpu.get('dedicated_memory_used_bytes_peak'))}",
        f"Shared GPU: {_bytes(gpu.get('shared_memory_used_bytes_average'))} / {_bytes(gpu.get('shared_memory_used_bytes_peak'))}",
        f"GPU utilization: {_text(gpu.get('utilization_percent_average'))}% / {_text(gpu.get('utilization_percent_peak'))}%",
        f"GPU temperature: {_text(gpu.get('temperature_celsius_average'))} C / {_text(gpu.get('temperature_celsius_peak'))} C",
        f"GPU power: {_text(gpu.get('power_watts_average'))} W / {_text(gpu.get('power_watts_peak'))} W (limit {_text(gpu.get('power_limit_watts'))} W)",
    ]
    phases = resources.get("phases") or {}
    if phases:
        lines.extend(["", "Measured phases", "---------------"])
        for name, phase in phases.items():
            phase_gpu = phase.get("gpu_usage") or {}
            lines.append(
                f"{name}: samples={_text(phase.get('sample_count'))}, "
                f"GPU peak={_bytes(phase_gpu.get('dedicated_memory_used_bytes_peak'))}, "
                f"power peak={_text(phase_gpu.get('power_watts_peak'))} W"
            )
    lines.extend(["", "Use --format json for the complete certified_trial evidence record."])
    return "\n".join(lines)


def _bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value / 1024**3:.2f} GiB"


def _discovery_event(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _detail(value)
    event = value.get("event")
    if event == "trial_started":
        configuration = value.get("configuration") or {}
        return f"[{value.get('trial_index')}/{value.get('total_trials')}] starting {configuration}"
    if event == "trial_completed":
        trial = value.get("trial") or {}
        metrics = trial.get("metrics") or {}
        configuration = trial.get("configuration") or {}
        return (
            f"[{value.get('completed_trials')}/{value.get('total_trials')}] "
            f"{trial.get('status')}: context={configuration.get('context_length')} "
            f"generation={configuration.get('generation_length')} kv={configuration.get('kv_cache_precision')} "
            f"time={_text(trial.get('runtime_seconds'))} s tok/s={_text(metrics.get('tokens_per_second'))} "
            f"TTFT={_text(metrics.get('time_to_first_token_seconds'))} s"
        )
    if event == "discovery_completed":
        return f"Discovery completed: certified profile {value.get('certified_profile_id')}"
    if event == "discovery_failed":
        return f"Discovery failed: {value.get('error')}"
    return _detail(value)


def _detail(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _text(value)
    return "\n".join(f"{key.replace('_', ' ').title()}: {_text(item)}" for key, item in value.items())
