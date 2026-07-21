from __future__ import annotations

import argparse
import csv
import io
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_PROMPT = "Can AI be applied to the travelling salesman problem?"
DEFAULT_BENCHMARK_DIRECTORY = Path("outputs/benchmarks")
DEFAULT_COMPARISON_DIRECTORY = Path("outputs/comparisons")

COLUMNS = (
    ("timestamp", "Timestamp"),
    ("model", "Model"),
    ("profile", "Profile"),
    ("run_type", "Run"),
    ("benchmark_name", "Benchmark"),
    ("prompt_tokens", "In tok"),
    ("output_tokens", "Out tok"),
    ("ttft", "TTFT"),
    ("prefill", "Prefill"),
    ("decode_tps", "Decode tok/s"),
    ("legacy_tps", "Legacy tok/s"),
    ("peak_vram", "Peak GiB"),
    ("cpu_offload", "CPU offload"),
    ("truncated", "Truncated"),
    ("answer_complete", "Complete"),
)
ARCHITECTURE_COLUMNS = (
    ("architecture_type", "Arch"),
    ("logical_parameters", "Logical"),
    ("active_parameters", "Active/tok"),
    ("active_fraction", "Active %"),
)


@dataclass
class CompareOptions:
    include_all: bool = False
    model: str | None = None
    benchmark: str | None = None
    latest: int | None = None
    output_format: str = "table"
    include_architecture: bool = False


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parse_compare_command(command: str) -> CompareOptions:
    parser = _Parser(add_help=False)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--benchmark")
    parser.add_argument("--latest", type=int)
    parser.add_argument("--format", choices=("table", "markdown", "csv"), default="table")
    parser.add_argument("--architecture", action="store_true")
    arguments = parser.parse_args(shlex.split(command)[1:])
    if arguments.latest is not None and arguments.latest <= 0:
        raise ValueError("--latest must be a positive integer")
    return CompareOptions(
        include_all=arguments.all,
        model=arguments.model,
        benchmark=arguments.benchmark,
        latest=arguments.latest,
        output_format=arguments.format,
        include_architecture=arguments.architecture,
    )


def _nested(data: dict[str, Any], section: str, field: str) -> Any:
    value = data.get(section)
    return value.get(field) if isinstance(value, dict) else None


def normalize_benchmark(data: dict[str, Any], path: Path) -> dict[str, Any]:
    performance = data.get("performance") if isinstance(data.get("performance"), dict) else {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    runtime = data.get("model_runtime") if isinstance(data.get("model_runtime"), dict) else {}
    model_family = runtime.get("model_family") if isinstance(runtime.get("model_family"), dict) else {}
    model_size = runtime.get("model_size") if isinstance(runtime.get("model_size"), dict) else {}
    quality = data.get("quality_indicators") if isinstance(data.get("quality_indicators"), dict) else {}
    run_type = data.get("run_type")
    if run_type is None and (
        data.get("benchmark_name") or data.get("prompt") == BENCHMARK_PROMPT
    ):
        run_type = "benchmark"
    run_type = run_type or "interactive"

    return {
        "timestamp": data.get("timestamp_utc") or path.stem,
        "model": data.get("model", "unknown"),
        "profile": data.get("effective_load_profile") or data.get("requested_load_profile") or "unknown",
        "run_type": run_type,
        "benchmark_name": data.get("benchmark_name") or "",
        "prompt_tokens": performance.get("prompt_tokens", metrics.get("prompt_tokens")),
        "output_tokens": performance.get("generated_tokens", metrics.get("generated_tokens")),
        "ttft": performance.get(
            "time_to_first_streamed_output_seconds",
            metrics.get("time_to_first_token_seconds"),
        ),
        "prefill": performance.get("prefill_seconds"),
        "decode_tps": performance.get("decode_tokens_per_second"),
        "legacy_tps": metrics.get("tokens_per_second"),
        "peak_vram": metrics.get("peak_vram_gb"),
        "cpu_offload": runtime.get("uses_cpu_offload"),
        "truncated": data.get("generation_truncated"),
        "answer_complete": quality.get("answer_complete"),
        "architecture_type": model_family.get("architecture_type"),
        "logical_parameters": model_size.get("logical_parameters"),
        "active_parameters": model_size.get("active_parameters_per_token"),
        "active_fraction": model_size.get("active_parameter_fraction"),
        "source_path": str(path),
    }


def load_comparison_rows(
    directory: Path = DEFAULT_BENCHMARK_DIRECTORY,
    options: CompareOptions | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    options = options or CompareOptions()
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not directory.exists():
        return rows, warnings

    for path in directory.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top-level JSON value is not an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipping malformed benchmark {path}: {exc}")
            continue

        row = normalize_benchmark(data, path)
        if not options.include_all and row["run_type"] != "benchmark":
            continue
        if options.model and options.model.casefold() not in str(row["model"]).casefold():
            continue
        if options.benchmark and options.benchmark.casefold() not in str(row["benchmark_name"]).casefold():
            continue
        rows.append(row)

    rows.sort(key=lambda row: str(row["timestamp"]))
    if options.latest is not None:
        rows = rows[-options.latest :]
    return rows, warnings


def _columns(include_architecture: bool) -> tuple[tuple[str, str], ...]:
    return COLUMNS + ARCHITECTURE_COLUMNS if include_architecture else COLUMNS


def _display(value: Any, width: int | None = None, key: str | None = None) -> str:
    if value is None:
        text = "-"
    elif isinstance(value, bool):
        text = "Yes" if value else "No"
    elif key in {"logical_parameters", "active_parameters"} and isinstance(
        value, (int, float)
    ):
        text = f"{value / 1_000_000_000:.2f}B"
    elif key == "active_fraction" and isinstance(value, (int, float)):
        text = f"{value * 100:.1f}%"
    elif isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    if width is not None and len(text) > width:
        return text[: max(1, width - 1)] + "…"
    return text


def render_table(
    rows: list[dict[str, Any]],
    include_architecture: bool = False,
) -> str:
    if not rows:
        return "No matching benchmark runs found."
    widths: dict[str, int] = {}
    caps = {"timestamp": 19, "model": 24, "benchmark_name": 22}
    columns = _columns(include_architecture)
    for key, heading in columns:
        widths[key] = min(
            caps.get(key, 14),
            max(len(heading), *(len(_display(row.get(key), key=key)) for row in rows)),
        )
    header = "  ".join(heading.ljust(widths[key]) for key, heading in columns)
    separator = "  ".join("-" * widths[key] for key, _ in columns)
    body = [
        "  ".join(
            _display(row.get(key), widths[key], key).ljust(widths[key])
            for key, _ in columns
        )
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def render_markdown(
    rows: list[dict[str, Any]],
    include_architecture: bool = False,
) -> str:
    columns = _columns(include_architecture)
    headings = [heading for _, heading in columns]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_display(row.get(key), key=key) for key, _ in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def render_csv(rows: list[dict[str, Any]], include_architecture: bool = False) -> str:
    output = io.StringIO()
    columns = _columns(include_architecture)
    writer = csv.DictWriter(output, fieldnames=[key for key, _ in columns])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key, _ in columns})
    return output.getvalue()


def save_comparison(
    rows: list[dict[str, Any]],
    output_format: str,
    directory: Path = DEFAULT_COMPARISON_DIRECTORY,
    include_architecture: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    extension = "md" if output_format == "markdown" else "csv"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"comparison_{timestamp}.{extension}"
    suffix = 1
    while path.exists():
        path = directory / f"comparison_{timestamp}_{suffix}.{extension}"
        suffix += 1
    content = (
        render_markdown(rows, include_architecture)
        if output_format == "markdown"
        else render_csv(rows, include_architecture)
    )
    with path.open("x", encoding="utf-8") as output_file:
        output_file.write(content)
    return path
