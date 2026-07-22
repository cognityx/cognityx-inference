from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_benchmark.app import approve_download, choose_loading_mode, parse_args
from llm_benchmark.download_preflight import inspect_download
from llm_benchmark.loading_decision import ModelLoadOptions, estimate_capacity, load_diagnostics
from llm_benchmark.quantization import resolve_load_profile


class DownloadPreflightTests(unittest.TestCase):
    def test_dry_run_totals_cached_and_missing_files(self) -> None:
        entries = [
            SimpleNamespace(filename="config.json", file_size=10, is_cached=True, will_download=False, local_path="/tmp/cache/model/config.json"),
            SimpleNamespace(filename="model.safetensors", file_size=90, is_cached=False, will_download=True, local_path="/tmp/cache/model/model.safetensors"),
        ]
        result = inspect_download("org/model", dry_runner=lambda **_: entries)
        self.assertFalse(result.cached_checkpoint)
        self.assertEqual((result.cached_bytes, result.missing_download_bytes), (10, 90))

    def test_cached_checkpoint_needs_no_approval(self) -> None:
        preflight = SimpleNamespace(cached_checkpoint=True, cache_location="/tmp/model", cached_bytes=100)
        allowed, confirmation = approve_download(preflight, allow=False, deny=False, non_interactive=True)
        self.assertFalse(allowed)
        self.assertTrue(confirmation["checkpoint_already_cached"])

    def test_noninteractive_missing_download_requires_flag(self) -> None:
        preflight = SimpleNamespace(cached_checkpoint=False, model="org/model", cached_bytes=0, missing_download_bytes=100, available_disk_bytes=1000, cache_location="/tmp")
        with self.assertRaisesRegex(RuntimeError, "--allow-download"):
            approve_download(preflight, allow=False, deny=False, non_interactive=True)

    def test_cli_download_and_offload_flags(self) -> None:
        args = parse_args(["--allow-download", "--non-interactive", "--allow-cpu-offload"])
        self.assertTrue(args.allow_download and args.non_interactive and args.allow_cpu_offload)


class LoadingDecisionTests(unittest.TestCase):
    def test_double_quantization_choice(self) -> None:
        choice, profile, checkpoint = choose_loading_mode("int4", cpu_flag=False, non_interactive=False, input_fn=lambda _: "2")
        self.assertEqual((choice, profile, checkpoint), ("gpu_only", "int4-double", None))

    @patch("llm_benchmark.loading_decision.gpu_memory", return_value=(32.0, 30.0))
    def test_capacity_is_advisory(self, _gpu: object) -> None:
        result = estimate_capacity(72_000_000_000, "int4")
        self.assertEqual(result["gpu_only_estimate"], "impossible")
        self.assertIn("KV cache", result["overhead_warning"])

    def test_cpu_placement_classification(self) -> None:
        result = load_diagnostics(ModelLoadOptions(cpu_offload_allowed=True), 1.0, 2.0, {"layer.0": 0, "layer.1": "cpu"})
        self.assertEqual(result["capacity_classification"], "gpu_cpu_offloaded_inference")
        self.assertFalse(result["gpu_only_capacity_pass"])

    def test_int4_double_sets_nested_bnb(self) -> None:
        with patch("llm_benchmark.quantization.BitsAndBytesConfig") as config:
            resolve_load_profile("int4-double", SimpleNamespace(quantization_config=None), has_bitsandbytes=True)
        self.assertTrue(config.call_args.kwargs["bnb_4bit_use_double_quant"])
