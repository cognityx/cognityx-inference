from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llm_benchmark.llm import detect_finish_reason
from llm_benchmark.model_diagnostics import (
    analyze_device_map,
    collect_model_runtime,
    detect_attention_implementation,
    extract_architecture_metadata,
    extract_cache_configuration,
)
from llm_benchmark.reporting import (
    benchmark_filename,
    get_gpu_status,
    sanitize_model_name,
    save_benchmark_result,
)


class FinishReasonTests(unittest.TestCase):
    def test_detects_eos(self) -> None:
        reason, truncated = detect_finish_reason([10, 2], {2}, 8)

        self.assertEqual(reason, "eos")
        self.assertFalse(truncated)

    def test_detects_token_limit_without_eos(self) -> None:
        reason, truncated = detect_finish_reason([10, 11, 12], {2}, 3)

        self.assertEqual(reason, "length")
        self.assertTrue(truncated)

    def test_short_output_without_eos_is_unknown(self) -> None:
        reason, truncated = detect_finish_reason([10], {2}, 3)

        self.assertEqual(reason, "unknown")
        self.assertFalse(truncated)


class ReportingTests(unittest.TestCase):
    def test_benchmark_filename_uses_utc_timestamp(self) -> None:
        timestamp = datetime(2026, 7, 21, 12, 34, 56, 123456, tzinfo=timezone.utc)

        self.assertEqual(
            benchmark_filename(timestamp),
            "benchmark_20260721T123456_123456Z.json",
        )

    def test_saved_json_preserves_structure(self) -> None:
        result = {
            "model": "test/model",
            "prompt": "hello",
            "settings": {"max_new_tokens": 10},
            "finish_reason": "eos",
            "generation_truncated": False,
            "result": {
                "raw_output": "answer",
                "thinking": "",
                "answer": "answer",
            },
            "metrics": {"generated_tokens": 1},
            "metadata": {
                "gpu_name": None,
                "gpu_total_memory_gb": None,
                "pytorch_version": "test",
                "cuda_version": None,
                "transformers_version": "test",
                "python_version": "test",
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = save_benchmark_result(result, Path(temporary_directory))
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved, result)
        self.assertEqual(path.parent.name, "test--model")
        self.assertEqual(saved["finish_reason"], "eos")
        self.assertIn("raw_output", saved["result"])
        self.assertIn("generated_tokens", saved["metrics"])
        self.assertIn("transformers_version", saved["metadata"])

    def test_sanitizes_model_name(self) -> None:
        self.assertEqual(sanitize_model_name("Qwen/Qwen3-14B"), "Qwen--Qwen3-14B")
        self.assertEqual(sanitize_model_name(" org/model name "), "org--model-name")

    @patch("llm_benchmark.reporting.torch.cuda.is_available", return_value=False)
    def test_gpu_status_without_cuda(self, _is_available: object) -> None:
        self.assertEqual(
            get_gpu_status(),
            {
                "name": None,
                "allocated_gb": None,
                "reserved_gb": None,
                "total_memory_gb": None,
            },
        )


class ModelDiagnosticTests(unittest.TestCase):
    def test_device_map_detects_cpu_and_disk_offload(self) -> None:
        diagnostics = analyze_device_map(
            {"embed": 0, "layers.0": "cuda:1", "layers.1": "cpu", "lm": "disk"}
        )

        self.assertEqual(diagnostics["devices_used"], ["cpu", "cuda:0", "cuda:1", "disk"])
        self.assertTrue(diagnostics["uses_cpu_offload"])
        self.assertTrue(diagnostics["uses_disk_offload"])
        self.assertTrue(diagnostics["uses_non_cuda_device"])

    def test_parameter_cpu_without_device_map_is_not_called_offload(self) -> None:
        diagnostics = analyze_device_map(None, ["cpu"])

        self.assertEqual(diagnostics["devices_used"], ["cpu"])
        self.assertFalse(diagnostics["uses_cpu_offload"])
        self.assertTrue(diagnostics["uses_non_cuda_device"])

    def test_extracts_generic_architecture_metadata(self) -> None:
        class Config:
            model_type = "mock"
            architectures = ["MockForCausalLM"]
            num_hidden_layers = 12
            hidden_size = 768
            intermediate_size = 3072
            num_attention_heads = 12
            num_key_value_heads = 4
            vocab_size = 32000
            max_position_embeddings = 8192
            rope_theta = 10000.0
            rope_scaling = {"type": "linear", "factor": 2.0}

        metadata = extract_architecture_metadata(Config(), 123456)

        self.assertEqual(metadata["architecture"], "MockForCausalLM")
        self.assertEqual(metadata["total_parameters"], 123456)
        self.assertEqual(metadata["num_key_value_heads"], 4)
        self.assertEqual(metadata["rope"]["scaling"]["factor"], 2.0)

    def test_missing_optional_attributes_and_attention_are_safe(self) -> None:
        class Empty:
            pass

        model = Empty()
        model.config = Empty()

        metadata = extract_architecture_metadata(model.config, 0)

        self.assertIsNone(metadata["num_key_value_heads"])
        self.assertIsNone(metadata["rope"]["scaling"])
        self.assertEqual(detect_attention_implementation(model), "unknown")
        self.assertEqual(
            extract_cache_configuration(model)["cache_implementation"],
            "unknown",
        )

    def test_attention_implementation_uses_explicit_config_value(self) -> None:
        class Config:
            _attn_implementation = "sdpa"

        class Model:
            config = Config()

        self.assertEqual(detect_attention_implementation(Model()), "sdpa")

    def test_collects_mixed_parameter_dtypes(self) -> None:
        class Config:
            model_type = "mock"

        class Model:
            config = Config()
            generation_config = object()
            hf_device_map = {"": "cpu"}

            @staticmethod
            def parameters() -> list[object]:
                class Parameter:
                    def __init__(self, dtype: str, size: int) -> None:
                        self.dtype = dtype
                        self.device = "cpu"
                        self.size = size

                    def numel(self) -> int:
                        return self.size

                return [Parameter("float16", 10), Parameter("float32", 5)]

        runtime = collect_model_runtime(Model())

        self.assertEqual(runtime["model_dtype"], "mixed")
        self.assertEqual(runtime["parameter_dtypes"], ["float16", "float32"])
        self.assertTrue(runtime["uses_mixed_dtypes"])
        self.assertEqual(runtime["architecture"]["total_parameters"], 15)


if __name__ == "__main__":
    unittest.main()
