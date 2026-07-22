from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from llm_benchmark.app import (
    BENCHMARK_NAME,
    BENCHMARK_PROMPT,
    benchmark_request,
    parse_args,
    parse_model_command,
    print_metrics,
    process_prompt,
)
from llm_benchmark.llm import (
    InterruptStoppingCriteria,
    LoadProfileError,
    build_model_load_kwargs,
    build_quality_indicators,
    detect_finish_reason,
    validate_load_profile,
)
from llm_benchmark.model_diagnostics import (
    analyze_device_map,
    collect_model_runtime,
    detect_attention_implementation,
    extract_architecture_metadata,
    extract_cache_configuration,
    extract_quantization_metadata,
)
from llm_benchmark.reporting import (
    benchmark_filename,
    get_gpu_status,
    sanitize_model_name,
    save_benchmark_result,
)


class GenerationCancellationTests(unittest.TestCase):
    def test_interrupt_stopping_criteria_tracks_event(self) -> None:
        cancelled = threading.Event()
        criteria = InterruptStoppingCriteria(cancelled)
        input_ids = torch.zeros((2, 4), dtype=torch.long)

        self.assertEqual(criteria(input_ids, torch.empty(0)).tolist(), [False, False])
        cancelled.set()
        self.assertEqual(criteria(input_ids, torch.empty(0)).tolist(), [True, True])


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
            "requested_load_profile": "int4",
            "effective_load_profile": "int4",
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
        self.assertEqual(path.parent.name, "int4")
        self.assertEqual(path.parent.parent.name, "test--model")
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

    def test_extracts_quantization_metadata(self) -> None:
        class QuantizationConfig:
            load_in_4bit = True
            load_in_8bit = False
            quant_method = "bitsandbytes"
            bnb_4bit_compute_dtype = torch.bfloat16
            bnb_4bit_quant_storage = torch.uint8

        class Config:
            quantization_config = QuantizationConfig()

        class Model:
            config = Config()
            is_loaded_in_4bit = True

        metadata = extract_quantization_metadata(Model(), "int4")

        self.assertEqual(metadata["effective_load_profile"], "int4")
        self.assertTrue(metadata["quantization_enabled"])
        self.assertEqual(metadata["quantization_bits"], 4)
        self.assertEqual(metadata["compute_dtype"], "torch.bfloat16")
        self.assertEqual(metadata["storage_dtype"], "torch.uint8")

    def test_non_quantized_bf16_metadata(self) -> None:
        class Model:
            config = object()

        metadata = extract_quantization_metadata(
            Model(),
            "bf16",
            torch.bfloat16,
        )

        self.assertEqual(metadata["effective_load_profile"], "bf16")
        self.assertFalse(metadata["quantization_enabled"])
        self.assertIsNone(metadata["quantization_bits"])
        self.assertEqual(metadata["compute_dtype"], "torch.bfloat16")


class CommandTests(unittest.TestCase):
    def test_engine_defaults_to_transformers(self) -> None:
        self.assertEqual(parse_args([]).engine, "transformers")

    def test_vllm_engine_and_fp8_cache_arguments(self) -> None:
        args = parse_args([
            "--engine", "vllm", "--kv-cache-dtype", "fp8",
            "--vllm-max-model-len", "30000",
        ])
        self.assertEqual(args.engine, "vllm")
        self.assertEqual(args.kv_cache_dtype, "fp8")
        self.assertEqual(args.vllm_max_model_len, 30000)

    def test_startup_argument_defaults(self) -> None:
        args = parse_args([])

        self.assertEqual(args.model, "Qwen/Qwen3-8B")
        self.assertEqual(args.profile, "bf16")

    def test_startup_arguments_accept_model_and_profile(self) -> None:
        args = parse_args(["--model", "Qwen/Qwen3-14B", "--profile", "int4"])

        self.assertEqual(args.model, "Qwen/Qwen3-14B")
        self.assertEqual(args.profile, "int4")

    def test_startup_accepts_config_path(self) -> None:
        args = parse_args(["--config", "custom.toml"])

        self.assertEqual(args.config, Path("custom.toml"))

    def test_startup_arguments_reject_unknown_profile(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--profile", "unknown"])

    def test_load_profile_validation(self) -> None:
        for profile in ("bf16", "fp16", "int8", "int4", "native", "auto"):
            self.assertEqual(validate_load_profile(profile), profile)
        with self.assertRaises(LoadProfileError):
            validate_load_profile("fp32")

    def test_model_command_defaults_to_bf16(self) -> None:
        self.assertEqual(
            parse_model_command("/model Qwen/Qwen3-8B"),
            ("Qwen/Qwen3-8B", "bf16"),
        )

    def test_model_command_accepts_profile(self) -> None:
        self.assertEqual(
            parse_model_command("/model Qwen/Qwen3-14B int8"),
            ("Qwen/Qwen3-14B", "int8"),
        )

    def test_benchmark_request_is_stable(self) -> None:
        self.assertEqual(
            benchmark_request(),
            (BENCHMARK_PROMPT, "benchmark", BENCHMARK_NAME),
        )
        self.assertEqual(
            BENCHMARK_PROMPT,
            "Can AI be applied to the travelling salesman problem?",
        )

    def test_missing_bitsandbytes_rejects_quantized_profiles(self) -> None:
        for profile in ("int8", "int4"):
            with self.assertRaisesRegex(LoadProfileError, "uv add bitsandbytes"):
                build_model_load_kwargs(profile, has_bitsandbytes=False)

    def test_bf16_does_not_require_bitsandbytes(self) -> None:
        kwargs, compute_dtype = build_model_load_kwargs(
            "bf16",
            has_bitsandbytes=False,
        )

        self.assertEqual(kwargs["torch_dtype"], torch.bfloat16)
        self.assertEqual(compute_dtype, torch.bfloat16)


class QualityIndicatorTests(unittest.TestCase):
    class Tokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[str]:
            return text.split()

    def test_quality_indicator_counts(self) -> None:
        quality = build_quality_indicators(
            self.Tokenizer(),
            "step one",
            "final answer",
            "step one </think> final answer",
            "eos",
            False,
        )

        self.assertTrue(quality["answer_complete"])
        self.assertTrue(quality["reasoning_detected"])
        self.assertEqual(quality["thinking_tokens"], 2)
        self.assertEqual(quality["answer_tokens"], 2)
        self.assertFalse(quality["empty_answer"])

    def test_length_truncation_is_incomplete(self) -> None:
        quality = build_quality_indicators(
            self.Tokenizer(), "", "partial", "partial", "length", True
        )

        self.assertFalse(quality["answer_complete"])
        self.assertTrue(quality["generation_truncated"])

    def test_eos_completion_is_complete(self) -> None:
        quality = build_quality_indicators(
            self.Tokenizer(), "", "done", "done", "eos", False
        )

        self.assertTrue(quality["answer_complete"])

    @patch("llm_benchmark.app.save_benchmark_result", return_value=Path("saved.json"))
    @patch("llm_benchmark.app.get_system_metadata", return_value={})
    def test_prompt_processing_marks_run_type(
        self,
        _metadata: object,
        _save: object,
    ) -> None:
        def generate(prompt: str, **kwargs: object) -> dict[str, object]:
            return {
                "model": "mock",
                "prompt": prompt,
                "run_type": kwargs["run_type"],
                "benchmark_name": kwargs["benchmark_name"],
                "finish_reason": "eos",
                "generation_truncated": False,
                "result": {"raw_output": "done", "thinking": "", "answer": "done"},
                "metrics": {
                    "time_to_first_token_seconds": 0.1,
                    "generation_seconds": 0.2,
                    "prompt_tokens": 1,
                    "generated_tokens": 1,
                    "total_tokens": 2,
                    "tokens_per_second": 5.0,
                    "peak_vram_gb": None,
                    "gpu_allocated_before_gb": None,
                    "gpu_reserved_before_gb": None,
                    "gpu_allocated_after_gb": None,
                    "gpu_reserved_after_gb": None,
                },
            }

        llm = Mock()
        llm.loading_diagnostics = {}
        llm.generate.side_effect = generate
        with contextlib.redirect_stdout(io.StringIO()):
            benchmark = process_prompt(
                llm,
                BENCHMARK_PROMPT,
                "benchmark",
                BENCHMARK_NAME,
            )
            interactive = process_prompt(llm, "hello")
            contextual = process_prompt(
                llm,
                " ".join(f"word{number}" for number in range(30)),
                context_metadata={
                    "final_prompt_tokens": 1,
                    "context_name": "sample",
                    "context_provider": "folder",
                    "actual_corpus_tokens": 5,
                    "chunks_used_count": 0,
                    "input_truncation_status": "not_truncated",
                },
            )

        self.assertEqual(benchmark["run_type"], "benchmark")
        self.assertEqual(benchmark["benchmark_name"], BENCHMARK_NAME)
        self.assertEqual(benchmark["schema_version"], 2)
        self.assertIn("api_equivalent_cost_estimate", benchmark)
        self.assertEqual(interactive["run_type"], "interactive")
        self.assertIsNone(interactive["benchmark_name"])
        self.assertTrue(contextual["prompt_truncated_for_reporting"])
        self.assertEqual(contextual["context"]["full_prompt_characters"], 199)
        self.assertEqual(len(contextual["context"]["full_prompt_sha256"]), 64)


class MetricDisplayTests(unittest.TestCase):
    def test_context_length_is_displayed(self) -> None:
        result = {
            "finish_reason": "eos",
            "generation_truncated": False,
            "metrics": {
                "time_to_first_token_seconds": 0.1,
                "generation_seconds": 0.2,
                "prompt_tokens": 10,
                "generated_tokens": 5,
                "total_tokens": 15,
                "total_context_length_tokens": 131072,
                "tokens_per_second": 25.0,
                "peak_vram_gb": None,
                "gpu_allocated_before_gb": None,
                "gpu_reserved_before_gb": None,
                "gpu_allocated_after_gb": None,
                "gpu_reserved_after_gb": None,
            },
        }

        with contextlib.redirect_stdout(io.StringIO()) as output:
            print_metrics(result)

        self.assertIn("Total context length supported: 131072 tokens", output.getvalue())


if __name__ == "__main__":
    unittest.main()
