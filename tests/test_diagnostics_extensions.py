from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_benchmark.app import parse_cost_command, parse_cost_model_command
from llm_benchmark.comparison import (
    CompareOptions,
    load_comparison_rows,
    parse_compare_command,
    render_csv,
    render_markdown,
    save_comparison,
)
from llm_benchmark.cost_estimation import (
    CostConfig,
    configure_cost_model,
    estimate_api_equivalent_cost,
)
from llm_benchmark.llm import calculate_performance_metrics
from llm_benchmark.model_size import bytes_to_gib, calculate_model_size


class Parameter:
    def __init__(self, elements: int, element_bytes: int = 2) -> None:
        self.elements = elements
        self.element_bytes = element_bytes
        self.device = "cpu"

    def numel(self) -> int:
        return self.elements

    def element_size(self) -> int:
        return self.element_bytes


class Params4bit(Parameter):
    quant_state = object()


class Model:
    config = object()

    def __init__(self, parameters: list[Parameter]) -> None:
        self._parameters = parameters

    def parameters(self) -> list[Parameter]:
        return self._parameters

    @staticmethod
    def buffers() -> list[Parameter]:
        return []


def profile(name: str, quantized: bool = False, bits: int | None = None) -> dict[str, object]:
    return {
        "requested_load_profile": name,
        "effective_load_profile": name,
        "quantization_enabled": quantized,
        "quantization_bits": bits,
        "quantization_method": "bitsandbytes" if quantized else None,
        "compute_dtype": "torch.bfloat16",
        "storage_dtype": None,
    }


class ModelSizeTests(unittest.TestCase):
    def test_ordinary_count_deduplicates_tied_parameter(self) -> None:
        tied = Parameter(100)
        size = calculate_model_size(Model([tied, tied]), profile("bf16"))

        self.assertEqual(size["logical_parameters"], 100)
        self.assertTrue(size["parameter_count_reliable"])
        self.assertEqual(size["parameter_count_source"], "unique_parameter_tensors")

    def test_quantized_count_expands_packed_4bit_parameter(self) -> None:
        packed = Params4bit(50, 1)
        unquantized = Parameter(10, 2)
        size = calculate_model_size(
            Model([packed, unquantized, packed]),
            profile("int4", True, 4),
        )

        self.assertEqual(size["logical_parameters"], 110)
        self.assertEqual(
            size["parameter_count_source"],
            "quantization_aware_parameter_tensors",
        )
        self.assertTrue(size["parameter_count_reliable"])
        self.assertIsNone(size["actual_parameter_storage_bytes"])

    def test_quantized_fallback_is_marked_unreliable(self) -> None:
        size = calculate_model_size(
            Model([Parameter(25, 1)]),
            profile("int4", True, 4),
        )

        self.assertEqual(size["logical_parameters"], 25)
        self.assertFalse(size["parameter_count_reliable"])
        self.assertEqual(size["parameter_count_source"], "packed_parameter_fallback")

    def test_config_derived_quantized_fallback(self) -> None:
        class Config:
            hidden_size = 8
            num_hidden_layers = 2
            intermediate_size = 16
            num_attention_heads = 2
            num_key_value_heads = 1
            vocab_size = 32
            tie_word_embeddings = True

        model = Model([Parameter(10, 1)])
        model.config = Config()
        size = calculate_model_size(model, profile("int4", True, 4))

        self.assertGreater(size["logical_parameters"], 10)
        self.assertEqual(
            size["parameter_count_source"],
            "config_derived_causal_lm_approximation",
        )
        self.assertFalse(size["parameter_count_reliable"])

    def test_nominal_storage_for_each_profile(self) -> None:
        class Config:
            logical_parameter_count = 16

        model = Model([Parameter(16)])
        model.config = Config()
        expectations = {"bf16": 32, "fp16": 32, "int8": 16, "int4": 8}
        for name, expected_bytes in expectations.items():
            quantized = name in {"int8", "int4"}
            bits = {"int8": 8, "int4": 4}.get(name)
            size = calculate_model_size(model, profile(name, quantized, bits))
            self.assertEqual(size["estimated_raw_weight_storage_bytes"], expected_bytes)

    def test_binary_storage_conversion(self) -> None:
        self.assertEqual(bytes_to_gib(1024**3), 1.0)


class PerformanceTests(unittest.TestCase):
    def test_prompt_prefill_and_decode_calculations(self) -> None:
        performance = calculate_performance_metrics(
            prompt_tokenization_seconds=0.012,
            generation_seconds=5.0,
            end_to_end_seconds=5.1,
            prompt_tokens=20,
            generated_tokens=41,
            first_token_seconds=1.0,
            first_streamed_output_seconds=1.2,
        )

        self.assertEqual(performance["prefill_seconds"], 1.0)
        self.assertEqual(performance["prompt_tokens_per_second"], 20.0)
        self.assertEqual(performance["decode_seconds"], 4.0)
        self.assertEqual(performance["decode_tokens_per_second"], 10.0)
        self.assertEqual(performance["time_to_first_streamed_output_seconds"], 1.2)

    def test_zero_and_one_token_edges(self) -> None:
        zero = calculate_performance_metrics(
            prompt_tokenization_seconds=0,
            generation_seconds=0,
            end_to_end_seconds=0,
            prompt_tokens=0,
            generated_tokens=0,
            first_token_seconds=None,
            first_streamed_output_seconds=None,
        )
        one = calculate_performance_metrics(
            prompt_tokenization_seconds=0.1,
            generation_seconds=1.5,
            end_to_end_seconds=1.6,
            prompt_tokens=1,
            generated_tokens=1,
            first_token_seconds=1.0,
            first_streamed_output_seconds=1.1,
        )

        self.assertIsNone(zero["decode_seconds"])
        self.assertEqual(zero["decode_tokens_per_second"], 0.0)
        self.assertEqual(one["decode_token_count"], 0)
        self.assertEqual(one["decode_tokens_per_second"], 0.0)


class CostTests(unittest.TestCase):
    def test_cost_estimate_uses_supplied_prices(self) -> None:
        config = configure_cost_model("comparison", 2.5, 10.0)
        estimate = estimate_api_equivalent_cost(config, 20, 1000)

        self.assertEqual(estimate["estimated_input_cost"], 0.00005)
        self.assertEqual(estimate["estimated_output_cost"], 0.01)
        self.assertEqual(estimate["estimated_total_cost"], 0.01005)

    def test_disabled_cost_estimate(self) -> None:
        estimate = estimate_api_equivalent_cost(CostConfig(), 20, 1000)

        self.assertFalse(estimate["enabled"])
        self.assertEqual(estimate["basis"], "disabled")
        self.assertNotIn("estimated_total_cost", estimate)

    def test_invalid_prices_are_rejected(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                configure_cost_model("bad", value, 1.0)

    def test_cost_command_parsing(self) -> None:
        config = parse_cost_model_command('/cost-model "Example API" 2.5 10')

        self.assertEqual(config.pricing_label, "Example API")
        self.assertEqual(config.input_price_per_million_tokens, 2.5)
        self.assertEqual(parse_cost_command("/cost-off"), ("off", None))
        self.assertEqual(parse_cost_command("/cost-status"), ("status", None))


class ComparisonTests(unittest.TestCase):
    def _write(self, directory: Path, name: str, data: object) -> None:
        (directory / name).write_text(json.dumps(data), encoding="utf-8")

    def test_old_new_malformed_sort_filter_and_benchmark_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write(
                directory,
                "old.json",
                {
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "model": "Old/Model",
                    "prompt": "Can AI be applied to the travelling salesman problem?",
                    "metrics": {"prompt_tokens": 5, "generated_tokens": 6},
                },
            )
            self._write(
                directory,
                "new.json",
                {
                    "schema_version": 2,
                    "timestamp_utc": "2026-01-02T00:00:00+00:00",
                    "model": "New/Model",
                    "run_type": "benchmark",
                    "benchmark_name": "travelling_salesman_problem",
                    "performance": {"decode_tokens_per_second": 12.5},
                    "metrics": {"generated_tokens": 10},
                },
            )
            self._write(directory, "interactive.json", {"run_type": "interactive"})
            (directory / "bad.json").write_text("{bad", encoding="utf-8")

            rows, warnings = load_comparison_rows(directory)
            filtered, _ = load_comparison_rows(
                directory,
                CompareOptions(model="New", latest=1),
            )

        self.assertEqual([row["model"] for row in rows], ["Old/Model", "New/Model"])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["decode_tps"], 12.5)

    def test_compare_command_and_exports(self) -> None:
        options = parse_compare_command(
            "/compare --all --model Qwen --latest 3 --format markdown"
        )
        rows = [{key: "x" for key in (
            "timestamp", "model", "profile", "run_type", "benchmark_name",
            "prompt_tokens", "output_tokens", "ttft", "prefill", "decode_tps",
            "legacy_tps", "peak_vram", "cpu_offload", "truncated", "answer_complete",
        )}]

        self.assertTrue(options.include_all)
        self.assertEqual(options.latest, 3)
        self.assertIn("| Timestamp |", render_markdown(rows))
        self.assertIn("timestamp,model", render_csv(rows))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = save_comparison(rows, "markdown", Path(temporary_directory))
            self.assertEqual(path.suffix, ".md")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
