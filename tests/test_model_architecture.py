from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from llm_benchmark.app import build_status
from llm_benchmark.comparison import normalize_benchmark, parse_compare_command, render_table
from llm_benchmark.llm import GenerationSettings
from llm_benchmark.model_size import calculate_model_size
from llm_benchmark.model_architecture import (
    classify_model_family,
    extract_moe_config,
)


class DenseConfig:
    model_type = "qwen3"
    architectures = ["Qwen3ForCausalLM"]
    hidden_size = 5120
    num_hidden_layers = 40
    intermediate_size = 17408
    num_attention_heads = 40
    num_key_value_heads = 8
    vocab_size = 151936
    tie_word_embeddings = False


class QwenMoeConfig:
    model_type = "qwen3_moe"
    architectures = ["Qwen3MoeForCausalLM"]
    hidden_size = 2048
    num_hidden_layers = 48
    intermediate_size = 6144
    moe_intermediate_size = 768
    shared_expert_intermediate_size = 768
    num_attention_heads = 32
    num_key_value_heads = 4
    vocab_size = 151936
    num_experts = 128
    num_experts_per_tok = 8
    num_shared_experts = 1
    decoder_sparse_step = 1
    router_aux_loss_coef = 0.001
    tie_word_embeddings = False


class MixtralStyleConfig:
    model_type = "mixtral"
    architectures = ["MixtralForCausalLM"]
    hidden_size = 4096
    num_hidden_layers = 32
    intermediate_size = 14336
    expert_intermediate_size = 14336
    num_attention_heads = 32
    num_key_value_heads = 8
    vocab_size = 32000
    num_local_experts = 8
    num_experts_per_token = 2
    tie_word_embeddings = False


class ModelArchitectureTests(unittest.TestCase):
    def test_dense_model_active_count_equals_logical_count(self) -> None:
        family = classify_model_family(DenseConfig(), 14_768_307_200, True)

        self.assertEqual(family["family"], "Qwen3")
        self.assertEqual(family["architecture_type"], "dense")
        self.assertEqual(family["active_parameters_per_token"], 14_768_307_200)
        self.assertEqual(family["active_parameter_fraction"], 1.0)
        self.assertTrue(family["active_parameter_count_reliable"])

    def test_qwen_style_moe_detection_and_shared_metadata(self) -> None:
        family = classify_model_family(QwenMoeConfig(), 30_500_000_000, True)

        self.assertEqual(family["architecture_type"], "moe")
        self.assertEqual(family["num_experts"], 128)
        self.assertEqual(family["num_experts_per_token"], 8)
        self.assertEqual(family["shared_experts"], 1)
        self.assertIsNotNone(family["active_parameters_per_token"])
        self.assertFalse(family["active_parameter_count_reliable"])
        self.assertEqual(family["active_parameter_count_source"], "config_derived_estimate")

    def test_non_qwen_alternate_expert_fields(self) -> None:
        moe = extract_moe_config(MixtralStyleConfig())
        family = classify_model_family(MixtralStyleConfig(), 46_000_000_000, True)

        self.assertEqual(moe["num_experts"], 8)
        self.assertEqual(moe["num_experts_per_token"], 2)
        self.assertEqual(family["architecture_type"], "moe")
        self.assertIsNotNone(family["active_parameters_per_token"])

    def test_explicit_active_parameter_metadata_takes_precedence(self) -> None:
        class Config(QwenMoeConfig):
            active_parameter_count = 3_300_000_000

        family = classify_model_family(Config(), 30_500_000_000, True)

        self.assertEqual(family["active_parameters_per_token"], 3_300_000_000)
        self.assertEqual(family["active_parameter_count_source"], "config")
        self.assertTrue(family["active_parameter_count_reliable"])

    def test_model_native_active_parameter_metadata_is_used(self) -> None:
        class Model:
            active_parameters_per_token = 3_250_000_000

        family = classify_model_family(
            QwenMoeConfig(),
            30_500_000_000,
            True,
            Model(),
        )

        self.assertEqual(family["active_parameters_per_token"], 3_250_000_000)
        self.assertEqual(family["active_parameter_count_source"], "model_metadata")
        self.assertTrue(family["active_parameter_count_reliable"])

    def test_unknown_active_count_is_null(self) -> None:
        class Config:
            model_type = "custom_moe"
            architectures = ["CustomMoeForCausalLM"]
            num_experts = 16

        family = classify_model_family(Config(), 10_000, True)

        self.assertEqual(family["architecture_type"], "moe")
        self.assertIsNone(family["active_parameters_per_token"])
        self.assertIsNone(family["active_parameter_fraction"])
        self.assertEqual(family["active_parameter_count_source"], "unavailable")

    def test_zero_logical_count_does_not_divide(self) -> None:
        family = classify_model_family(QwenMoeConfig(), 0, False)

        self.assertIsNone(family["active_parameter_fraction"])

    def test_quantization_does_not_change_dense_active_count(self) -> None:
        logical = 14_768_307_200
        bf16 = classify_model_family(DenseConfig(), logical, True)
        int4 = classify_model_family(DenseConfig(), logical, True)

        self.assertEqual(
            bf16["active_parameters_per_token"],
            int4["active_parameters_per_token"],
        )

    def test_moe_logical_fallback_counts_all_experts(self) -> None:
        class Model:
            config = QwenMoeConfig()

            @staticmethod
            def parameters() -> list[object]:
                return []

            @staticmethod
            def buffers() -> list[object]:
                return []

        size = calculate_model_size(
            Model(),
            {
                "requested_load_profile": "int4",
                "effective_load_profile": "int4",
                "quantization_enabled": True,
                "quantization_bits": 4,
            },
        )
        family = classify_model_family(
            QwenMoeConfig(),
            size["logical_parameters"],
            size["parameter_count_reliable"],
        )

        self.assertGreater(size["logical_parameters"], 30_000_000_000)
        self.assertLess(
            family["active_parameters_per_token"],
            size["logical_parameters"],
        )


class StatusAndComparisonTests(unittest.TestCase):
    @patch(
        "llm_benchmark.app.get_gpu_status",
        return_value={
            "name": None,
            "allocated_gb": None,
            "reserved_gb": None,
            "total_memory_gb": None,
        },
    )
    def test_status_has_compact_and_detailed_architecture(self, _gpu: object) -> None:
        family = classify_model_family(QwenMoeConfig(), 30_500_000_000, True)

        class LLM:
            model_name = "Qwen/Qwen3-30B-A3B"
            requested_load_profile = "int4"
            effective_load_profile = "int4"
            load_profile_metadata = {}
            model_load_metrics = {}
            settings = GenerationSettings()
            model_runtime = {
                "model_family": family,
                "model_size": {
                    "logical_parameters": 30_500_000_000,
                    "logical_parameters_billions": 30.5,
                    "estimated_raw_weight_storage_gib": 14.203,
                    "parameter_count_source": "model_metadata",
                },
                "kv_cache": {},
            }

        compact = build_status(LLM(), None)
        detailed = build_status(LLM(), None, detailed=True)

        self.assertEqual(compact["model_runtime"]["architecture_type"], "moe")
        self.assertEqual(compact["model_runtime"]["experts"]["total"], 128)
        self.assertNotIn("model_family", compact["model_runtime"])
        self.assertIn("moe_config", detailed["model_runtime"]["model_family"])

    def test_compare_old_and_new_architecture_fields(self) -> None:
        old = normalize_benchmark(
            {"model": "old", "run_type": "benchmark"},
            Path("old.json"),
        )
        new = normalize_benchmark(
            {
                "model": "new",
                "run_type": "benchmark",
                "model_runtime": {
                    "model_family": {"architecture_type": "moe"},
                    "model_size": {
                        "logical_parameters": 30_500_000_000,
                        "active_parameters_per_token": 3_300_000_000,
                        "active_parameter_fraction": 0.1082,
                    },
                },
            },
            Path("new.json"),
        )
        options = parse_compare_command("/compare --architecture")
        table = render_table([old, new], options.include_architecture)

        self.assertIsNone(old["architecture_type"])
        self.assertEqual(new["architecture_type"], "moe")
        self.assertIn("30.50B", table)
        self.assertIn("3.30B", table)
        self.assertIn("10.8%", table)


if __name__ == "__main__":
    unittest.main()
