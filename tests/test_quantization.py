from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from llm_benchmark.model_loading import (
    attempted_load_configuration,
    classify_model_load_failure,
    wrap_expected_model_load_failure,
)
from llm_benchmark.quantization import (
    HardwareInfo,
    NativeQuantizationRequiredError,
    extract_checkpoint_quantization_method,
    checkpoint_has_quantization_config,
    detect_loaded_native_quantization_runtime,
    inspect_native_quantization_runtime,
    resolve_load_profile,
)


HARDWARE = HardwareInfo(
    accelerator_type="cuda",
    gpu_name="Mock GPU",
    compute_capability="9.0",
    cuda_version="13.0",
    pytorch_version="test",
    transformers_version="test",
    triton_version=None,
    triton_kernels_available=False,
    triton_compatible=True,
    triton_requirement=">=3.4.0",
    kernels_version="0.15.2",
    kernels_compatible=True,
    kernels_requirement=">=0.15.2,<0.16.0",
)


class Config:
    quantization_config = None


class QuantizationResolutionTests(unittest.TestCase):
    def test_unquantized_gemma_int4_uses_bitsandbytes(self) -> None:
        resolution = resolve_load_profile(
            "int4", Config(), HARDWARE, has_bitsandbytes=True
        )

        self.assertEqual(resolution.effective_profile, "int4")
        self.assertEqual(resolution.quantization_source, "runtime_bitsandbytes")
        self.assertTrue(resolution.quantization_config.load_in_4bit)
        self.assertEqual(resolution.quantization_config.bnb_4bit_quant_type, "nf4")

    def test_unquantized_qwen_int8_uses_bitsandbytes(self) -> None:
        resolution = resolve_load_profile(
            "int8", Config(), HARDWARE, has_bitsandbytes=True
        )

        self.assertTrue(resolution.quantization_config.load_in_8bit)
        self.assertEqual(resolution.runtime_quantization_method, "bitsandbytes")

    def test_mxfp4_native_passes_no_runtime_quantization_config(self) -> None:
        config = Config()
        config.quantization_config = {"quant_method": "mxfp4"}

        resolution = resolve_load_profile("native", config, HARDWARE)
        kwargs = resolution.load_kwargs()

        self.assertEqual(resolution.effective_profile, "native")
        self.assertEqual(resolution.checkpoint_quantization_method, "mxfp4")
        self.assertNotIn("quantization_config", kwargs)
        self.assertNotIn("torch_dtype", kwargs)

    def test_mxfp4_auto_resolves_native(self) -> None:
        config = Config()
        config.quantization_config = {"quant_method": "mxfp4"}

        resolution = resolve_load_profile("auto", config, HARDWARE)

        self.assertEqual(resolution.effective_profile, "native")
        self.assertFalse(resolution.profile_was_adapted)

    def test_mxfp4_int4_adapts_with_warning(self) -> None:
        config = Config()
        config.quantization_config = {"quant_method": "mxfp4"}

        resolution = resolve_load_profile("int4", config, HARDWARE)

        self.assertEqual(resolution.effective_profile, "native")
        self.assertTrue(resolution.profile_was_adapted)
        self.assertEqual(resolution.adaptation_reason, "checkpoint_already_quantized")
        self.assertIn("will not be applied", resolution.warning or "")

    def test_native_rejects_unquantized_checkpoint(self) -> None:
        with self.assertRaises(NativeQuantizationRequiredError):
            resolve_load_profile("native", Config(), HARDWARE)

    def test_auto_unquantized_defaults_to_bf16(self) -> None:
        resolution = resolve_load_profile("auto", Config(), HARDWARE)

        self.assertEqual(resolution.effective_profile, "bf16")
        self.assertEqual(resolution.compute_dtype, torch.bfloat16)
        self.assertEqual(resolution.quantization_source, "automatic_resolution")

    def test_method_extraction_supports_dict_and_object(self) -> None:
        dictionary = Config()
        dictionary.quantization_config = {"quantization_method": "FP8"}

        class ObjectQuantization:
            quant_method = "MXFP4"

        object_config = Config()
        object_config.quantization_config = ObjectQuantization()

        self.assertEqual(extract_checkpoint_quantization_method(dictionary), "fp8")
        self.assertEqual(extract_checkpoint_quantization_method(object_config), "mxfp4")

    def test_optional_method_can_be_unknown_without_losing_native_detection(self) -> None:
        config = Config()
        config.quantization_config = {}

        self.assertTrue(checkpoint_has_quantization_config(config))
        resolution = resolve_load_profile("auto", config, HARDWARE)
        self.assertTrue(resolution.checkpoint_quantized)
        self.assertIsNone(resolution.checkpoint_quantization_method)


class NativeRuntimeDiagnosticTests(unittest.TestCase):
    def test_enabled_mxfp4_runtime(self) -> None:
        runtime = inspect_native_quantization_runtime("mxfp4", HARDWARE)

        self.assertEqual(runtime["status"], "enabled")
        self.assertEqual(runtime["runtime_backend"], "mxfp4")
        self.assertIsNone(runtime["runtime_fallback"])

        config = Config()
        config.quantization_config = {"quant_method": "mxfp4"}
        diagnostic = resolve_load_profile("native", config, HARDWARE).diagnostic()
        self.assertEqual(
            diagnostic["native_quantization_runtime_preflight"]["status"],
            "enabled",
        )

    def test_missing_kernels_reports_bf16_fallback(self) -> None:
        hardware = replace(
            HARDWARE, kernels_version=None, kernels_compatible=False
        )

        runtime = inspect_native_quantization_runtime("mxfp4", hardware)

        self.assertEqual(runtime["status"], "unavailable")
        self.assertIn("Missing Python package: kernels", runtime["reason"])
        self.assertEqual(runtime["runtime_fallback"], "bf16")
        self.assertEqual(runtime["expected_performance_impact"], "high")

    def test_incompatible_kernels_version_is_not_called_missing(self) -> None:
        hardware = replace(
            HARDWARE, kernels_version="0.16.0", kernels_compatible=False
        )

        runtime = inspect_native_quantization_runtime("mxfp4", hardware)

        self.assertIn("Incompatible kernels version", runtime["reason"])
        self.assertNotIn("Missing Python package", runtime["reason"])

    def test_post_load_dequantize_state_is_authoritative(self) -> None:
        class QuantizationConfig:
            dequantize = True

        class Quantizer:
            quantization_config = QuantizationConfig()

        class Model:
            hf_quantizer = Quantizer()

        runtime = detect_loaded_native_quantization_runtime(
            Model(),
            "mxfp4",
            inspect_native_quantization_runtime("mxfp4", HARDWARE),
        )

        self.assertEqual(runtime["status"], "unavailable")
        self.assertEqual(runtime["runtime_fallback"], "bf16")
        self.assertEqual(
            runtime["detection_source"], "transformers_quantizer_post_load"
        )


class QuantizationFailureClassificationTests(unittest.TestCase):
    def test_conflicting_config(self) -> None:
        error = ValueError(
            "quantization_config conflicts because the checkpoint is already quantized"
        )
        self.assertEqual(
            classify_model_load_failure(error), "conflicting_quantization_config"
        )
        wrapped = wrap_expected_model_load_failure(
            error,
            "test/model",
            "int4",
            attempted_load_configuration("int4"),
            loading_stage="model_initialization",
        )
        self.assertIsNotNone(wrapped)
        self.assertEqual(wrapped.failure_category, "quantization")
        self.assertEqual(wrapped.failure_code, "conflicting_quantization_config")

    def test_missing_triton_support(self) -> None:
        error = ImportError("No module named triton; triton is required for MXFP4")
        self.assertEqual(
            classify_model_load_failure(error),
            "native_quantization_dependency_missing",
        )

    def test_unsupported_gpu_capability(self) -> None:
        error = RuntimeError(
            "MXFP4 quantization backend: GPU compute capability is not supported"
        )
        self.assertEqual(
            classify_model_load_failure(error),
            "native_quantization_hardware_unsupported",
        )

    def test_unsupported_native_method(self) -> None:
        self.assertEqual(
            classify_model_load_failure(
                ValueError("Unknown quantization type experimental_format")
            ),
            "unsupported_native_quantization",
        )

    def test_download_failure_during_model_load_gets_download_stage(self) -> None:
        wrapped = wrap_expected_model_load_failure(
            OSError("Repository not found"),
            "missing/model",
            "auto",
            attempted_load_configuration("auto"),
            loading_stage="model_initialization",
        )

        self.assertIsNotNone(wrapped)
        self.assertEqual(wrapped.failure_category, "download")
        self.assertEqual(wrapped.loading_stage, "model_download")


if __name__ == "__main__":
    unittest.main()
