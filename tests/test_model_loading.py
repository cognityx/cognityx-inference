from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from transformers import Mistral3Config, MistralConfig, Qwen3Config

from llm_benchmark.llm import GenerationSettings, LocalLLM
from llm_benchmark.model_loading import (
    ModelPlacementError,
    attempted_load_configuration,
    classify_model_load_failure,
    format_model_load_failure,
    save_failed_load,
    wrap_expected_model_load_failure,
)
from llm_benchmark.model_selection import (
    UnsupportedModelArchitectureError,
    select_model_loader,
)


class ModelSelectionTests(unittest.TestCase):
    def test_qwen3_selects_causal_lm(self) -> None:
        selection = select_model_loader(Qwen3Config())
        self.assertEqual(selection.task, "causal-lm")
        self.assertEqual(selection.auto_model_class.__name__, "AutoModelForCausalLM")

    def test_mistral_selects_causal_lm(self) -> None:
        selection = select_model_loader(MistralConfig())
        self.assertEqual(selection.task, "causal-lm")

    def test_mistral3_selects_image_text(self) -> None:
        selection = select_model_loader(Mistral3Config())
        self.assertEqual(selection.task, "image-text-to-text")
        self.assertEqual(
            selection.auto_model_class.__name__, "AutoModelForImageTextToText"
        )

    @patch(
        "llm_benchmark.model_selection._mapping_supports",
        side_effect=[False, True],
    )
    def test_other_mapped_image_text_config(self, _supports: object) -> None:
        class FutureImageTextConfig:
            pass

        selection = select_model_loader(FutureImageTextConfig())
        self.assertEqual(selection.task, "image-text-to-text")
        self.assertEqual(selection.selection_source, "transformers_auto_mapping")

    @patch("llm_benchmark.model_selection._mapping_supports", return_value=False)
    def test_unknown_config_is_clear(self, _supports: object) -> None:
        class UnknownConfig:
            pass

        with self.assertRaisesRegex(
            UnsupportedModelArchitectureError, "UnknownConfig"
        ):
            select_model_loader(UnknownConfig())


class FailureClassificationTests(unittest.TestCase):
    def test_incorrect_auto_model_is_not_placement(self) -> None:
        error = ValueError(
            "Unrecognized configuration class Mistral3Config for AutoModelForCausalLM"
        )
        self.assertEqual(
            classify_model_load_failure(error), "incorrect_auto_model_class"
        )
    def test_cuda_out_of_memory(self) -> None:
        error = torch.cuda.OutOfMemoryError("CUDA out of memory")
        self.assertEqual(classify_model_load_failure(error), "cuda_out_of_memory")

    def test_bitsandbytes_cpu_offload_error(self) -> None:
        error = ValueError(
            "bitsandbytes modules were dispatched on the CPU; provide a custom device_map"
        )
        self.assertEqual(classify_model_load_failure(error), "cpu_offload_required")

    def test_disk_offload_error(self) -> None:
        error = ValueError("The whole model was dispatched to disk; use disk_offload")
        self.assertEqual(classify_model_load_failure(error), "disk_offload_required")

    def test_generic_placement_failure(self) -> None:
        error = RuntimeError("Accelerate device_map could not fit the model")
        self.assertEqual(classify_model_load_failure(error), "placement_failure")

    def test_download_failure(self) -> None:
        error = OSError("Repository not found: private/model")
        self.assertEqual(classify_model_load_failure(error), "download_failure")

    def test_unknown_exception_is_not_wrapped(self) -> None:
        error = RuntimeError("unexpected programming invariant")
        attempted = attempted_load_configuration("bf16")

        self.assertIsNone(classify_model_load_failure(error))
        self.assertIsNone(
            wrap_expected_model_load_failure(error, "test/model", "bf16", attempted)
        )


class FailureReportingTests(unittest.TestCase):
    def _error(self, category: str = "cpu_offload_required") -> ModelPlacementError:
        return ModelPlacementError(
            "Test/MoE",
            "int4",
            RuntimeError("modules dispatched on the CPU"),
            category,
            {
                "requested_load_profile": "int4",
                "effective_load_profile": None,
                "quantization_enabled": True,
                "quantization_bits": 4,
                "quantization_method": "bitsandbytes",
                "compute_dtype": "torch.bfloat16",
                "storage_dtype": "torch.uint8",
            },
        )

    @patch(
        "llm_benchmark.model_loading.get_gpu_status",
        return_value={
            "name": "Mock GPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_memory_gb": 31.84,
        },
    )
    def test_cpu_failure_format_is_actionable(self, _gpu: object) -> None:
        message = format_model_load_failure(self._error())

        self.assertIn("Model loading failed", message)
        self.assertIn("Automatic placement determined", message)
        self.assertIn("bitsandbytes 4-bit", message)
        self.assertIn("Mock GPU", message)
        self.assertIn("No benchmark was executed", message)

    @patch(
        "llm_benchmark.model_loading.get_gpu_status",
        return_value={
            "name": "Mock GPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_memory_gb": 8.0,
        },
    )
    def test_cuda_oom_format(self, _gpu: object) -> None:
        message = format_model_load_failure(self._error("cuda_out_of_memory"))

        self.assertIn("CUDA ran out of memory", message)
        self.assertIn("Use a smaller model", message)

    @patch(
        "llm_benchmark.model_loading.get_gpu_status",
        return_value={
            "name": "Mock GPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_memory_gb": 31.84,
        },
    )
    def test_architecture_failure_says_placement_was_not_evaluated(
        self, _gpu: object
    ) -> None:
        error = ModelPlacementError(
            "test/model",
            "int4",
            ValueError("unsupported"),
            "incorrect_auto_model_class",
            attempted_load_configuration("int4"),
            {
                "config_class": "Mistral3Config",
                "auto_model_class": "AutoModelForCausalLM",
                "task": "image-text-to-text",
            },
            "model_loading",
        )

        message = format_model_load_failure(error)

        self.assertIn("Model architecture selection failed", message)
        self.assertIn("Mistral3Config", message)
        self.assertIn("before GPU memory placement was evaluated", message)
        self.assertNotIn("Automatic placement determined", message)

    @patch(
        "llm_benchmark.model_loading.get_system_metadata",
        return_value={
            "gpu_name": "Mock GPU",
            "gpu_total_memory_gb": 31.84,
            "cuda_version": "13.0",
            "pytorch_version": "test",
        },
    )
    def test_failed_load_json(self, _metadata: object) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = save_failed_load(self._error(), Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["model"], "Test/MoE")
        self.assertEqual(payload["profile"], "int4")
        self.assertEqual(payload["failure_category"], "cpu_offload_required")
        self.assertEqual(payload["gpu"]["total_memory_gb"], 31.84)


class LoaderBehaviorTests(unittest.TestCase):
    class Parameter:
        device = torch.device("cpu")

    class Model:
        generation_config = object()

        @staticmethod
        def eval() -> None:
            return None

        @staticmethod
        def parameters() -> object:
            return iter([LoaderBehaviorTests.Parameter()])

    @patch("llm_benchmark.llm.torch.cuda.is_available", return_value=False)
    @patch("llm_benchmark.llm.collect_model_runtime", return_value={})
    @patch("llm_benchmark.llm.extract_quantization_metadata")
    @patch("llm_benchmark.model_selection.AutoModelForCausalLM.from_pretrained")
    @patch("llm_benchmark.llm.AutoTokenizer.from_pretrained", return_value=Mock())
    @patch("llm_benchmark.llm.AutoConfig.from_pretrained", return_value=Qwen3Config())
    def test_successful_loading_is_unaffected(
        self,
        _config: object,
        _tokenizer: object,
        model_loader: Mock,
        quantization_metadata: Mock,
        _runtime: object,
        _cuda: object,
    ) -> None:
        model_loader.return_value = self.Model()
        quantization_metadata.return_value = {
            "requested_load_profile": "bf16",
            "effective_load_profile": "bf16",
            "quantization_enabled": False,
            "quantization_bits": None,
            "quantization_method": None,
            "compute_dtype": "torch.bfloat16",
            "storage_dtype": "torch.bfloat16",
        }

        with contextlib.redirect_stdout(io.StringIO()) as output:
            llm = LocalLLM("test/model", GenerationSettings(), "bf16")

        self.assertIsNotNone(llm.model)
        self.assertIn("Placement validated.", output.getvalue())
        self.assertEqual(llm.model_runtime["model_loader"]["task"], "causal-lm")

    @patch("llm_benchmark.llm.torch.cuda.is_available", return_value=False)
    @patch(
        "llm_benchmark.model_selection.AutoModelForCausalLM.from_pretrained",
        side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"),
    )
    @patch("llm_benchmark.llm.AutoTokenizer.from_pretrained", return_value=Mock())
    @patch("llm_benchmark.llm.AutoConfig.from_pretrained", return_value=Qwen3Config())
    def test_expected_loader_failure_is_structured(
        self,
        _config: object,
        _tokenizer: object,
        _model_loader: object,
        _cuda: object,
    ) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ModelPlacementError) as raised:
                LocalLLM("test/model", GenerationSettings(), "bf16")

        self.assertEqual(raised.exception.failure_category, "cuda_out_of_memory")

    @patch("llm_benchmark.llm.torch.cuda.is_available", return_value=False)
    @patch(
        "llm_benchmark.model_selection.AutoModelForCausalLM.from_pretrained",
        side_effect=RuntimeError("programming bug"),
    )
    @patch("llm_benchmark.llm.AutoTokenizer.from_pretrained", return_value=Mock())
    @patch("llm_benchmark.llm.AutoConfig.from_pretrained", return_value=Qwen3Config())
    def test_unknown_loader_failure_propagates_original_exception(
        self,
        _config: object,
        _tokenizer: object,
        _model_loader: object,
        _cuda: object,
    ) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "programming bug"):
                LocalLLM("test/model", GenerationSettings(), "bf16")

    @patch("llm_benchmark.llm.torch.cuda.is_available", return_value=False)
    @patch("llm_benchmark.llm.AutoConfig.from_pretrained")
    def test_unknown_architecture_fails_before_placement(
        self, config_loader: Mock, _cuda: object
    ) -> None:
        class UnknownConfig:
            pass

        config_loader.return_value = UnknownConfig()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(ModelPlacementError) as raised:
                LocalLLM("test/unknown", GenerationSettings(), "bf16")

        self.assertEqual(
            raised.exception.failure_category, "unsupported_model_architecture"
        )
        self.assertEqual(raised.exception.loading_stage, "architecture_selection")
        self.assertNotIn("Validating placement", output.getvalue())
        self.assertNotIn("Placement validation failed", output.getvalue())

    @patch("llm_benchmark.llm.torch.cuda.is_available", return_value=False)
    @patch("llm_benchmark.llm.collect_model_runtime", return_value={})
    @patch(
        "llm_benchmark.llm.extract_quantization_metadata",
        return_value={
            "requested_load_profile": "int4",
            "effective_load_profile": "int4",
            "quantization_enabled": True,
            "quantization_bits": 4,
            "quantization_method": "bitsandbytes",
            "compute_dtype": "torch.float16",
            "storage_dtype": "torch.uint8",
        },
    )
    @patch(
        "llm_benchmark.model_selection.AutoModelForImageTextToText.from_pretrained"
    )
    @patch("llm_benchmark.llm.AutoProcessor.from_pretrained")
    @patch("llm_benchmark.llm.AutoConfig.from_pretrained", return_value=Mistral3Config())
    @patch("llm_benchmark.llm.build_model_load_kwargs")
    def test_multimodal_loader_preserves_model_kwargs(
        self,
        build_kwargs: Mock,
        _config: object,
        processor_loader: Mock,
        model_loader: Mock,
        _quantization: object,
        _runtime: object,
        _cuda: object,
    ) -> None:
        quantization_config = object()
        load_kwargs = {"quantization_config": quantization_config, "device_map": "auto"}
        build_kwargs.return_value = (load_kwargs, torch.float16)
        processor_loader.return_value = Mock(tokenizer=Mock())
        model_loader.return_value = self.Model()

        LocalLLM("test/multimodal", GenerationSettings(), "int4")

        called_kwargs = model_loader.call_args.kwargs
        self.assertIs(called_kwargs["quantization_config"], quantization_config)
        self.assertEqual(called_kwargs["device_map"], "auto")
        self.assertIsInstance(called_kwargs["config"], Mistral3Config)

    def test_multimodal_text_only_input_preparation(self) -> None:
        llm = LocalLLM.__new__(LocalLLM)
        llm.settings = GenerationSettings()
        llm.tokenizer = Mock()
        llm.processor = Mock()
        llm.processor.apply_chat_template.return_value = "rendered chat"
        encoded = Mock()
        encoded.input_ids.shape = (1, 7)
        llm.processor.return_value = encoded

        inputs = llm._prepare_text_inputs("hello")

        messages = llm.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(
            messages,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        llm.processor.assert_called_once_with(
            text="rendered chat", return_tensors="pt"
        )
        self.assertEqual(inputs.input_ids.shape[1], 7)


if __name__ == "__main__":
    unittest.main()
