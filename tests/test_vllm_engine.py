from __future__ import annotations

import sys
import unittest
from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

from llm_benchmark.llm import GenerationSettings
from llm_benchmark.loading_decision import ModelLoadOptions
from llm_benchmark.vllm_engine import VLLMEngineError, VLLMLLM


class FakeLLM:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def get_tokenizer(self) -> object:
        return SimpleNamespace()


class FakeTokenizer:
    def __init__(self, result: object) -> None:
        self.result = result
        self.kwargs: dict[str, object] = {}

    def apply_chat_template(
        self, _messages: object, **kwargs: object
    ) -> object:
        self.kwargs = kwargs
        return self.result

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return list(range(len(text.split())))


class FakeSamplingParams:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)
        self.output_kind = None


class FakeCoreEngine:
    def __init__(self) -> None:
        metrics = SimpleNamespace(
            first_token_latency=0.1, first_token_ts=1.0, last_token_ts=2.0
        )
        self.outputs = [
            SimpleNamespace(
                request_id="0", finished=False, metrics=metrics,
                outputs=[SimpleNamespace(text="Hello ", token_ids=[1], finish_reason=None)],
            ),
            SimpleNamespace(
                request_id="0", finished=True, metrics=metrics,
                num_cached_tokens=1,
                outputs=[SimpleNamespace(text="world", token_ids=[2], finish_reason="stop")],
            ),
        ]

    def add_request(self, *_args: object, **_kwargs: object) -> None:
        return None

    def has_unfinished_requests(self) -> bool:
        return bool(self.outputs)

    def step(self) -> list[object]:
        return [self.outputs.pop(0)]


class VLLMConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeLLM.calls.clear()

    def _build(self, kv_cache_dtype: str) -> VLLMLLM:
        fake_module = SimpleNamespace(__version__="test", LLM=FakeLLM)
        with (
            patch.dict(sys.modules, {"vllm": fake_module}),
            patch("llm_benchmark.vllm_engine.AutoConfig.from_pretrained", return_value=object()),
            patch("llm_benchmark.vllm_engine.extract_architecture_metadata", return_value={"model_type": "qwen3"}),
        ):
            return VLLMLLM(
                "Qwen/Qwen3-32B",
                GenerationSettings(max_new_tokens=8192),
                "int4",
                ModelLoadOptions(strict_gpu_only=True),
                kv_cache_dtype=kv_cache_dtype,
                max_model_len=20000,
                enable_prefix_caching=True,
            )

    def test_normal_kv_cache_uses_inflight_bitsandbytes(self) -> None:
        engine = self._build("auto")
        kwargs = FakeLLM.calls[-1]
        self.assertEqual(kwargs["quantization"], "bitsandbytes")
        self.assertEqual(kwargs["load_format"], "bitsandbytes")
        self.assertEqual(kwargs["kv_cache_dtype"], "auto")
        self.assertIs(kwargs["enable_prefix_caching"], True)
        self.assertEqual(engine.model_runtime["engine"], "vllm")

    def test_fp8_kv_cache_is_forwarded(self) -> None:
        self._build("fp8")
        self.assertEqual(FakeLLM.calls[-1]["kv_cache_dtype"], "fp8")

    def test_rejects_context_capacity_above_model_limit(self) -> None:
        fake_module = SimpleNamespace(__version__="test", LLM=FakeLLM)
        with (
            patch.dict(sys.modules, {"vllm": fake_module}),
            patch(
                "llm_benchmark.vllm_engine.AutoConfig.from_pretrained",
                return_value=object(),
            ),
            patch(
                "llm_benchmark.vllm_engine.extract_architecture_metadata",
                return_value={"max_position_embeddings": 40960},
            ),
            self.assertRaisesRegex(
                VLLMEngineError, "exceeds the model's declared context limit"
            ),
        ):
            VLLMLLM(
                "Qwen/Qwen3-32B",
                GenerationSettings(max_new_tokens=8192),
                "int4",
                ModelLoadOptions(strict_gpu_only=True),
                kv_cache_dtype="fp8",
                max_model_len=50000,
            )

    def test_prompt_template_requests_plain_token_ids(self) -> None:
        engine = self._build("auto")
        tokenizer = FakeTokenizer([1, 2, 3])
        engine.tokenizer = tokenizer

        self.assertEqual(engine._prompt_token_ids("hello"), [1, 2, 3])
        self.assertIs(tokenizer.kwargs["return_dict"], False)

    def test_prompt_template_rejects_mapping_result(self) -> None:
        engine = self._build("auto")
        engine.tokenizer = FakeTokenizer({"input_ids": [1, 2, 3]})

        with self.assertRaisesRegex(VLLMEngineError, "flat list of token IDs"):
            engine._prompt_token_ids("hello")

    def test_generation_streams_delta_text_and_retains_complete_output(self) -> None:
        engine = self._build("auto")
        engine.tokenizer = FakeTokenizer([10, 11])
        engine.engine._preprocess_cmpl_one = lambda prompt: prompt
        engine.engine.request_counter = count()
        engine.engine.llm_engine = FakeCoreEngine()
        chunks: list[str] = []
        fake_vllm = SimpleNamespace(SamplingParams=FakeSamplingParams)
        fake_sampling = SimpleNamespace(
            RequestOutputKind=SimpleNamespace(DELTA="delta")
        )

        with (
            patch.dict(
                sys.modules,
                {"vllm": fake_vllm, "vllm.sampling_params": fake_sampling},
            ),
            patch(
                "llm_benchmark.vllm_engine.get_gpu_status",
                return_value={"allocated_gb": 0.0, "reserved_gb": 0.0},
            ),
        ):
            result = engine.generate("hello", on_text=chunks.append)

        self.assertEqual(chunks, ["Hello ", "world"])
        self.assertEqual(result["result"]["raw_output"], "Hello world")
        self.assertEqual(result["metrics"]["generated_tokens"], 2)
        self.assertEqual(result["metrics"]["vllm:prefix_cache_queries"], 2)
        self.assertEqual(result["metrics"]["vllm:prefix_cache_hits"], 1)

        engine.engine.request_counter = count()
        engine.engine.llm_engine = FakeCoreEngine()
        with (
            patch.dict(
                sys.modules,
                {"vllm": fake_vllm, "vllm.sampling_params": fake_sampling},
            ),
            patch(
                "llm_benchmark.vllm_engine.get_gpu_status",
                return_value={"allocated_gb": 0.0, "reserved_gb": 0.0},
            ),
        ):
            second = engine.generate("hello")
        self.assertEqual(second["metrics"]["vllm:prefix_cache_queries"], 4)
        self.assertEqual(second["metrics"]["vllm:prefix_cache_hits"], 2)
