from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llm_benchmark.llm import detect_finish_reason
from llm_benchmark.reporting import (
    benchmark_filename,
    get_gpu_status,
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
        self.assertEqual(saved["finish_reason"], "eos")
        self.assertIn("raw_output", saved["result"])
        self.assertIn("generated_tokens", saved["metrics"])
        self.assertIn("transformers_version", saved["metadata"])

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


if __name__ == "__main__":
    unittest.main()
