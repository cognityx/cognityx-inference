from __future__ import annotations

import os
from pathlib import Path

import pytest

from cognityx_inference.vllm_runtime import VLLMRuntimeError, ensure_vllm_runtime


def test_isolated_runtime_import_failure_is_actionable(monkeypatch) -> None:
    monkeypatch.setenv("COGNITYX_INFERENCE_VLLM_RUNTIME", "1")
    monkeypatch.setattr(
        "cognityx_inference.vllm_runtime.importlib.util.find_spec",
        lambda name: None,
    )
    monkeypatch.delitem(os.environ, "VLLM_FAKE_TEST", raising=False)
    with pytest.raises(VLLMRuntimeError, match="cannot import vLLM"):
        ensure_vllm_runtime(["serve"])


def test_runtime_path_includes_shared_service_sources() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert (project_root.parent / "cognityx-resource" / "src").is_dir()
    assert (project_root.parent / "cognityx-storage" / "src").is_dir()
    assert (project_root.parent / "cognityx-jobs" / "src").is_dir()
