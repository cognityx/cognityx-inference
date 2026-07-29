from __future__ import annotations

import os

import pytest

from cognityx_inference.vllm_runtime import (
    VLLMRuntimeError,
    _runtime_source_roots,
    ensure_vllm_runtime,
)


def test_isolated_runtime_import_failure_is_actionable(monkeypatch) -> None:
    monkeypatch.setenv("COGNITYX_INFERENCE_VLLM_RUNTIME", "1")
    monkeypatch.setattr(
        "cognityx_inference.vllm_runtime.importlib.util.find_spec",
        lambda name: None,
    )
    monkeypatch.delitem(os.environ, "VLLM_FAKE_TEST", raising=False)
    with pytest.raises(VLLMRuntimeError, match="cannot import vLLM"):
        ensure_vllm_runtime(["serve"])


def test_runtime_path_includes_only_available_service_sources(tmp_path) -> None:
    project_root = tmp_path / "cognityx-inference"
    project_source = project_root / "src"
    resource_source = tmp_path / "cognityx-resource" / "src"
    project_source.mkdir(parents=True)
    resource_source.mkdir(parents=True)

    assert _runtime_source_roots(project_root) == (
        str(project_source),
        str(resource_source),
    )
