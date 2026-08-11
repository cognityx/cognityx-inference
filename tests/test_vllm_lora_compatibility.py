from __future__ import annotations

import importlib.util
import inspect

import pytest


def test_installed_vllm_exposes_required_lora_api() -> None:
    """Optional runtime check; ordinary CI has no vLLM installation."""
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed in this test environment")

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.lora.request import LoRARequest

    assert "enable_lora" in inspect.signature(EngineArgs).parameters
    assert "lora_request" in inspect.signature(LLM.generate).parameters
    parameters = inspect.signature(LoRARequest).parameters
    assert {"lora_name", "lora_int_id", "lora_path"}.issubset(parameters)
