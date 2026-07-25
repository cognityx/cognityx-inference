from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cognityx_inference.capabilities import (
    CertifiedContextLimitExceeded,
    CertifiedInferenceProfile,
    ModelContextLimitExceeded,
    RuntimeCompatibility,
    certified_profile_from_benchmark_result,
    resolve_context,
)


def compatibility() -> RuntimeCompatibility:
    return RuntimeCompatibility(
        hardware_fingerprint="gpu-a",
        model="model-a",
        model_revision="revision-1",
        backend="vllm",
        backend_version="1.0",
        profile="int4",
        kv_cache_precision="auto",
    )


def profile(certified_limit: int = 8192) -> CertifiedInferenceProfile:
    return CertifiedInferenceProfile(
        profile_id="profile-1",
        created_at=datetime.now(timezone.utc).isoformat(),
        compatibility=compatibility(),
        model_context_limit=32768,
        maximum_certified_context_length=certified_limit,
    )


def test_context_is_minimum_of_model_and_certified_limits() -> None:
    resolution = resolve_context(32768, profile(8192), None)

    assert resolution.model_limit == 32768
    assert resolution.certified_limit == 8192
    assert resolution.effective_limit == 8192


def test_model_and_hardware_context_failures_are_distinct() -> None:
    with pytest.raises(ModelContextLimitExceeded):
        resolve_context(4096, profile(8192), 5000)
    with pytest.raises(CertifiedContextLimitExceeded):
        resolve_context(32768, profile(8192), 9000)


def test_legacy_benchmark_certifies_only_exercised_tokens() -> None:
    result = {
        "run_type": "benchmark",
        "benchmark_name": "context",
        "model": "model-a",
        "model_revision": "revision-1",
        "engine": "vllm",
        "effective_load_profile": "int4",
        "kv_cache_precision": "auto",
        "finish_reason": "stop",
        "model_runtime": {
            "architecture": {"max_position_embeddings": 32768}
        },
        "metrics": {
            "prompt_tokens": 4000,
            "generated_tokens": 96,
            "total_context_length_tokens": 32768,
            "tokens_per_second": 20,
        },
        "metadata": {
            "gpu_name": "GPU",
            "gpu_total_memory_gb": 32,
            "cuda_version": "13",
            "vllm_version": "1.0",
        },
    }

    certified = certified_profile_from_benchmark_result(result)

    assert certified.maximum_certified_context_length == 4096
    assert certified.model_context_limit == 32768
    assert (
        certified.metadata["configured_context_capacity"] == 32768
    )
