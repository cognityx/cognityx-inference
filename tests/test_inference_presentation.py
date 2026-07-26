from cognityx_inference.presentation import render


def test_certified_profile_list_is_compact_table() -> None:
    output = render(
        [{
            "profile_id": "profile-1",
            "created_at": "2026-07-26T07:13:09Z",
            "maximum_certified_context_length": 40960,
            "minimum_observed_tokens_per_second": 21.67,
            "maximum_time_to_first_token_seconds": 0.196,
            "compatibility": {
                "model": "Qwen/Qwen3-8B", "backend": "vllm",
                "profile": "int4", "kv_cache_precision": "fp8",
            },
            "certified_configuration": {"generation_length": 1024},
        }],
        kind="certified_profile_list",
    )

    assert "Profile ID" in output
    assert "Qwen/Qwen3-8B" in output
    assert "fp8" in output
    assert "40960" in output


def test_certified_profile_show_summarizes_nested_resources() -> None:
    output = render(
        {
            "profile_id": "profile-1", "created_at": "now", "evidence_job_id": "job-1",
            "maximum_certified_context_length": 40960,
            "compatibility": {"model": "model", "backend": "vllm", "profile": "int4", "kv_cache_precision": "fp8"},
            "certified_configuration": {"generation_length": 1024},
            "performance": {"tokens_per_second": 20, "time_to_first_token_seconds": 0.2},
            "resource_summary": {
                "host_cpu_average_percent": 10, "host_cpu_peak_percent": 20,
                "gpu_usage": {"dedicated_memory_used_bytes_peak": 1024**3, "power_watts_peak": 250, "power_limit_watts": 575},
                "phases": {"model_loading": {"sample_count": 2, "gpu_usage": {"dedicated_memory_used_bytes_peak": 1024**3, "power_watts_peak": 250}}},
            },
        },
        kind="certified_profile_show",
    )

    assert "Certified profile" in output
    assert "Resources (overall average / peak)" in output
    assert "model_loading" in output
    assert "--format json" in output


def test_discovery_events_are_single_readable_lines() -> None:
    output = render(
        {
            "event": "trial_completed", "completed_trials": 2, "total_trials": 3,
            "trial": {
                "status": "completed", "runtime_seconds": 4.2,
                "configuration": {"context_length": 4096, "generation_length": 1024, "kv_cache_precision": "fp8"},
                "metrics": {"tokens_per_second": 20, "time_to_first_token_seconds": 0.2},
            },
        },
        kind="discovery_event",
    )

    assert output == "[2/3] completed: context=4096 generation=1024 kv=fp8 time=4.20 s tok/s=20 TTFT=0.20 s"
