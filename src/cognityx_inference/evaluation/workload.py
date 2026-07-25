"""Run boundary trials through the normalized inference service."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable, Mapping

from cognityx_inference.contracts import InferenceRequest
from cognityx_inference.evaluation.boundary import TrialOutcome, TrialSpec
from cognityx_inference.service import InferenceService
from cognityx_inference.telemetry import ResourceMonitor


class InferenceTrialRunner:
    """Translate a trial configuration into one measured inference call."""

    def __init__(
        self,
        service: InferenceService,
        base_request: InferenceRequest,
        *,
        monitor_factory: Callable[[], ResourceMonitor] | None = None,
    ) -> None:
        self.service = service
        self.base_request = base_request
        self.monitor_factory = monitor_factory

    def __call__(self, trial: TrialSpec) -> TrialOutcome:
        request = self._request(trial.configuration)
        monitor = self.monitor_factory() if self.monitor_factory else None
        if monitor is not None:
            monitor.start()
        started = time.monotonic()
        try:
            response = self.service.infer(request)
            runtime = time.monotonic() - started
            telemetry = monitor.stop() if monitor is not None else {}
            return TrialOutcome(
                trial_id=trial.trial_id,
                status="completed",
                configuration=trial.configuration,
                runtime_seconds=runtime,
                metrics={
                    "request_id": response.request_id,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "finish_reason": response.finish_reason.value,
                    "time_to_first_token_seconds": (
                        response.timings.time_to_first_token_seconds
                    ),
                    "tokens_per_second": response.timings.tokens_per_second,
                    "model_loading_seconds": response.timings.model_loading_seconds,
                    "prompt_processing_seconds": (
                        response.timings.prompt_processing_seconds
                    ),
                    "token_generation_seconds": (
                        response.timings.token_generation_seconds
                    ),
                    "telemetry": telemetry,
                },
            )
        except Exception as exc:
            runtime = time.monotonic() - started
            telemetry = monitor.stop() if monitor is not None else {}
            name = type(exc).__name__.lower()
            status = "out_of_memory" if "outofmemory" in name or "out of memory" in str(exc).lower() else "failed"
            return TrialOutcome(
                trial_id=trial.trial_id,
                status=status,
                configuration=trial.configuration,
                runtime_seconds=runtime,
                metrics={"telemetry": telemetry},
                error=f"{type(exc).__name__}: {exc}",
            )

    def _request(self, values: Mapping[str, Any]) -> InferenceRequest:
        runtime = dict(self.base_request.extensions.get("runtime", {}))
        for axis in (
            "quantization",
            "context_length",
            "kv_cache_precision",
            "gpu_memory_utilization",
            "tensor_parallelism",
            "attention_implementation",
        ):
            if axis in values:
                runtime[axis] = values[axis]
        extensions = dict(self.base_request.extensions)
        extensions["runtime"] = runtime
        replacements: dict[str, Any] = {"extensions": extensions}
        aliases = {
            "model": "model",
            "backend": "backend",
            "generation_length": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
        }
        for axis, field_name in aliases.items():
            if axis in values:
                replacements[field_name] = values[axis]
        return replace(self.base_request, **replacements)
