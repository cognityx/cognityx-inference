from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cognityx_inference.errors import ResearchRunError
from cognityx_inference.tracking import (
    MLflowTracker,
    NoOpTracker,
    SafeTracker,
    TrackingConfiguration,
)


def _payload() -> dict:
    return {
        "name": "inference-adapter-run",
        "tags": {
            "cognityx.component": "inference",
            "cognityx.experiment_id": "exp-1",
        },
        "metrics": {"request_count": 2},
        "references": {
            "predictions_uri": "storage://local-main/artifacts/predictions.jsonl",
            "adapter_manifest_uri": "storage://local-main/models/adapter-manifest.json",
        },
        "parent_run_id": "parent-1",
    }


def test_noop_and_warn_policy_do_not_invalidate_storage_publication() -> None:
    assert SafeTracker(NoOpTracker()).record(**_payload()) is None

    class Broken:
        def record(self, **payload):
            del payload
            raise RuntimeError("offline")

    with pytest.warns(UserWarning, match="tracking failed"):
        diagnostic = SafeTracker(Broken(), failure_policy="warn").record(**_payload())
    assert diagnostic == {"status": "failed", "message": "offline"}


def test_strict_tracking_policy_fails_with_machine_code() -> None:
    class Broken:
        def record(self, **payload):
            del payload
            raise RuntimeError("offline")

    with pytest.raises(ResearchRunError) as captured:
        SafeTracker(Broken(), failure_policy="error").record(**_payload())
    assert captured.value.code == "tracking_failed"


def test_mlflow_receives_tags_metrics_and_storage_references_only(monkeypatch) -> None:
    calls: dict = {}

    class Run:
        info = SimpleNamespace(run_id="external-run")

    def set_tags(values):
        calls.setdefault("tags", {}).update(values)

    fake = SimpleNamespace(
        set_tracking_uri=lambda value: calls.setdefault("tracking_uri", value),
        set_experiment=lambda value: calls.setdefault("experiment", value),
        start_run=lambda **value: (calls.setdefault("run", value), Run())[1],
        log_metric=lambda name, value, **kwargs: calls.setdefault("metrics", []).append(
            (name, value, kwargs.get("step"))
        ),
        set_tag=lambda name, value: calls.setdefault("tags", {}).update({name: value}),
        set_tags=set_tags,
        end_run=lambda **value: calls.setdefault("end", value),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    tracker = MLflowTracker(
        TrackingConfiguration(
            backend="mlflow",
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="shared-experiment",
        )
    )

    tracker.record(**_payload())

    assert calls["experiment"] == "shared-experiment"
    assert calls["run"]["tags"]["mlflow.parentRunId"] == "parent-1"
    assert calls["run"]["tags"]["cognityx.experiment_id"] == "exp-1"
    assert calls["tags"]["cognityx.storage.predictions_uri.uri"].startswith(
        "storage://"
    )
    assert calls["metrics"] == [("request_count", 2.0, None)]
    assert calls["end"] == {"status": "FINISHED"}
    assert not hasattr(fake, "log_artifact")
