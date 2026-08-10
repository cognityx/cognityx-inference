from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_inference.adapters import AdapterRepository
from cognityx_inference.backends.legacy import VLLMBackend
from cognityx_inference.contracts import (
    AdapterPurpose,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    InferenceTimings,
    ModelCapabilities,
    TokenUsage,
)
from cognityx_inference.errors import AdapterError, ResearchRunError
from cognityx_inference.lifecycle import ModelManager
from cognityx_inference.research import (
    EvaluationSetRepository,
    InferencePairRequest,
    InferencePairRunner,
    ResearchContext,
    ResearchPublisher,
    _dataforge_checksum,
    _manifest_checksum,
)
from cognityx_inference.service import InferenceService
from cognityx_inference.tracking import SafeTracker
from llm_benchmark.vllm_engine import VLLMLLM

FIXTURE = Path(__file__).parent / "fixtures" / "training_adapter_manifest.json"


def _runtime(tmp_path: Path) -> StorageRuntime:
    return StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))


def _adapter(
    runtime: StorageRuntime,
    *,
    transform=lambda value: value,
    weights: bytes = b"fixture-weights",
) -> str:
    manifest = transform(json.loads(FIXTURE.read_text(encoding="utf-8")))
    store = runtime.for_role("model")
    root = "adapters/adapter-fixture/1"
    store.put_bytes(
        f"{root}/adapter_config.json",
        b'{"peft_type":"LORA"}\n',
        media_type="application/json",
    )
    store.put_bytes(
        f"{root}/adapter_model.safetensors",
        weights,
        media_type="application/octet-stream",
    )
    store.put_json(
        f"{root}/checksums.json",
        {
            "schema_version": "cognityx.training.adapter-checksums/v1",
            "files": manifest["files"],
            "bundle_checksum": manifest["bundle_checksum"],
        },
    )
    store.put_json(f"{root}/adapter-manifest.json", manifest)
    return store.uri(f"{root}/adapter-manifest.json")


def _evaluation(runtime: StorageRuntime, *, trainable: bool = False) -> str:
    records = [
        {
            "record_id": "eval-1",
            "question": "First?",
            "gold_reference": "One",
            "source_record_id": "source-1",
            "source_reference_id": "ref-1",
            "record_provenance": {"passage": "one"},
            "research_role": "exact_recall",
            "training_eligible": trainable,
            "metadata": {
                "research_role": "exact_recall",
                "training_eligible": trainable,
            },
        },
        {
            "record_id": "eval-2",
            "question": "Second?",
            "gold_reference": "Two",
            "source_record_id": "source-2",
            "source_reference_id": "ref-2",
            "source_evidence": {"passage": "two"},
            "research_role": "exact_recall",
            "training_eligible": False,
            "metadata": {
                "research_role": "exact_recall",
                "training_eligible": False,
            },
        },
    ]
    raw = b"".join(
        json.dumps(
            row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for row in records
    )
    store = runtime.for_role("dataset")
    root = "dataforge/evaluation-sets/eval-set/v1"
    records_key = f"{root}/records.jsonl"
    store.put_bytes(records_key, raw, media_type="application/x-ndjson")
    manifest = {
        "schema": "cognityx.dataforge.evaluation-set/v1",
        "evaluation_set_id": "eval-set",
        "evaluation_set_version": "v1",
        "evaluation_set_name": "Fixture",
        "research_role": "exact_recall",
        "training_eligible": False,
        "record_count": len(records),
        "records_uri": store.uri(records_key),
        "records_checksum": _dataforge_checksum(raw.decode("utf-8")),
        "source_refs": [{"dataset_id": "demo"}],
        "freeze_policy": "evaluation-only-v1",
        "created_at": "2026-08-10T00:00:00+00:00",
    }
    manifest["freeze_checksum"] = _dataforge_checksum(
        {
            "evaluation_set_id": "eval-set",
            "evaluation_set_version": "v1",
            "research_role": "exact_recall",
            "training_eligible": False,
            "records_checksum": manifest["records_checksum"],
            "record_count": len(records),
            "source_refs": manifest["source_refs"],
            "freeze_policy": "evaluation-only-v1",
        }
    )
    manifest["manifest_checksum"] = _manifest_checksum(manifest)
    store.put_json(f"{root}/manifest.json", manifest)
    return store.uri(f"{root}/manifest.json")


class FakeAdapterBackend:
    capabilities = ModelCapabilities(lifecycle=True, adapters=True, seed=True)
    selections: list[str | None] = []
    prompts: list[str] = []
    identity_suffix = ""

    def __init__(self, model: str, runtime: dict) -> None:
        self.model_name = model
        self.runtime = runtime

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, str]:
        return {
            "name": self.model_name,
            "requested_revision": "main",
            "resolved_revision": f"commit-1{self.identity_suffix}",
            "tokenizer_revision": "commit-1",
            "chat_template_checksum": "template-sha",
        }

    def infer(self, request, on_text=None, *, adapter=None):
        del on_text
        self.selections.append(adapter.adapter_id if adapter else None)
        self.prompts.append(request.prompt)
        return InferenceResponse(
            request_id=f"request-{len(self.selections)}",
            content=f"answer:{request.prompt}:{adapter.adapter_id if adapter else 'base'}",
            model=request.model,
            provider="local",
            backend="fake",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            timings=InferenceTimings(
                latency_seconds=0.2,
                time_to_first_token_seconds=0.05,
                tokens_per_second=10,
            ),
        )


@pytest.fixture(autouse=True)
def _clear_backend() -> None:
    FakeAdapterBackend.selections.clear()
    FakeAdapterBackend.prompts.clear()
    FakeAdapterBackend.identity_suffix = ""


def _service(runtime: StorageRuntime, tmp_path: Path) -> InferenceService:
    return InferenceService(
        ModelManager({"fake": FakeAdapterBackend}),
        adapter_repository=AdapterRepository(runtime, tmp_path / "cache"),
    )


def test_training_adapter_manifest_is_verified_and_checksum_cached(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    uri = _adapter(runtime)
    repository = AdapterRepository(runtime, tmp_path / "cache")

    first = repository.verify_and_materialize(uri)
    second = repository.verify_and_materialize(uri)

    assert first.adapter_id == "adapter-fixture"
    assert first.bundle_checksum == second.bundle_checksum
    assert first.local_path == second.local_path
    assert (
        first.local_path / "adapter_model.safetensors"
    ).read_bytes() == b"fixture-weights"


def test_wrong_schema_missing_object_and_corruption_are_rejected(tmp_path) -> None:
    runtime = _runtime(tmp_path / "schema")
    wrong = _adapter(
        runtime,
        transform=lambda value: {**value, "schema_version": "unknown/v1"},
    )
    with pytest.raises(AdapterError) as schema_error:
        AdapterRepository(runtime, tmp_path / "cache-schema").verify_and_materialize(
            wrong
        )
    assert schema_error.value.code == "adapter_manifest_invalid"

    missing_runtime = _runtime(tmp_path / "missing")
    missing_uri = missing_runtime.for_role("model").uri("absent/adapter-manifest.json")
    with pytest.raises(AdapterError) as missing_error:
        AdapterRepository(
            missing_runtime, tmp_path / "cache-missing"
        ).verify_and_materialize(missing_uri)
    assert missing_error.value.code == "adapter_artifact_missing"

    corrupt_runtime = _runtime(tmp_path / "corrupt")
    corrupt = _adapter(corrupt_runtime, weights=b"changed-weights")
    with pytest.raises(AdapterError) as checksum_error:
        AdapterRepository(
            corrupt_runtime, tmp_path / "cache-corrupt"
        ).verify_and_materialize(corrupt)
    assert checksum_error.value.code == "adapter_checksum_mismatch"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("name", "Other/Model", "adapter_base_model_mismatch"),
        ("resolved_revision", "other-commit", "adapter_base_model_mismatch"),
        ("tokenizer_revision", "other-tokenizer", "adapter_tokenizer_mismatch"),
        ("chat_template_checksum", "other-template", "adapter_chat_template_mismatch"),
    ],
)
def test_known_base_identity_mismatch_fails_before_generation(
    tmp_path, field, value, code
) -> None:
    runtime = _runtime(tmp_path)
    uri = _adapter(
        runtime,
        transform=lambda manifest: {
            **manifest,
            "base_model": {**manifest["base_model"], field: value},
        },
    )
    service = _service(runtime, tmp_path)

    with pytest.raises(AdapterError) as captured:
        service.infer(
            InferenceRequest(
                model="Qwen/Qwen3-8B",
                prompt="hello",
                backend="fake",
                adapter_manifest_uri=uri,
                adapter_purpose=AdapterPurpose.EVALUATION,
            )
        )

    assert captured.value.code == code
    assert FakeAdapterBackend.selections == []


def test_adapter_selection_and_deselection_share_one_resident_base(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    uri = _adapter(runtime)
    service = _service(runtime, tmp_path)

    adapted = service.infer(
        InferenceRequest(
            model="Qwen/Qwen3-8B",
            prompt="adapted",
            backend="fake",
            adapter_manifest_uri=uri,
            adapter_purpose="evaluation",
        )
    )
    base = service.infer(
        InferenceRequest(model="Qwen/Qwen3-8B", prompt="base", backend="fake")
    )

    assert FakeAdapterBackend.selections == ["adapter-fixture", None]
    assert len(service.models.statuses()) == 1
    assert adapted.extensions["adapter"]["adapter_id"] == "adapter-fixture"
    assert "adapter" not in base.extensions


def test_vllm_builds_one_lora_request_and_base_uses_none(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    verified = AdapterRepository(runtime, tmp_path / "cache").verify_and_materialize(
        _adapter(runtime)
    )
    captured: list[object] = []

    class FakeLoRARequest:
        def __init__(self, name, numeric_id, path, *, base_model_name):
            self.name = name
            self.numeric_id = numeric_id
            self.path = path
            self.base_model_name = base_model_name

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.lora", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.lora.request",
        SimpleNamespace(LoRARequest=FakeLoRARequest),
    )

    def generate(self, prompt, on_text=None, lora_request=None):
        del self, prompt, on_text
        captured.append(lora_request)
        return {"result": {"raw_output": "ok"}, "metrics": {}}

    monkeypatch.setattr(VLLMLLM, "generate", generate)
    backend = VLLMBackend("Qwen/Qwen3-8B", {"context_length": 128})
    backend.engine = object.__new__(VLLMLLM)
    request = InferenceRequest(model="Qwen/Qwen3-8B", prompt="hello")

    backend._generate(request, on_text=None, adapter=verified)
    backend._generate(request, on_text=None, adapter=None)

    selected = captured[0]
    assert isinstance(selected, FakeLoRARequest)
    assert selected.name == "adapter-fixture"
    assert selected.numeric_id == verified.lora_int_id
    assert selected.path == str(verified.local_path)
    assert selected.base_model_name == "Qwen/Qwen3-8B"
    assert captured[1] is None


def test_unsupported_backend_and_remote_provider_never_fall_back(tmp_path) -> None:
    class Unsupported(FakeAdapterBackend):
        capabilities = ModelCapabilities(lifecycle=True, adapters=False)

    runtime = _runtime(tmp_path)
    uri = _adapter(runtime)
    service = InferenceService(
        ModelManager({"unsupported": Unsupported}),
        adapter_repository=AdapterRepository(runtime, tmp_path / "cache"),
    )
    request = InferenceRequest(
        model="Qwen/Qwen3-8B",
        prompt="hello",
        backend="unsupported",
        adapter_manifest_uri=uri,
        adapter_purpose="evaluation",
    )

    with pytest.raises(AdapterError) as captured:
        service.infer(request)
    assert captured.value.code == "adapter_not_supported"
    assert Unsupported.selections == []

    with pytest.raises(AdapterError) as remote:
        service.infer(replace(request, provider="openai"))
    assert remote.value.code == "adapter_not_supported"


class CaptureTracker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(self, **payload) -> None:
        self.calls.append(payload)


def test_pair_preserves_order_lineage_manifests_and_shared_tracking(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    adapter_uri = _adapter(runtime)
    evaluation_uri = _evaluation(runtime)
    service = _service(runtime, tmp_path)
    tracker = CaptureTracker()
    publisher = ResearchPublisher(runtime.for_role("artifact"))
    runner = InferencePairRunner(
        service,
        EvaluationSetRepository(runtime),
        publisher,
        SafeTracker(tracker),
    )
    request = InferencePairRequest(
        evaluation_manifest_uri=evaluation_uri,
        model="Qwen/Qwen3-8B",
        adapter_manifest_uri=adapter_uri,
        backend="fake",
        context=ResearchContext(
            experiment_id="exp-demo",
            comparison_id="compare-1",
            arm_id="arm-1",
            seed=7,
            parent_run_id="mlflow-parent",
            training_variant_id="tvar-fixture",
            training_run_id="trun-fixture",
        ),
    )

    pair = runner.run(request)

    assert pair["pair_validation"] == "passed"
    assert FakeAdapterBackend.prompts == ["First?", "Second?", "First?", "Second?"]
    assert FakeAdapterBackend.selections == [
        None,
        None,
        "adapter-fixture",
        "adapter-fixture",
    ]
    assert (
        pair["base_run"]["runtime_fingerprint_sha256"]
        == pair["adapter_run"]["runtime_fingerprint_sha256"]
    )
    assert pair["base_run"]["manifest_checksum"]
    assert pair["adapter_run"]["predictions_checksum"]
    base_key = f"inference/research/runs/{pair['base_run']['inference_run_id']}/predictions.jsonl"
    with runtime.for_role("artifact").open(base_key) as source:
        rows = [json.loads(line) for line in source]
    assert [row["evaluation_record_id"] for row in rows] == ["eval-1", "eval-2"]
    assert rows[0]["source_reference_id"] == "ref-1"
    assert rows[0]["record_provenance"] == {"passage": "one"}
    assert rows[0]["reference_answer"] == "One"
    run_manifest = publisher.verify_manifest(
        f"inference/research/runs/{pair['base_run']['inference_run_id']}/manifest.json"
    )
    assert run_manifest["manifest_checksum"] == pair["base_run"]["manifest_checksum"]
    assert len(tracker.calls) == 3
    adapter_call = next(
        call for call in tracker.calls if call["tags"].get("cognityx.mode") == "adapter"
    )
    assert adapter_call["tags"]["cognityx.experiment_id"] == "exp-demo"
    assert adapter_call["tags"]["cognityx.training_run_id"] == "trun-fixture"
    assert adapter_call["parent_run_id"] == "mlflow-parent"
    assert set(adapter_call["references"]) >= {
        "run_manifest_uri",
        "predictions_uri",
        "adapter_manifest_uri",
    }
    assert "artifact" not in adapter_call and "files" not in adapter_call


def test_pair_rejects_runtime_mismatch_and_publishes_failure(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    adapter_uri = _adapter(runtime)
    evaluation_uri = _evaluation(runtime)
    service = _service(runtime, tmp_path)
    original = service.infer

    def changing(request, *args, **kwargs):
        response = original(request, *args, **kwargs)
        if request.adapter_manifest_uri:
            changed = dict(response.extensions["runtime_fingerprint"])
            changed["sha256"] = "different"
            response = replace(
                response,
                extensions={**response.extensions, "runtime_fingerprint": changed},
            )
        return response

    service.infer = changing
    runner = InferencePairRunner(
        service,
        EvaluationSetRepository(runtime),
        ResearchPublisher(runtime.for_role("artifact")),
    )

    with pytest.raises(ResearchRunError) as captured:
        runner.run(
            InferencePairRequest(
                evaluation_manifest_uri=evaluation_uri,
                model="Qwen/Qwen3-8B",
                adapter_manifest_uri=adapter_uri,
                backend="fake",
                context=ResearchContext(experiment_id="exp-demo"),
            )
        )
    assert captured.value.code == "pair_validation_failed"
    assert (
        captured.value.details["mismatches"][0]["field"] == "runtime_fingerprint.sha256"
    )
    assert captured.value.details["failure_uri"]


def test_trainable_evaluation_record_is_rejected(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    uri = _evaluation(runtime, trainable=True)

    with pytest.raises(ResearchRunError) as captured:
        EvaluationSetRepository(runtime).load(uri)
    assert captured.value.code == "evaluation_record_marked_trainable"
