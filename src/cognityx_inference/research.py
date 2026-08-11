"""Frozen evaluation execution and immutable base/adapter pair publications."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from cognityx_inference.contracts import (
    AdapterPurpose,
    InferenceRequest,
    InferenceResponse,
    LoadPolicy,
    ThinkingMode,
)
from cognityx_inference.errors import ResearchRunError
from cognityx_inference.tracking import NoOpTracker, SafeTracker

EVALUATION_SET_SCHEMA = "cognityx.dataforge.evaluation-set/v1"
INFERENCE_RUN_SCHEMA = "cognityx.inference.run/v1"
INFERENCE_PAIR_SCHEMA = "cognityx.inference.pair/v1"
EVALUATION_ROLES = frozenset(
    {"exact_recall", "paraphrase_evaluation", "heldout_knowledge_unit"}
)
_SAFE_CALLER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _dataforge_checksum(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _manifest_checksum(value: Mapping[str, Any]) -> str:
    return _dataforge_checksum(
        {key: item for key, item in value.items() if key != "manifest_checksum"}
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(dict(row)) + b"\n" for row in rows)


@dataclass(frozen=True, slots=True)
class EvaluationSet:
    manifest_uri: str
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]


class EvaluationSetRepository:
    """Consume DataForge's frozen public schema without importing DataForge."""

    def __init__(self, storage_runtime: Any) -> None:
        self.storage_runtime = storage_runtime

    def load(self, manifest_uri: str) -> EvaluationSet:
        try:
            manifest_object = self.storage_runtime.resolve_uri(
                manifest_uri, role_name="dataset"
            )
            with manifest_object.open() as source:
                manifest = json.load(source)
        except Exception as exc:
            raise ResearchRunError(
                "evaluation_manifest_invalid",
                f"Evaluation manifest cannot be read: {manifest_uri}",
            ) from exc
        if not isinstance(manifest, dict):
            raise ResearchRunError(
                "evaluation_manifest_invalid", "Evaluation manifest must be an object"
            )
        if manifest.get("schema") != EVALUATION_SET_SCHEMA:
            raise ResearchRunError(
                "evaluation_manifest_invalid",
                f"Unsupported evaluation schema: {manifest.get('schema')!r}",
            )
        if manifest.get("manifest_checksum") != _manifest_checksum(manifest):
            raise ResearchRunError(
                "evaluation_checksum_mismatch",
                "Evaluation manifest checksum does not match",
            )
        if manifest.get("training_eligible") is not False:
            raise ResearchRunError(
                "evaluation_record_marked_trainable",
                "Evaluation manifests must set training_eligible=false",
            )
        freeze_identity = {
            "evaluation_set_id": manifest.get("evaluation_set_id"),
            "evaluation_set_version": manifest.get("evaluation_set_version"),
            "research_role": manifest.get("research_role"),
            "training_eligible": False,
            "records_checksum": manifest.get("records_checksum"),
            "record_count": manifest.get("record_count"),
            "source_refs": manifest.get("source_refs", []),
            "freeze_policy": "evaluation-only-v1",
        }
        if manifest.get("freeze_policy") != "evaluation-only-v1" or manifest.get(
            "freeze_checksum"
        ) != _dataforge_checksum(freeze_identity):
            raise ResearchRunError(
                "evaluation_checksum_mismatch",
                "Evaluation freeze checksum does not match",
            )
        role = str(manifest.get("research_role", ""))
        if role not in EVALUATION_ROLES:
            raise ResearchRunError(
                "evaluation_manifest_invalid",
                f"Unsupported evaluation research_role: {role!r}",
            )
        try:
            records_object = self.storage_runtime.resolve_uri(
                str(manifest["records_uri"]), role_name="dataset"
            )
            with records_object.open() as source:
                raw = source.read()
            text = raw.decode("utf-8")
            records = tuple(
                json.loads(line) for line in text.splitlines() if line.strip()
            )
        except Exception as exc:
            raise ResearchRunError(
                "evaluation_artifact_missing",
                "Evaluation records cannot be read",
            ) from exc
        if _dataforge_checksum(text) != manifest.get("records_checksum"):
            raise ResearchRunError(
                "evaluation_checksum_mismatch",
                "Evaluation records checksum does not match",
            )
        if len(records) != manifest.get("record_count"):
            raise ResearchRunError(
                "evaluation_manifest_invalid", "Evaluation record_count does not match"
            )
        self._validate_records(records, role)
        return EvaluationSet(manifest_uri, manifest, records)

    @staticmethod
    def _validate_records(records: tuple[Any, ...], role: str) -> None:
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict) or not record.get("record_id"):
                raise ResearchRunError(
                    "evaluation_manifest_invalid",
                    "Every evaluation row requires a record_id",
                )
            record_id = str(record["record_id"])
            if record_id in seen:
                raise ResearchRunError(
                    "evaluation_manifest_invalid",
                    f"Duplicate evaluation record_id: {record_id}",
                )
            seen.add(record_id)
            eligible = record.get(
                "training_eligible",
                (record.get("metadata") or {}).get("training_eligible", False),
            )
            if eligible is not False:
                raise ResearchRunError(
                    "evaluation_record_marked_trainable",
                    f"Evaluation record is trainable: {record_id}",
                )
            record_role = record.get("research_role") or (
                record.get("metadata") or {}
            ).get("research_role")
            if record_role != role:
                raise ResearchRunError(
                    "evaluation_manifest_invalid",
                    f"Evaluation record role does not match: {record_id}",
                )


@dataclass(frozen=True, slots=True)
class ResearchContext:
    experiment_id: str
    comparison_id: str | None = None
    arm_id: str | None = None
    seed: int | None = None
    parent_run_id: str | None = None
    research_package_id: str | None = None
    training_variant_id: str | None = None
    training_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class InferencePairRequest:
    evaluation_manifest_uri: str
    model: str
    adapter_manifest_uri: str
    context: ResearchContext
    model_revision: str | None = None
    backend: str = "vllm"
    profile: str = "bf16"
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    top_k: int | None = None
    max_output_tokens: int = 512
    thinking: ThinkingMode = ThinkingMode.DISABLED
    stop: tuple[str, ...] = ()
    required_context_length: int | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    inference_pair_id: str | None = None

    def __post_init__(self) -> None:
        if (
            self.inference_pair_id
            and _SAFE_CALLER_ID.fullmatch(self.inference_pair_id) is None
        ):
            raise ValueError("inference_pair_id must be one safe Storage path segment")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferencePairRequest":
        context = dict(value.get("research_context") or {})
        if "experiment_id" not in context and value.get("experiment_id"):
            context["experiment_id"] = value["experiment_id"]
        return cls(
            evaluation_manifest_uri=str(value["evaluation_manifest_uri"]),
            model=str(value["model"]),
            adapter_manifest_uri=str(value["adapter_manifest_uri"]),
            context=ResearchContext(**context),
            model_revision=value.get("model_revision"),
            backend=str(value.get("backend", "vllm")),
            profile=str(value.get("profile", "bf16")),
            temperature=value.get("temperature", 0.0),
            top_p=value.get("top_p", 1.0),
            top_k=value.get("top_k"),
            max_output_tokens=int(value.get("max_output_tokens", 512)),
            thinking=ThinkingMode(value.get("thinking", "disabled")),
            stop=tuple(value.get("stop") or ()),
            required_context_length=value.get("required_context_length"),
            runtime=dict(value.get("runtime") or {}),
            inference_pair_id=value.get("inference_pair_id"),
        )


class ResearchPublisher:
    """Publish predictions first and immutable manifests last."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def publish_run(
        self, run: Mapping[str, Any], predictions: tuple[Mapping[str, Any], ...]
    ) -> dict[str, Any]:
        run_id = str(run["inference_run_id"])
        root = f"inference/research/runs/{run_id}"
        predictions_key = f"{root}/predictions.jsonl"
        manifest_key = f"{root}/manifest.json"
        content = _jsonl(predictions)
        self._put_bytes(predictions_key, content, media_type="application/x-ndjson")
        manifest = {
            **dict(run),
            "schema": INFERENCE_RUN_SCHEMA,
            "status": "completed",
            "predictions_uri": self.store.uri(predictions_key),
            "predictions_checksum": _sha256(content),
        }
        manifest["manifest_checksum"] = _manifest_checksum(manifest)
        self._put_json(manifest_key, manifest)
        return {**manifest, "manifest_uri": self.store.uri(manifest_key)}

    def publish_pair(self, pair: Mapping[str, Any]) -> dict[str, Any]:
        pair_id = str(pair["inference_pair_id"])
        key = f"inference/research/pairs/{pair_id}/manifest.json"
        manifest = {
            **dict(pair),
            "schema": INFERENCE_PAIR_SCHEMA,
            "status": "completed",
        }
        manifest["manifest_checksum"] = _manifest_checksum(manifest)
        self._put_json(key, manifest)
        return {**manifest, "manifest_uri": self.store.uri(key)}

    def publish_failure(self, pair_id: str, value: Mapping[str, Any]) -> str:
        key = f"inference/research/pairs/{pair_id}/failure.json"
        self._put_json(key, dict(value))
        return self.store.uri(key)

    def verify_manifest(self, manifest_key: str) -> dict[str, Any]:
        """Verify one role-local immutable manifest after publication."""
        with self.store.open(manifest_key) as source:
            manifest = json.load(source)
        if manifest.get("manifest_checksum") != _manifest_checksum(manifest):
            raise ResearchRunError(
                "inference_manifest_checksum_mismatch",
                "Inference manifest checksum does not match",
            )
        return manifest

    def load_pair(self, pair_id: str) -> dict[str, Any] | None:
        return self._load_manifest(f"inference/research/pairs/{pair_id}/manifest.json")

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        return self._load_manifest(f"inference/research/runs/{run_id}/manifest.json")

    def _load_manifest(self, key: str) -> dict[str, Any] | None:
        if not self.store.exists(key):
            return None
        manifest = self.verify_manifest(key)
        return {**manifest, "manifest_uri": self.store.uri(key)}

    def _put_bytes(self, key: str, content: bytes, *, media_type: str) -> None:
        if self.store.exists(key):
            with self.store.open(key) as source:
                if source.read() != content:
                    raise ResearchRunError(
                        "storage_publication_conflict",
                        f"Immutable Storage object conflicts: {key}",
                    )
            return
        self.store.put_bytes(key, content, media_type=media_type)

    def _put_json(self, key: str, value: Mapping[str, Any]) -> None:
        content = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self._put_bytes(key, content, media_type="application/json")


class InferencePairRunner:
    """Run the same frozen rows through base then exactly one adapter."""

    def __init__(
        self,
        service: Any,
        evaluations: EvaluationSetRepository,
        publisher: ResearchPublisher,
        tracker: SafeTracker | None = None,
    ) -> None:
        self.service = service
        self.evaluations = evaluations
        self.publisher = publisher
        self.tracker = tracker or SafeTracker(NoOpTracker())

    def run(
        self, request: InferencePairRequest, *, owner_id: str = "local"
    ) -> dict[str, Any]:
        pair_id = request.inference_pair_id or str(uuid.uuid4())
        request_checksum = _dataforge_checksum(asdict(request))
        existing = self.publisher.load_pair(pair_id)
        if existing is not None:
            if existing.get("request_checksum") != request_checksum:
                raise ResearchRunError(
                    "inference_pair_idempotency_conflict",
                    "Existing inference pair does not match this request",
                )
            return existing
        started_at = _utc_now()
        evaluation: EvaluationSet | None = None
        try:
            evaluation = self.evaluations.load(request.evaluation_manifest_uri)
            base = self._run_arm(
                request, evaluation, pair_id, "base", None, owner_id=owner_id
            )
            adapter = self._run_arm(
                request,
                evaluation,
                pair_id,
                "adapter",
                request.adapter_manifest_uri,
                owner_id=owner_id,
            )
            mismatches = self._pair_mismatches(base, adapter)
            if mismatches:
                raise ResearchRunError(
                    "pair_validation_failed",
                    "Base and adapter executions are not a reproducible pair",
                    details={"mismatches": mismatches},
                )
            pair = self.publisher.publish_pair(
                {
                    "inference_pair_id": pair_id,
                    "experiment_id": request.context.experiment_id,
                    "comparison_id": request.context.comparison_id,
                    "pair_validation": "passed",
                    "mismatch_reasons": [],
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "evaluation_set": self._evaluation_identity(evaluation),
                    "base_run": self._run_reference(base),
                    "adapter_run": self._run_reference(adapter),
                    "adapter_manifest_uri": request.adapter_manifest_uri,
                    "research_context": asdict(request.context),
                    "request_checksum": request_checksum,
                }
            )
            diagnostic = self._track_pair(request, pair, evaluation)
            return {**pair, **({"tracking": diagnostic} if diagnostic else {})}
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, ResearchRunError)
                else ResearchRunError("inference_pair_failed", str(exc))
            )
            failure = {
                "schema": "cognityx.inference.pair-failure/v1",
                "inference_pair_id": pair_id,
                "experiment_id": request.context.experiment_id,
                "comparison_id": request.context.comparison_id,
                "phase": "execution_or_validation",
                "error": error.code,
                "message": str(error),
                "details": error.details,
                "evaluation_manifest_uri": request.evaluation_manifest_uri,
                "adapter_manifest_uri": request.adapter_manifest_uri,
                "failed_at": _utc_now(),
            }
            try:
                failure_uri = self.publisher.publish_failure(pair_id, failure)
                error.details.setdefault("failure_uri", failure_uri)
            except Exception:
                pass
            raise error

    def _run_arm(
        self,
        request: InferencePairRequest,
        evaluation: EvaluationSet,
        pair_id: str,
        mode: str,
        adapter_manifest_uri: str | None,
        *,
        owner_id: str,
    ) -> dict[str, Any]:
        run_id = f"irun-{_dataforge_checksum([pair_id, mode])[:24]}"
        request_checksum = _dataforge_checksum(asdict(request))
        existing = self.publisher.load_run(run_id)
        if existing is not None:
            if (
                existing.get("request_checksum") != request_checksum
                or existing.get("mode") != mode
                or existing.get("inference_pair_id") != pair_id
            ):
                raise ResearchRunError(
                    "inference_run_idempotency_conflict",
                    "Existing inference run does not match this request arm",
                )
            return existing
        started_at = _utc_now()
        rows: list[dict[str, Any]] = []
        fingerprint: Mapping[str, Any] | None = None
        adapter_identity: Mapping[str, Any] | None = None
        for record in evaluation.records:
            inference_request = self._inference_request(
                request, record, adapter_manifest_uri, mode
            )
            response = self.service.infer(inference_request, owner_id=owner_id)
            current = response.extensions.get("runtime_fingerprint")
            if not isinstance(current, Mapping):
                raise ResearchRunError(
                    "runtime_fingerprint_missing",
                    "Research inference did not return a runtime fingerprint",
                )
            if fingerprint is None:
                fingerprint = current
            elif current.get("sha256") != fingerprint.get("sha256"):
                raise ResearchRunError(
                    "runtime_fingerprint_changed",
                    f"Runtime changed within the {mode} inference run",
                )
            if mode == "adapter":
                selected = response.extensions.get("adapter")
                if not isinstance(selected, Mapping):
                    raise ResearchRunError(
                        "adapter_identity_missing",
                        "Adapted execution did not report adapter identity",
                    )
                adapter_identity = selected
            rows.append(
                self._prediction_row(
                    run_id, pair_id, evaluation, record, response, fingerprint, mode
                )
            )
        assert fingerprint is not None
        run = self.publisher.publish_run(
            {
                "inference_run_id": run_id,
                "inference_pair_id": pair_id,
                "experiment_id": request.context.experiment_id,
                "comparison_id": request.context.comparison_id,
                "arm_id": request.context.arm_id,
                "mode": mode,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "base_model": dict(fingerprint.get("base_model") or {}),
                "runtime_fingerprint": dict(fingerprint),
                "adapter": dict(adapter_identity) if adapter_identity else None,
                "training_lineage": self._training_lineage(
                    adapter_identity, request.context
                ),
                "evaluation_set": self._evaluation_identity(evaluation),
                "record_count": len(rows),
                "aggregate_metrics": self._aggregate(rows),
                "research_context": asdict(request.context),
                "request_checksum": request_checksum,
            },
            tuple(rows),
        )
        diagnostic = self._track_run(request, run, evaluation)
        return {**run, **({"tracking": diagnostic} if diagnostic else {})}

    @staticmethod
    def _inference_request(
        request: InferencePairRequest,
        record: Mapping[str, Any],
        adapter_manifest_uri: str | None,
        mode: str,
    ) -> InferenceRequest:
        question = record.get("question", record.get("input"))
        if question is None:
            raise ResearchRunError(
                "evaluation_manifest_invalid",
                f"Evaluation record has no question/input: {record.get('record_id')}",
            )
        return InferenceRequest(
            model=request.model,
            model_revision=request.model_revision,
            prompt=str(question),
            backend=request.backend,
            profile=request.profile,
            adapter_manifest_uri=adapter_manifest_uri,
            adapter_purpose=(
                AdapterPurpose.EVALUATION if adapter_manifest_uri else None
            ),
            load_policy=(
                LoadPolicy.AUTO if mode == "base" else LoadPolicy.REQUIRE_LOADED
            ),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_output_tokens=request.max_output_tokens,
            thinking=request.thinking,
            stop=request.stop,
            seed=request.context.seed,
            required_context_length=request.required_context_length,
            request_metadata={"evaluation_record_id": record["record_id"]},
            extensions={"runtime": dict(request.runtime)},
        )

    @staticmethod
    def _prediction_row(
        run_id: str,
        pair_id: str,
        evaluation: EvaluationSet,
        record: Mapping[str, Any],
        response: InferenceResponse,
        fingerprint: Mapping[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        manifest = evaluation.manifest
        return {
            "prediction_id": str(uuid.uuid4()),
            "inference_run_id": run_id,
            "inference_pair_id": pair_id,
            "mode": mode,
            "evaluation_record_id": record["record_id"],
            "evaluation_set_id": manifest["evaluation_set_id"],
            "evaluation_set_version": manifest["evaluation_set_version"],
            "research_role": manifest["research_role"],
            "source_record_id": record.get("source_record_id"),
            "source_reference_id": record.get("source_reference_id"),
            "fact_group_id": record.get("fact_group_id"),
            "knowledge_unit_id": record.get("knowledge_unit_id"),
            "question": record.get("question", record.get("input")),
            "generated_answer": response.content,
            "reference_answer": record.get(
                "gold_reference", record.get("reference_answer")
            ),
            "record_provenance": record.get("record_provenance"),
            "source_evidence": record.get("source_evidence"),
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "finish_reason": response.finish_reason.value,
            "thinking": dict(response.extensions.get("thinking") or {}),
            "latency_seconds": response.timings.latency_seconds,
            "time_to_first_token_seconds": (
                response.timings.time_to_first_token_seconds
            ),
            "tokens_per_second": response.timings.tokens_per_second,
            "runtime_fingerprint_sha256": fingerprint["sha256"],
            "adapter": (
                response.extensions.get("adapter") if mode == "adapter" else None
            ),
        }

    @staticmethod
    def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"request_count": len(rows)}
        for source, label in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ):
            values = [row[source] for row in rows if isinstance(row.get(source), int)]
            if values:
                result[label] = sum(values)
        for source, label in (
            ("latency_seconds", "latency_seconds"),
            ("time_to_first_token_seconds", "ttft_seconds"),
            ("tokens_per_second", "tokens_per_second"),
        ):
            values = sorted(
                float(row[source])
                for row in rows
                if isinstance(row.get(source), int | float)
            )
            if values:
                result[f"{label}_average"] = sum(values) / len(values)
                result[f"{label}_p50"] = _percentile(values, 0.50)
                result[f"{label}_p95"] = _percentile(values, 0.95)
        return result

    @staticmethod
    def _pair_mismatches(
        base: Mapping[str, Any], adapter: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        mismatches: list[dict[str, Any]] = []
        if (
            base["runtime_fingerprint"]["sha256"]
            != adapter["runtime_fingerprint"]["sha256"]
        ):
            mismatches.append(
                {
                    "field": "runtime_fingerprint.sha256",
                    "base": base["runtime_fingerprint"]["sha256"],
                    "adapter": adapter["runtime_fingerprint"]["sha256"],
                }
            )
        if base["evaluation_set"] != adapter["evaluation_set"]:
            mismatches.append({"field": "evaluation_set"})
        if base["record_count"] != adapter["record_count"]:
            mismatches.append({"field": "record_count"})
        return mismatches

    @staticmethod
    def _evaluation_identity(evaluation: EvaluationSet) -> dict[str, Any]:
        manifest = evaluation.manifest
        return {
            "manifest_uri": evaluation.manifest_uri,
            "manifest_checksum": manifest["manifest_checksum"],
            "evaluation_set_id": manifest["evaluation_set_id"],
            "evaluation_set_version": manifest["evaluation_set_version"],
            "research_role": manifest["research_role"],
            "records_uri": manifest["records_uri"],
            "records_checksum": manifest["records_checksum"],
        }

    @staticmethod
    def _training_lineage(
        adapter: Mapping[str, Any] | None, context: ResearchContext
    ) -> dict[str, Any] | None:
        if adapter is None:
            return None
        return {
            "adapter_manifest_uri": adapter.get("adapter_manifest_uri"),
            "adapter_manifest_checksum": adapter.get("adapter_manifest_checksum"),
            "adapter_id": adapter.get("adapter_id"),
            "experiment_id": adapter.get("experiment_id"),
            "training_variant_id": adapter.get("training_variant_id")
            or context.training_variant_id,
            "training_run_id": adapter.get("training_run_id")
            or context.training_run_id,
            "dataset": adapter.get("dataset"),
            "research_package_id": context.research_package_id,
        }

    @staticmethod
    def _run_reference(run: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "inference_run_id": run["inference_run_id"],
            "manifest_uri": run["manifest_uri"],
            "manifest_checksum": run["manifest_checksum"],
            "predictions_uri": run["predictions_uri"],
            "predictions_checksum": run["predictions_checksum"],
            "runtime_fingerprint_sha256": run["runtime_fingerprint"]["sha256"],
        }

    def _track_run(
        self,
        request: InferencePairRequest,
        run: Mapping[str, Any],
        evaluation: EvaluationSet,
    ) -> dict[str, str] | None:
        tags = self._tags(request, evaluation)
        tags.update(
            {
                "cognityx.inference_run_id": run["inference_run_id"],
                "cognityx.inference_pair_id": run["inference_pair_id"],
                "cognityx.mode": run["mode"],
                "cognityx.adapter_id": (run.get("adapter") or {}).get("adapter_id"),
                "cognityx.training_run_id": (run.get("adapter") or {}).get(
                    "training_run_id"
                )
                or request.context.training_run_id,
                "cognityx.training_variant_id": (run.get("adapter") or {}).get(
                    "training_variant_id"
                )
                or request.context.training_variant_id,
            }
        )
        metrics = {
            key: value
            for key, value in run["aggregate_metrics"].items()
            if isinstance(value, int | float)
        }
        refs = {
            "run_manifest_uri": run["manifest_uri"],
            "run_manifest_checksum": run["manifest_checksum"],
            "predictions_uri": run["predictions_uri"],
            "predictions_checksum": run["predictions_checksum"],
        }
        adapter = run.get("adapter") or {}
        if adapter.get("adapter_manifest_uri"):
            refs["adapter_manifest_uri"] = adapter["adapter_manifest_uri"]
            refs["adapter_manifest_checksum"] = adapter["adapter_manifest_checksum"]
        return self.tracker.record(
            name=f"inference-{run['mode']}-{run['inference_run_id']}",
            tags=tags,
            metrics=metrics,
            references=refs,
            parent_run_id=request.context.parent_run_id,
        )

    def _track_pair(
        self,
        request: InferencePairRequest,
        pair: Mapping[str, Any],
        evaluation: EvaluationSet,
    ) -> dict[str, str] | None:
        tags = self._tags(request, evaluation)
        tags["cognityx.inference_pair_id"] = pair["inference_pair_id"]
        return self.tracker.record(
            name=f"inference-pair-{pair['inference_pair_id']}",
            tags=tags,
            metrics={"pair_validation_passed": 1},
            references={
                "pair_manifest_uri": pair["manifest_uri"],
                "pair_manifest_checksum": pair["manifest_checksum"],
                "adapter_manifest_uri": request.adapter_manifest_uri,
            },
            parent_run_id=request.context.parent_run_id,
        )

    @staticmethod
    def _tags(
        request: InferencePairRequest, evaluation: EvaluationSet
    ) -> dict[str, Any]:
        context = request.context
        return {
            "cognityx.component": "inference",
            "cognityx.experiment_id": context.experiment_id,
            "cognityx.comparison_id": context.comparison_id,
            "cognityx.arm_id": context.arm_id,
            "cognityx.research_role": evaluation.manifest["research_role"],
            "cognityx.evaluation_set_id": evaluation.manifest["evaluation_set_id"],
            "cognityx.training_run_id": context.training_run_id,
            "cognityx.training_variant_id": context.training_variant_id,
        }


def _percentile(values: list[float], fraction: float) -> float:
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[index]
