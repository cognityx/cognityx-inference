"""Verify and materialize Training adapters through Cognityx Storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cognityx_inference.errors import AdapterError

ADAPTER_SCHEMA = "cognityx.training.adapter/v1"
REQUIRED_ADAPTER_FILES = frozenset({"adapter_config.json", "adapter_model.safetensors"})


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _bundle_checksum(files: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in files
    ]
    normalized.sort(key=lambda item: str(item["path"]))
    return hashlib.sha256(_stable_json(normalized).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedAdapter:
    """Scientific adapter identity plus one checksum-keyed local copy."""

    adapter_id: str
    manifest_uri: str
    manifest_checksum: str
    bundle_checksum: str
    local_path: Path
    base_model: Mapping[str, Any]
    adapter: Mapping[str, Any]
    dataset: Mapping[str, Any]
    experiment_id: str
    training_variant_id: str
    training_run_id: str
    status: str

    @property
    def lora_int_id(self) -> int:
        digest = hashlib.sha256(
            f"{self.adapter_id}\x1f{self.bundle_checksum}".encode("utf-8")
        ).hexdigest()
        return int(digest[:15], 16) % 2_147_483_646 + 1

    def public_identity(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_manifest_uri": self.manifest_uri,
            "adapter_manifest_checksum": self.manifest_checksum,
            "bundle_checksum": self.bundle_checksum,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "training_variant_id": self.training_variant_id,
            "training_run_id": self.training_run_id,
            "dataset": dict(self.dataset),
            "base_model": dict(self.base_model),
            "adapter": dict(self.adapter),
        }


class AdapterRepository:
    """Consume the public Training manifest without importing Training."""

    def __init__(
        self, storage_runtime: Any, cache_root: str | Path | None = None
    ) -> None:
        self.storage_runtime = storage_runtime
        selected = cache_root or os.environ.get("COGNITYX_INFERENCE_ADAPTER_CACHE")
        self.cache_root = (
            Path(selected)
            if selected
            else (Path.home() / ".cache" / "cognityx" / "inference" / "adapters")
        )

    def verify_and_materialize(self, manifest_uri: str) -> VerifiedAdapter:
        try:
            manifest_object = self.storage_runtime.resolve_uri(
                manifest_uri, role_name="model"
            )
        except Exception as exc:
            raise AdapterError(
                "adapter_manifest_invalid",
                f"Adapter manifest URI cannot be resolved: {manifest_uri}",
            ) from exc
        if not manifest_object.exists():
            raise AdapterError(
                "adapter_artifact_missing",
                f"Adapter manifest does not exist: {manifest_uri}",
            )
        try:
            with manifest_object.open() as source:
                raw_manifest = source.read()
            manifest = json.loads(raw_manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError(
                "adapter_manifest_invalid", "Adapter manifest is not valid JSON"
            ) from exc
        if not isinstance(manifest, dict):
            raise AdapterError(
                "adapter_manifest_invalid", "Adapter manifest must be an object"
            )
        if manifest.get("schema_version") != ADAPTER_SCHEMA:
            raise AdapterError(
                "adapter_manifest_invalid",
                f"Unsupported adapter schema: {manifest.get('schema_version')!r}",
            )
        required = (
            "adapter_id",
            "experiment_id",
            "training_variant_id",
            "training_run_id",
            "status",
            "base_model",
            "adapter",
            "dataset",
            "files",
            "bundle_checksum",
        )
        missing_fields = [name for name in required if name not in manifest]
        if missing_fields:
            raise AdapterError(
                "adapter_manifest_invalid",
                "Adapter manifest is missing required fields",
                details={"fields": missing_fields},
            )
        if manifest["status"] != "candidate":
            raise AdapterError(
                "adapter_manifest_invalid",
                "Only candidate adapters may be selected for evaluation",
            )
        if any(
            not isinstance(manifest[name], dict)
            for name in ("base_model", "adapter", "dataset")
        ):
            raise AdapterError(
                "adapter_manifest_invalid",
                "Adapter base_model, adapter, and dataset fields must be objects",
            )
        if manifest["adapter"].get("format") != "peft" or manifest["adapter"].get(
            "type"
        ) not in {"lora", "qlora"}:
            raise AdapterError(
                "adapter_manifest_invalid",
                "Adapter must use the published PEFT LoRA or QLoRA format",
            )
        files = manifest["files"]
        if not isinstance(files, list) or any(
            not isinstance(item, dict) for item in files
        ):
            raise AdapterError(
                "adapter_manifest_invalid", "Adapter manifest files must be objects"
            )
        if any(not {"path", "sha256", "size_bytes"}.issubset(item) for item in files):
            raise AdapterError(
                "adapter_manifest_invalid",
                "Every adapter file requires path, sha256, and size_bytes",
            )
        names = {str(item.get("path", "")) for item in files}
        if len(names) != len(files):
            raise AdapterError(
                "adapter_manifest_invalid", "Adapter file paths must be unique"
            )
        try:
            invalid_file = next(
                (
                    item
                    for item in files
                    if int(item["size_bytes"]) < 0 or len(str(item["sha256"])) != 64
                ),
                None,
            )
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                "adapter_manifest_invalid",
                "Adapter file size and checksum values are invalid",
            ) from exc
        if invalid_file is not None:
            raise AdapterError(
                "adapter_manifest_invalid",
                "Adapter file size and checksum values are invalid",
            )
        missing_files = sorted(REQUIRED_ADAPTER_FILES - names)
        if missing_files:
            raise AdapterError(
                "adapter_artifact_missing",
                "Adapter manifest omits required files",
                details={"files": missing_files},
            )
        calculated_bundle = _bundle_checksum(files)
        if calculated_bundle != manifest["bundle_checksum"]:
            raise AdapterError(
                "adapter_checksum_mismatch", "Adapter bundle checksum does not match"
            )
        self._verify_checksums_object(manifest_object, files, calculated_bundle)
        target = self.cache_root / calculated_bundle
        if target.is_dir():
            self._verify_local_files(target, files)
        else:
            self._materialize_files(manifest_object, target, files)
        return VerifiedAdapter(
            adapter_id=str(manifest["adapter_id"]),
            manifest_uri=manifest_uri,
            manifest_checksum=_sha256(raw_manifest),
            bundle_checksum=calculated_bundle,
            local_path=target,
            base_model=dict(manifest["base_model"]),
            adapter=dict(manifest["adapter"]),
            dataset=dict(manifest["dataset"]),
            experiment_id=str(manifest["experiment_id"]),
            training_variant_id=str(manifest["training_variant_id"]),
            training_run_id=str(manifest["training_run_id"]),
            status=str(manifest["status"]),
        )

    @staticmethod
    def compatibility(
        adapter: VerifiedAdapter, actual: Mapping[str, Any]
    ) -> dict[str, str]:
        comparisons = {
            "name": "adapter_base_model_mismatch",
            "resolved_revision": "adapter_base_model_mismatch",
            "tokenizer_revision": "adapter_tokenizer_mismatch",
            "chat_template_checksum": "adapter_chat_template_mismatch",
        }
        result: dict[str, str] = {}
        expected_requested = adapter.base_model.get("requested_revision")
        actual_requested = actual.get("requested_revision")
        if expected_requested is None or actual_requested is None:
            result["requested_revision"] = "unverified"
        elif str(expected_requested) == str(actual_requested):
            result["requested_revision"] = "matched"
        elif (
            adapter.base_model.get("resolved_revision") is not None
            and actual.get("resolved_revision") is not None
            and str(adapter.base_model["resolved_revision"])
            == str(actual["resolved_revision"])
        ):
            result["requested_revision"] = "different_reference_same_revision"
        else:
            raise AdapterError(
                "adapter_base_model_mismatch",
                "Adapter requested revision does not match the selected base model",
                details={
                    "expected": expected_requested,
                    "actual": actual_requested,
                },
            )
        for field, code in comparisons.items():
            expected = adapter.base_model.get(field)
            observed = actual.get(field)
            if expected is None or observed is None:
                result[field] = "unverified"
            elif str(expected) != str(observed):
                raise AdapterError(
                    code,
                    f"Adapter {field} does not match the selected base model",
                    details={"expected": expected, "actual": observed},
                )
            else:
                result[field] = "matched"
        return result

    def _verify_checksums_object(
        self, manifest_object: Any, files: list[dict[str, Any]], bundle: str
    ) -> None:
        try:
            checksums_object = manifest_object.resolve_relative("checksums.json")
            with checksums_object.open() as source:
                checksums = json.load(source)
        except Exception as exc:
            raise AdapterError(
                "adapter_artifact_missing", "Adapter checksums.json is unavailable"
            ) from exc
        if not isinstance(checksums, dict):
            raise AdapterError(
                "adapter_checksum_mismatch",
                "Adapter checksums.json must be an object",
            )
        if (
            checksums.get("schema_version") != "cognityx.training.adapter-checksums/v1"
            or checksums.get("files") != files
            or checksums.get("bundle_checksum") != bundle
        ):
            raise AdapterError(
                "adapter_checksum_mismatch",
                "Adapter checksums.json does not match the manifest",
            )

    def _materialize_files(
        self, manifest_object: Any, target: Path, files: list[dict[str, Any]]
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            for item in files:
                relative = str(item["path"])
                try:
                    source = manifest_object.resolve_relative(relative)
                    if not source.exists():
                        raise FileNotFoundError(relative)
                    materialized = source.materialize()
                except Exception as exc:
                    raise AdapterError(
                        "adapter_artifact_missing",
                        f"Adapter file is unavailable: {relative}",
                    ) from exc
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(materialized, destination)
            self._verify_local_files(temporary, files)
            try:
                temporary.replace(target)
            except FileExistsError:
                self._verify_local_files(target, files)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def _verify_local_files(root: Path, files: list[dict[str, Any]]) -> None:
        for item in files:
            relative = str(item["path"])
            path = root / relative
            if not path.is_file():
                raise AdapterError(
                    "adapter_artifact_missing", f"Adapter file is missing: {relative}"
                )
            content = path.read_bytes()
            if len(content) != int(item["size_bytes"]):
                raise AdapterError(
                    "adapter_checksum_mismatch",
                    f"Adapter file size does not match: {relative}",
                )
            if _sha256(content) != str(item["sha256"]):
                raise AdapterError(
                    "adapter_checksum_mismatch",
                    f"Adapter file checksum does not match: {relative}",
                )
