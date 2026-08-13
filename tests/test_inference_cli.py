from __future__ import annotations

import json
import stat

import pytest

from cognityx_inference.cli import build_service, main
from cognityx_inference.discovery import DiscoveryConfig


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def stream_chat(self, **kwargs):
        yield {"choices": [{"delta": {"content": "first "}}]}
        yield {"choices": [{"delta": {"content": "second"}}]}
        yield {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 4},
        }

    def chat(self, **kwargs):
        return {"choices": [{"message": {"content": "complete"}}]}


def test_installed_service_uses_built_in_discovery_defaults(monkeypatch) -> None:
    """Starting an installed wheel must not depend on source-only examples."""

    class SelectedConfiguration:
        providers = {}
        tracking = {}

        @staticmethod
        def credential_resolver():
            return object()

    class StorageClient:
        @staticmethod
        def for_shared_data():
            return object()

    class StopAfterDiscoveryConfig(Exception):
        pass

    captured = {}

    def capture_discovery(*args, config, **kwargs):
        captured["config"] = config
        raise StopAfterDiscoveryConfig

    def reject_source_example(*args, **kwargs):
        raise AssertionError("The service tried to read a source-checkout example")

    monkeypatch.setattr(
        "cognityx_inference.cli.build_provider_adapters", lambda *args: {}
    )
    monkeypatch.setattr("cognityx_inference.cli.ModelManager", lambda *args: object())
    monkeypatch.setattr(
        "cognityx_inference.cli.ProviderRegistry", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr("cognityx_storage.StorageClient", StorageClient)
    monkeypatch.setattr(
        "cognityx_inference.cli.CertifiedProfileRepository", lambda *args: object()
    )
    monkeypatch.setattr(
        "cognityx_inference.cli.BoundaryArtifactRepository", lambda *args: object()
    )
    monkeypatch.setattr("cognityx_inference.cli.JobRepository", lambda *args: object())
    monkeypatch.setattr(
        "cognityx_inference.cli.BoundaryDiscoveryCoordinator", capture_discovery
    )
    monkeypatch.setattr(DiscoveryConfig, "from_toml", reject_source_example)

    with pytest.raises(StopAfterDiscoveryConfig):
        build_service(SelectedConfiguration())

    assert captured["config"] == DiscoveryConfig()


def test_infer_streams_text_by_default(monkeypatch, capsys) -> None:
    monkeypatch.setattr("cognityx_inference.cli.CognityxInferenceClient", FakeClient)

    main(["infer", "--model", "model-a", "--prompt", "hello"])

    assert capsys.readouterr().out == "first second\n"


def test_config_inspection_dispatches_before_runtime_setup(
    tmp_path, monkeypatch, capsys
) -> None:
    config = tmp_path / "inference.toml"
    config.write_text(
        'secrets_file="private.json"\n[manager]\nport=9001\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cognityx_inference.cli.build_service",
        lambda *_args, **_kwargs: pytest.fail("service was built"),
    )
    monkeypatch.setattr(
        "cognityx_inference.cli.build_provider_adapters",
        lambda *_args, **_kwargs: pytest.fail("provider adapter was built"),
    )

    main(["config", "show", "--config", str(config)])
    shown = json.loads(capsys.readouterr().out)

    assert shown["master_config"]["path"] == str(config.resolve())
    assert shown["master_config"]["selected_by"] == "explicit"
    assert shown["effective"]["secrets_file"]["contents_read"] is False
    assert "manager.port" in shown["config_layers"][0]["changed_keys"]
    assert shown["field_sources"]["manager.port"] == str(config.resolve())
    assert shown["field_sources"]["manager.host"] == "built-in"
    assert shown["valid"] is True

    main(["config", "validate", "--config", str(config)])
    validated = json.loads(capsys.readouterr().out)
    assert validated["master_config"] == shown["master_config"]


def test_config_validate_uses_project_discovery_and_redacts_secret_names(
    tmp_path, monkeypatch, capsys
) -> None:
    config = tmp_path / ".cognityx" / "inference.toml"
    config.parent.mkdir()
    config.write_text(
        '[providers.demo]\nadapter="openai_compatible"\n'
        'base_url="https://user:password@example.test/v1?access_token=uri-secret"\n'
        'api_key_secret="never-show"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COGNITYX_INFERENCE_CONFIG", raising=False)

    main(["config", "validate"])
    output = capsys.readouterr().out
    shown = json.loads(output)

    assert shown["master_config"]["selected_by"] == "project"
    assert "never-show" not in output
    assert "user:password" not in output
    assert "uri-secret" not in output


def test_missing_config_returns_nonzero_json(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["config", "validate", "--config", "missing.toml"])

    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_infer_no_stream_preserves_json_response(monkeypatch, capsys) -> None:
    monkeypatch.setattr("cognityx_inference.cli.CognityxInferenceClient", FakeClient)

    main(
        [
            "infer",
            "--model",
            "model-a",
            "--prompt",
            "hello",
            "--no-stream",
        ]
    )

    assert '"content": "complete"' in capsys.readouterr().out


def test_infer_forwards_adapter_options(monkeypatch, capsys) -> None:
    captured = {}

    class Client(FakeClient):
        def chat(self, **kwargs):
            captured.update(kwargs)
            return super().chat(**kwargs)

    monkeypatch.setattr("cognityx_inference.cli.CognityxInferenceClient", Client)

    main(
        [
            "infer",
            "--model",
            "model-a",
            "--model-revision",
            "commit-1",
            "--prompt",
            "hello",
            "--adapter-manifest",
            "storage://local-main/models/adapter/manifest.json",
            "--no-stream",
        ]
    )

    assert captured["model_revision"] == "commit-1"
    assert captured["adapter_purpose"] == "evaluation"
    assert captured["adapter_manifest_uri"].startswith("storage://")
    assert captured["thinking"] == "disabled"
    assert '"content": "complete"' in capsys.readouterr().out


def test_research_pair_cli_forwards_frozen_context(monkeypatch, capsys) -> None:
    captured = {}

    class Client(FakeClient):
        def run_research_pair(self, payload):
            captured.update(payload)
            return {"pair_validation": "passed"}

    monkeypatch.setattr("cognityx_inference.cli.CognityxInferenceClient", Client)
    main(
        [
            "research",
            "pair",
            "--evaluation-manifest",
            "storage://local-main/datasets/evaluation/manifest.json",
            "--adapter-manifest",
            "storage://local-main/models/adapter/manifest.json",
            "--model",
            "Qwen/Qwen3-8B",
            "--experiment-id",
            "exp-1",
            "--seed",
            "7",
        ]
    )

    assert captured["research_context"]["experiment_id"] == "exp-1"
    assert captured["research_context"]["seed"] == 7
    assert captured["thinking"] == "disabled"
    assert captured["max_output_tokens"] == 512
    assert json.loads(capsys.readouterr().out)["pair_validation"] == "passed"


def test_cli_enables_thinking_explicitly(monkeypatch, capsys) -> None:
    captured = {}

    class Client(FakeClient):
        def chat(self, **kwargs):
            captured.update(kwargs)
            return super().chat(**kwargs)

    monkeypatch.setattr("cognityx_inference.cli.CognityxInferenceClient", Client)
    main(
        [
            "infer",
            "--model",
            "model-a",
            "--prompt",
            "hello",
            "--thinking",
            "--no-stream",
        ]
    )

    assert captured["thinking"] == "enabled"
    assert '"content": "complete"' in capsys.readouterr().out


def test_provider_setup_creates_only_an_empty_private_template(
    tmp_path, capsys
) -> None:
    secrets = tmp_path / "private" / "providers.json"
    config = tmp_path / "inference.toml"
    config.write_text(
        f'secrets_file = "{secrets}"\n',
        encoding="utf-8",
    )

    main(
        [
            "providers",
            "setup",
            "--config",
            str(config),
            "--provider",
            "groq",
            "--create-template",
            "--yes",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["created"] is True
    assert json.loads(secrets.read_text(encoding="utf-8")) == {"GROQ_API_KEY": ""}
    assert stat.S_IMODE(secrets.stat().st_mode) == 0o600
