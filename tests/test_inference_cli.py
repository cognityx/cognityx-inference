from __future__ import annotations

import json
import stat

from cognityx_inference.cli import main


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


def test_infer_streams_text_by_default(monkeypatch, capsys) -> None:
    monkeypatch.setattr("cognityx_inference.cli.CognityxInferenceClient", FakeClient)

    main(["infer", "--model", "model-a", "--prompt", "hello"])

    assert capsys.readouterr().out == "first second\n"


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
