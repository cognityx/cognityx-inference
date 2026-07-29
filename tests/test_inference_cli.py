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
