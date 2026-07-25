from __future__ import annotations

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
    monkeypatch.setattr(
        "cognityx_inference.cli.CognityxInferenceClient", FakeClient
    )

    main(["infer", "--model", "model-a", "--prompt", "hello"])

    assert capsys.readouterr().out == "first second\n"


def test_infer_no_stream_preserves_json_response(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "cognityx_inference.cli.CognityxInferenceClient", FakeClient
    )

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
