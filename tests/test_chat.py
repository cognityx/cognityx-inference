from __future__ import annotations

from cognityx_inference.chat import ChatSettings, CognityxChatSession
from cognityx_inference.storage import ChatSessionRepository
from cognityx_storage import LocalStorageBackend, StorageClient


class FakeClient:
    def __init__(self) -> None:
        self.counts = []
        self.summaries = []

    def model_status(self):
        return [{"identity": {"model": "model-a", "backend": "vllm", "runtime": {"context_length": "40", "certified_profile_id": "profile-1", "quantization": "int4"}}}]

    def count_input_tokens(self, *, messages, **kwargs):
        self.counts.append(messages)
        return sum(len(item["content"].split()) + 1 for item in messages)

    def stream_chat(self, **kwargs):
        yield {"choices": [{"delta": {"content": "answer"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "cognityx": {"content": "answer", "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}, "timings": {"token_generation_seconds": 0.1, "time_to_first_token_seconds": 0.01}, "extensions": {"legacy_result": {"quality_indicators": {"thinking_tokens": 0, "answer_tokens": 1}}}}}

    def chat(self, **kwargs):
        self.summaries.append(kwargs)
        return {"choices": [{"message": {"content": "compressed summary"}}]}

    def load_model(self, *args, **kwargs):
        return {"context_resolution": {"effective_limit": 40}, "identity": {"runtime": {"certified_profile_id": "profile-1"}}}


def test_chat_compresses_before_exceeding_certified_budget() -> None:
    client = FakeClient()
    session = CognityxChatSession(
        client, model="model-a", backend="vllm", profile="int4",
        certified_context_length=100, settings=ChatSettings(max_tokens=8),
        messages=[
            {"role": "user", "content": "one two three four five six seven eight"},
            {"role": "assistant", "content": "nine ten eleven twelve thirteen fourteen"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ],
    )

    final = session.send("next question with some more words")

    assert final["cognityx"]["content"] == "answer"
    assert session.summary == "compressed summary"
    assert client.summaries
    assert session.messages[-1] == {"role": "assistant", "content": "answer"}


def test_autosave_persists_each_completed_turn_and_loads_latest(tmp_path) -> None:
    repo = ChatSessionRepository(StorageClient(LocalStorageBackend(tmp_path)).for_user("alice"))
    session = CognityxChatSession(
        FakeClient(), repo, model="model-a", certified_context_length=100,
        settings=ChatSettings(max_tokens=8), autosave=True,
    )
    session.send("hello")
    first_revision = session.revision
    session.send("again")

    loaded = CognityxChatSession(FakeClient(), repo)
    loaded.load_chat(session.chat_id or "")

    assert first_revision == 1
    assert loaded.revision == 2
    assert loaded.messages[-1]["content"] == "answer"


def test_refresh_requires_active_model() -> None:
    class Empty(FakeClient):
        def model_status(self): return []

    session = CognityxChatSession(Empty())
    try:
        session.refresh_active_model()
    except Exception as exc:
        assert "Use /load" in str(exc)
    else:
        raise AssertionError("missing loaded-model error")


def test_switching_server_clears_stale_model_state() -> None:
    class Switchable(FakeClient):
        base_url = "http://first:8000"

        def set_base_url(self, value):
            self.base_url = value.rstrip("/")

    session = CognityxChatSession(
        Switchable(), model="model-a", certified_context_length=100,
        certified_profile_id="profile-1",
    )

    session.set_server_url("http://second:8013/")

    assert session.client.base_url == "http://second:8013"
    assert session.model is None
    assert session.certified_context_length is None
