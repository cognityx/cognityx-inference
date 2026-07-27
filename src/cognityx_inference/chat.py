"""Conversation-aware client built on the Cognityx inference client."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
from typing import Any, Callable, Mapping

from cognityx_inference.client import CognityxInferenceClient
from cognityx_inference.storage import ChatSessionRepository


class ChatModelNotLoadedError(RuntimeError):
    """The interactive session needs an explicit local model load."""


class ChatContextError(RuntimeError):
    """History cannot be safely compressed within the certified budget."""


@dataclass(slots=True)
class ChatSettings:
    temperature: float | None = 0.6
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    log_probabilities: bool = False
    top_log_probabilities: int | None = None
    reasoning: bool = False
    timeout_seconds: float | None = None
    first_token_timeout_seconds: float | None = None
    no_token_progress_timeout_seconds: float | None = None

    def request_parameters(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature, "top_p": self.top_p,
            "top_k": self.top_k, "min_p": self.min_p,
            "max_tokens": self.max_tokens, "stop": self.stop,
            "seed": self.seed, "log_probabilities": self.log_probabilities,
            "top_log_probabilities": self.top_log_probabilities,
            "reasoning": {"enabled": True} if self.reasoning else {},
            "timeout_seconds": self.timeout_seconds,
            "first_token_timeout_seconds": self.first_token_timeout_seconds,
            "no_token_progress_timeout_seconds": self.no_token_progress_timeout_seconds,
        }


@dataclass(slots=True)
class CognityxChatSession:
    """Keep bounded conversation history against a certified local context."""

    client: CognityxInferenceClient
    repository: ChatSessionRepository | None = None
    chat_id: str | None = None
    model: str | None = None
    backend: str = "vllm"
    profile: str = "bf16"
    certified_context_length: int | None = None
    certified_profile_id: str | None = None
    system_prompt: str | None = None
    summary: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    settings: ChatSettings = field(default_factory=ChatSettings)
    autosave: bool = False
    show: dict[str, bool] = field(default_factory=lambda: {"power": True, "cpu": True, "ram": True, "perf": True})
    revision: int = 0
    dirty: bool = False

    def load_model(self, model: str, backend: str = "vllm", profile: str = "bf16") -> dict[str, Any]:
        result = self.client.load_model(model, backend, profile, discovery_policy="ask")
        self.model, self.backend, self.profile = model, backend, profile
        resolution = result.get("context_resolution") or {}
        self.certified_context_length = _positive_int(resolution.get("effective_limit"))
        self.certified_profile_id = _clean(result.get("identity", {}).get("runtime", {}).get("certified_profile_id"))
        if self.certified_context_length is None:
            self.refresh_active_model()
        return result

    def refresh_active_model(self) -> None:
        statuses = self.client.model_status()
        if not statuses:
            raise ChatModelNotLoadedError("No local model is loaded. Use /load <model> [backend] [profile].")
        active = statuses[-1]
        identity = active.get("identity") or {}
        runtime = identity.get("runtime") or {}
        self.model = identity.get("model")
        self.backend = identity.get("backend", "vllm")
        self.profile = _clean(runtime.get("quantization")) or self.profile
        self.certified_context_length = _positive_int(_clean(runtime.get("context_length")))
        self.certified_profile_id = _clean(runtime.get("certified_profile_id"))
        if self.certified_context_length is None and self.certified_profile_id:
            profile = self.client.get_certified_profile(self.certified_profile_id)
            self.certified_context_length = _positive_int(profile.get("maximum_certified_context_length"))
        if self.certified_context_length is None:
            raise ChatModelNotLoadedError("The loaded model has no certified context limit. Load it through Cognityx inference first.")

    def send(self, text: str, on_text: Callable[[str], None] | None = None) -> dict[str, Any]:
        if not self.model or not self.certified_context_length:
            self.refresh_active_model()
        self._make_room(text)
        user = {"role": "user", "content": text}
        request_messages = [*self._base_messages(), *self.messages, user]
        final: dict[str, Any] | None = None
        for chunk in self.client.stream_chat(
            model=self.model or "", messages=request_messages, backend=self.backend,
            profile=self.profile, discovery_policy="require_existing", **self.settings.request_parameters(),
        ):
            choices = chunk.get("choices") or []
            if choices:
                content = (choices[0].get("delta") or {}).get("content")
                if content and on_text:
                    on_text(content)
            if "cognityx" in chunk:
                final = chunk
        if final is None:
            raise RuntimeError("Inference stream ended without a final response.")
        response = final.get("cognityx") or {}
        self.messages.extend([user, {"role": "assistant", "content": str(response.get("content") or "")}])
        self.dirty = True
        if self.autosave:
            self.save()
        return final

    def _base_messages(self) -> list[dict[str, str]]:
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            result.append({"role": "system", "content": f"Conversation summary of earlier turns:\n{self.summary}"})
        return result

    def _make_room(self, next_user_text: str) -> None:
        assert self.model and self.certified_context_length
        budget = self.certified_context_length - self.settings.max_tokens - self._compression_reserve()
        if budget <= 0:
            raise ChatContextError("max_tokens and compression reserve exceed the certified context limit.")
        while self._count([*self._base_messages(), *self.messages, {"role": "user", "content": next_user_text}]) > budget:
            if not self.messages:
                raise ChatContextError("System prompt and next message exceed the safe certified chat budget.")
            self._compress_oldest()

    def _compress_oldest(self) -> None:
        # Retain the most recent two turns (four messages) for local continuity.
        old = self.messages[:-4] or self.messages[:1]
        self.messages = self.messages[len(old):]
        text = "\n".join(f"{item['role']}: {item['content']}" for item in old)
        if not text:
            return
        self._summarize_text(text)

    def _summarize_text(self, text: str) -> None:
        assert self.model and self.certified_context_length
        reserve = self._compression_reserve()
        prompt = "Summarize the following earlier conversation faithfully and concisely. Preserve decisions, facts, requests, and unresolved questions.\n\n"
        candidate = text
        while True:
            messages = [
                {"role": "system", "content": "You compress conversation history."},
                *([{ "role": "assistant", "content": f"Existing summary:\n{self.summary}" }] if self.summary else []),
                {"role": "user", "content": prompt + candidate},
            ]
            if self._count(messages) + reserve <= self.certified_context_length:
                response = self.client.chat(
                    model=self.model, messages=messages, backend=self.backend, profile=self.profile,
                    max_tokens=reserve, temperature=0, discovery_policy="require_existing",
                    load_policy="require_loaded",
                )
                self.summary = str(response["choices"][0]["message"].get("content") or "")
                return
            if len(candidate) <= 256:
                raise ChatContextError("History cannot be compressed safely within the certified context. Reduce max_tokens or clear/start a new chat.")
            candidate = candidate[: len(candidate) // 2]

    def _count(self, messages: list[dict[str, str]]) -> int:
        assert self.model
        value = self.client.count_input_tokens(model=self.model, messages=messages, backend=self.backend, profile=self.profile)
        if value is None:
            raise ChatContextError("The active backend cannot count input tokens; safe history compression is unavailable.")
        return value

    def _compression_reserve(self) -> int:
        return min(512, max(64, self.settings.max_tokens // 2))

    def save(self, chat_id: str | None = None) -> str:
        if self.repository is None:
            raise RuntimeError("Chat persistence is unavailable in this client.")
        self.chat_id = chat_id or self.chat_id or self.repository.new_id()
        self.revision += 1
        self.repository.save(self.chat_id, self.revision, self.to_dict())
        self.dirty = False
        return self.chat_id

    def load_chat(self, chat_id: str) -> None:
        if self.repository is None:
            raise RuntimeError("Chat persistence is unavailable in this client.")
        value = self.repository.load_latest(chat_id)
        self.chat_id = chat_id
        self.model = value.get("model")
        self.backend = value.get("backend", "vllm")
        self.profile = value.get("profile", "bf16")
        self.certified_context_length = _positive_int(value.get("certified_context_length"))
        self.certified_profile_id = value.get("certified_profile_id")
        self.system_prompt = value.get("system_prompt")
        self.summary = value.get("summary")
        self.messages = list(value.get("messages") or [])
        self.settings = ChatSettings(**(value.get("settings") or {}))
        self.autosave = bool(value.get("autosave", False))
        self.show.update(value.get("show") or {})
        self.revision = int(value.get("revision", 0))
        self.dirty = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0", "chat_id": self.chat_id, "revision": self.revision,
            "model": self.model, "backend": self.backend, "profile": self.profile,
            "certified_context_length": self.certified_context_length,
            "certified_profile_id": self.certified_profile_id,
            "system_prompt": self.system_prompt, "summary": self.summary,
            "messages": copy.deepcopy(self.messages), "settings": asdict(self.settings),
            "autosave": self.autosave, "show": dict(self.show),
        }


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text.strip("'\"")


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip("'\""))
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None
