from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from cognityx_inference.contracts import ModelCapabilities
from cognityx_inference.lifecycle import ModelIdentity, ModelManager, ModelState


@dataclass
class FakeBackend:
    model: str
    runtime: dict
    loaded: bool = False
    unload_count: int = 0
    capabilities = ModelCapabilities(lifecycle=True)

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False
        self.unload_count += 1

    def status(self) -> dict:
        return {"ready": self.loaded}


def test_model_is_reused_and_deferred_unload_waits_for_lease() -> None:
    created: list[FakeBackend] = []

    def factory(model: str, runtime: dict) -> FakeBackend:
        backend = FakeBackend(model, runtime)
        created.append(backend)
        return backend

    manager = ModelManager({"fake": factory})
    first = manager.acquire("model-a", "fake", {"quantization": "int4"})
    second = manager.acquire("model-a", "fake", {"quantization": "int4"})
    identity = ModelIdentity.create(
        "model-a", "fake", {"quantization": "int4"}
    )

    assert len(created) == 1
    assert manager.statuses()[0].active_leases == 2
    assert manager.unload(identity) is False
    assert manager.statuses()[0].state is ModelState.DRAINING

    first.release()
    assert created[0].loaded
    second.release()
    assert not created[0].loaded
    assert manager.statuses() == ()


def test_runtime_identity_prevents_incompatible_reuse() -> None:
    manager = ModelManager({"fake": FakeBackend})

    manager.load("model-a", "fake", {"quantization": "bf16"})
    manager.load("model-a", "fake", {"quantization": "int4"})

    assert len(manager.statuses()) == 2


def test_concurrent_loads_share_one_initialization() -> None:
    started = threading.Event()
    release = threading.Event()
    created = []

    class SlowBackend(FakeBackend):
        def load(self) -> None:
            started.set()
            release.wait(timeout=2)
            super().load()

    def factory(model, runtime):
        backend = SlowBackend(model, runtime)
        created.append(backend)
        return backend

    manager = ModelManager({"fake": factory})
    outcomes = []
    first = threading.Thread(
        target=lambda: outcomes.append(manager.load("model-a", "fake"))
    )
    second = threading.Thread(
        target=lambda: outcomes.append(manager.load("model-a", "fake"))
    )
    first.start()
    started.wait(timeout=1)
    loading = manager.statuses()
    assert len(loading) == 1
    assert loading[0].state is ModelState.LOADING
    second.start()
    time.sleep(0.02)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert len(created) == 1
    assert len(outcomes) == 2
    assert all(item.state is ModelState.READY for item in outcomes)
