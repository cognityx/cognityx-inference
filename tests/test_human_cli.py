from pathlib import Path
from types import SimpleNamespace

import pytest

from cognityx_inference import cli, configuration
from cognityx_inference.evaluation import cli as boundary_cli
from cognityx_inference.presentation import render_human


def test_human_renderer_handles_empty_table_nested_and_full_values() -> None:
    assert render_human([]) == "No records."
    assert render_human([{}]) == "Record 1:\n  No fields."
    uri = "storage://local-main/models/certified/full/profile.json"
    output = render_human([{"profile_id": "profile-full-id", "uri": uri}])
    assert "profile-full-id" in output
    assert uri in output
    assert "\x1b" not in output
    assert "Nested:\n  State: ready" in render_human({"nested": {"state": "ready"}})


def test_config_human_calls_static_resolver_once(monkeypatch, capsys) -> None:
    calls = 0

    class Resolution:
        def to_dict(self):
            return {
                "component": "inference",
                "valid": True,
                "master_config": {"kind": "built-in", "sha256": None},
                "config_layers": [],
                "overrides": [],
                "effective": {"secrets_file": {"contents_read": False}},
                "warnings": [],
                "errors": [],
            }

    def resolve(path):
        nonlocal calls
        calls += 1
        return Resolution()

    monkeypatch.setattr(configuration, "resolve_inference_configuration", resolve)
    cli.main(["config", "show", "--human"])
    assert calls == 1
    output = capsys.readouterr().out
    assert "Component: inference" in output
    assert "Contents read: false" in output


def test_server_status_and_watch_human_call_manager_once(monkeypatch, capsys) -> None:
    status_calls = 0

    class Client:
        def __init__(self, *args, **kwargs):
            return None

        def server_status(self):
            nonlocal status_calls
            status_calls += 1
            return {"state": "ready", "worker_id": "worker-full-id"}

        def stream_server_events(self, *, after):
            yield {"sequence": after + 1, "state": "ready"}

    monkeypatch.setattr(cli, "CognityxInferenceClient", Client)
    cli.main(["server", "status", "--human"])
    assert status_calls == 1
    assert "Worker id: worker-full-id" in capsys.readouterr().out

    cli.main(["server", "watch", "--after", "6", "--human"])
    output = capsys.readouterr().out
    assert "Sequence: 7" in output
    assert "State: ready" in output


def test_provider_human_invokes_discovery_once(monkeypatch, capsys) -> None:
    calls = 0
    configured = SimpleNamespace(
        providers={},
        credential_resolver=lambda: object(),
    )

    class Registry:
        def __init__(self, *args, **kwargs):
            return None

        def discover_models(self, provider, *, refresh, timeout_seconds):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                to_dict=lambda: {
                    "provider": provider,
                    "status": "ready",
                    "models": ["model-full-id"],
                }
            )

    monkeypatch.setattr(cli.InferenceConfiguration, "load", lambda path: configured)
    monkeypatch.setattr(cli, "build_provider_adapters", lambda *args: {})
    monkeypatch.setattr(cli, "ProviderRegistry", Registry)
    cli.main(
        [
            "providers",
            "models",
            "--provider",
            "openai",
            "--human",
        ]
    )
    assert calls == 1
    assert "model-full-id" in capsys.readouterr().out


def test_provider_json_human_conflict_precedes_configuration(monkeypatch) -> None:
    called = False

    def load(path):
        nonlocal called
        called = True

    monkeypatch.setattr(cli.InferenceConfiguration, "load", load)
    with pytest.raises(SystemExit) as captured:
        cli.main(["providers", "status", "--json", "--human"])

    assert captured.value.code == 2
    assert called is False


def test_boundary_plan_human_is_non_json(capsys) -> None:
    config = Path(__file__).parents[1] / "examples" / "boundary" / "config.toml"
    boundary_cli.main(["--config", str(config), "--plan", "--human"])
    output = capsys.readouterr().out
    assert "Configured trial count:" in output
    assert "Axes:" in output
    assert not output.lstrip().startswith("{")
