from __future__ import annotations

from cognityx_inference.model_references import (
    cached_model_ids,
    resolve_local_model_reference,
)


def _cached(root, model):
    organization, name = model.split("/", 1)
    (root / f"models--{organization}--{name}").mkdir()


def test_unique_same_size_reference_resolves_to_cached_model(tmp_path):
    _cached(tmp_path, "Qwen/Qwen3-8B")
    _cached(tmp_path, "Qwen/Qwen3-14B")

    resolution = resolve_local_model_reference("Qwen/Qwen-8B", tmp_path)

    assert resolution.resolved == "Qwen/Qwen3-8B"
    assert resolution.changed
    assert resolution.source == "unique_cached_organization_and_size"


def test_exact_cached_identifier_always_wins(tmp_path):
    _cached(tmp_path, "Qwen/Qwen3-8B")

    resolution = resolve_local_model_reference("Qwen/Qwen3-8B", tmp_path)

    assert resolution.resolved == "Qwen/Qwen3-8B"
    assert not resolution.changed


def test_ambiguous_size_match_is_not_silently_resolved(tmp_path):
    _cached(tmp_path, "Qwen/Qwen3-8B")
    _cached(tmp_path, "Qwen/Qwen2.5-8B")

    resolution = resolve_local_model_reference("Qwen/Qwen-8B", tmp_path)

    assert resolution.resolved == "Qwen/Qwen-8B"
    assert resolution.source == "ambiguous_cache_match"
    assert set(resolution.candidates) == {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen2.5-8B",
    }


def test_cached_model_ids_decodes_hub_directories(tmp_path):
    _cached(tmp_path, "Qwen/Qwen3-8B")

    assert cached_model_ids(tmp_path) == ("Qwen/Qwen3-8B",)
