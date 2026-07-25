from __future__ import annotations

import os

from cognityx_inference.environment import configure_huggingface_cache
from cognityx_inference.api import _model_metadata_http_exception
from cognityx_inference.errors import ModelMetadataUnavailableError


def test_shared_huggingface_cache_is_selected_when_present(tmp_path, monkeypatch):
    root = tmp_path / "huggingface"
    (root / "hub").mkdir(parents=True)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    selected = configure_huggingface_cache(root)

    assert selected == root
    assert os.environ["HF_HOME"] == str(root)
    assert os.environ["HF_HUB_CACHE"] == str(root / "hub")


def test_explicit_huggingface_cache_is_preserved(tmp_path, monkeypatch):
    root = tmp_path / "huggingface"
    root.mkdir()
    monkeypatch.setenv("HF_HOME", "/explicit/home")
    monkeypatch.setenv("HF_HUB_CACHE", "/explicit/hub")

    configure_huggingface_cache(root)

    assert os.environ["HF_HOME"] == "/explicit/home"
    assert os.environ["HF_HUB_CACHE"] == "/explicit/hub"


def test_missing_model_metadata_is_a_not_found_api_error():
    error = _model_metadata_http_exception(
        ModelMetadataUnavailableError(
            "Cannot find the requested files in the disk cache"
        )
    )

    assert error.status_code == 404
    assert error.detail["error"] == "model_metadata_unavailable"
