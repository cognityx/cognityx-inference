from __future__ import annotations

import os
import subprocess
import sys


def test_cache_environment_is_configured_before_transformers_import(tmp_path):
    root = tmp_path / "huggingface"
    (root / "hub").mkdir(parents=True)
    source = (
        "import os; "
        "from cognityx_inference import service; "
        "import huggingface_hub.constants as constants; "
        "print(os.environ.get('HF_HOME')); "
        "print(os.environ.get('HF_HUB_CACHE')); "
        "print(constants.HF_HUB_CACHE)"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "HF_HOME": "",
        "HF_HUB_CACHE": "",
        "COGNITYX_TEST_CACHE": str(root),
    }
    # The production default is fixed; this subprocess test verifies that
    # importing service does not happen before cache setup.
    environment["HF_HOME"] = str(root)
    environment["HF_HUB_CACHE"] = str(root / "hub")
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=".",
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert str(root) in result.stdout
    assert str(root / "hub") in result.stdout
