"""Launch the inference API in the isolated, CUDA-compatible vLLM runtime."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


class VLLMRuntimeError(RuntimeError):
    """The optional local vLLM runtime is unavailable or unhealthy."""


def _runtime_source_roots(project_root: Path) -> tuple[str, ...]:
    """Return project and available sibling service source roots."""
    candidates = (
        project_root / "src",
        project_root.parent / "cognityx-resource" / "src",
        project_root.parent / "cognityx-storage" / "src",
        project_root.parent / "cognityx-jobs" / "src",
    )
    return tuple(str(path) for path in candidates if path.is_dir())


def ensure_vllm_runtime(argv: list[str]) -> None:
    """Re-exec the API in ``.venv-vllm`` and validate its vLLM import.

    The regular project environment deliberately does not depend on vLLM.  A
    server does, however, expose vLLM as its default local backend, so this
    check must happen before it accepts discovery or model-load requests.
    """
    project_root = Path(__file__).resolve().parents[2]
    isolated_python = project_root / ".venv-vllm" / "bin" / "python"
    marker = "COGNITYX_INFERENCE_VLLM_RUNTIME"
    if os.environ.get(marker) == "1":
        try:
            import vllm  # noqa: F401
        except Exception as exc:
            raise VLLMRuntimeError(
                "The isolated .venv-vllm cannot import vLLM. Run the "
                "repository vLLM setup instructions, then retry. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return
    if importlib.util.find_spec("vllm") is not None:
        return
    if not isolated_python.is_file():
        raise VLLMRuntimeError(
            "Missing isolated .venv-vllm. Create it with `uv venv "
            ".venv-vllm --python 3.12 --seed` and install vllm there."
        )
    environment = os.environ.copy()
    environment[marker] = "1"
    environment.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    environment["PYTHONPATH"] = ":".join(
        item
        for item in (
            *_runtime_source_roots(project_root),
            environment.get("PYTHONPATH"),
        )
        if item
    )
    cuda_home = next(
        (project_root / ".venv-vllm" / "lib").glob(
            "python*/site-packages/nvidia/cu13"
        ),
        None,
    )
    if cuda_home is not None:
        environment.setdefault("CUDA_HOME", str(cuda_home))
        paths = [str(cuda_home / "lib")]
        if Path("/usr/lib/wsl/lib").is_dir():
            paths.append("/usr/lib/wsl/lib")
        for name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            if environment.get(name):
                paths.append(environment[name])
            environment[name] = ":".join(paths)
        environment["PATH"] = ":".join(
            (str(isolated_python.parent), str(cuda_home / "bin"), environment["PATH"])
        )
    os.execve(
        str(isolated_python),
        [str(isolated_python), "-m", "cognityx_inference.cli", *argv],
        environment,
    )
