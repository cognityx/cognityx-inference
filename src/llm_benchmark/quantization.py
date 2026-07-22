from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import transformers
from transformers import BitsAndBytesConfig
from transformers.utils import is_kernels_available, is_triton_available
from transformers.utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION


class QuantizationResolutionError(ValueError):
    """Raised when a requested profile cannot be resolved safely."""


class NativeQuantizationRequiredError(QuantizationResolutionError):
    """Raised when ``native`` is requested for an unquantized checkpoint."""


class BitsAndBytesUnavailableError(QuantizationResolutionError):
    """Raised when runtime INT4/INT8 is requested without bitsandbytes."""


@dataclass(frozen=True)
class HardwareInfo:
    accelerator_type: str
    gpu_name: str | None
    compute_capability: str | None
    cuda_version: str | None
    pytorch_version: str
    transformers_version: str
    triton_version: str | None
    triton_kernels_available: bool
    triton_compatible: bool
    triton_requirement: str
    kernels_version: str | None
    kernels_compatible: bool
    kernels_requirement: str


@dataclass(frozen=True)
class QuantizationResolution:
    requested_profile: str
    effective_profile: str
    checkpoint_quantized: bool
    checkpoint_quantization_method: str | None
    runtime_quantization_method: str | None
    quantization_source: str
    pass_quantization_config: bool
    quantization_config: Any
    compute_dtype: torch.dtype | None
    profile_was_adapted: bool
    adaptation_reason: str | None
    warning: str | None
    bf16_dequantization_fallback_attempted: bool
    hardware: HardwareInfo

    def load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.compute_dtype is not None:
            kwargs["torch_dtype"] = self.compute_dtype
        if self.pass_quantization_config:
            kwargs["quantization_config"] = self.quantization_config
        return kwargs

    def diagnostic(self) -> dict[str, Any]:
        result = {
            "requested_profile": self.requested_profile,
            "effective_profile": self.effective_profile,
            "checkpoint_quantized": self.checkpoint_quantized,
            "checkpoint_quantization_method": self.checkpoint_quantization_method,
            "runtime_quantization_method": self.runtime_quantization_method,
            "quantization_source": self.quantization_source,
            "profile_was_adapted": self.profile_was_adapted,
            "adaptation_reason": self.adaptation_reason,
            "pass_quantization_config": self.pass_quantization_config,
            "warning": self.warning,
            "bf16_dequantization_fallback_attempted": (
                self.bf16_dequantization_fallback_attempted
            ),
            "hardware_preflight": asdict(self.hardware),
            "native_quantization_runtime_preflight": inspect_native_quantization_runtime(
                self.checkpoint_quantization_method, self.hardware
            ),
        }
        return result


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_hardware_info() -> HardwareInfo:
    gpu_name: str | None = None
    compute_capability: str | None = None
    accelerator = torch.accelerator.current_accelerator() or torch.device("cpu")
    accelerator_type = accelerator.type
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = f"{major}.{minor}"
    triton_requirement = "3.4.0" if accelerator_type == "cuda" else "3.5.0"
    return HardwareInfo(
        accelerator_type=accelerator_type,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        cuda_version=torch.version.cuda,
        pytorch_version=str(torch.__version__),
        transformers_version=transformers.__version__,
        triton_version=_package_version("triton"),
        triton_kernels_available=importlib.util.find_spec("triton_kernels") is not None,
        triton_compatible=is_triton_available(triton_requirement),
        triton_requirement=f">={triton_requirement}",
        kernels_version=_package_version("kernels"),
        kernels_compatible=is_kernels_available(),
        kernels_requirement=f">={KERNELS_MIN_VERSION},<{KERNELS_MAX_VERSION}",
    )


def _unavailable_native_runtime(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "runtime_backend": None,
        "reason": reason,
        "runtime_fallback": "bf16",
        "expected_performance_impact": "high",
        "detection_source": "preflight",
    }


def inspect_native_quantization_runtime(
    method: str | None,
    hardware: HardwareInfo,
) -> dict[str, Any]:
    """Predict the native execution path using Transformers' backend requirements."""
    if method != "mxfp4":
        return {
            "status": "unknown" if method else "not_applicable",
            "runtime_backend": method,
            "reason": (
                "No model-specific runtime preflight is implemented for this method."
                if method
                else None
            ),
            "runtime_fallback": None,
            "expected_performance_impact": None,
            "detection_source": "preflight",
        }

    capability = hardware.compute_capability
    if hardware.accelerator_type == "cuda" and capability is not None:
        major, minor = (int(part) for part in capability.split(".", maxsplit=1))
        if (major, minor) < (7, 5):
            return _unavailable_native_runtime(
                f"CUDA compute capability {capability}; MXFP4 requires >=7.5"
            )
    if not hardware.triton_compatible:
        reason = (
            f"Missing Python package: triton{hardware.triton_requirement}"
            if hardware.triton_version is None
            else f"Incompatible Triton version: installed {hardware.triton_version}; "
            f"required {hardware.triton_requirement}"
        )
        return _unavailable_native_runtime(reason)
    if not hardware.kernels_compatible:
        reason = (
            f"Missing Python package: kernels{hardware.kernels_requirement}"
            if hardware.kernels_version is None
            else f"Incompatible kernels version: installed {hardware.kernels_version}; "
            f"required {hardware.kernels_requirement}"
        )
        return _unavailable_native_runtime(reason)
    return {
        "status": "enabled",
        "runtime_backend": "mxfp4",
        "reason": None,
        "runtime_fallback": None,
        "expected_performance_impact": None,
        "detection_source": "preflight",
    }


def detect_loaded_native_quantization_runtime(
    model: Any,
    method: str | None,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Read Transformers' post-load quantizer state to report the actual path."""
    if method is None:
        return preflight
    quantizer = getattr(model, "hf_quantizer", None)
    quantization_config = getattr(quantizer, "quantization_config", None)
    dequantize = getattr(quantization_config, "dequantize", None)
    if dequantize is True:
        result = dict(preflight)
        result.update(
            {
                "status": "unavailable",
                "runtime_backend": None,
                "runtime_fallback": "bf16",
                "expected_performance_impact": "high",
                "detection_source": "transformers_quantizer_post_load",
            }
        )
        if not result.get("reason"):
            result["reason"] = "Transformers enabled BF16 dequantization fallback."
        return result
    if dequantize is False:
        return {
            "status": "enabled",
            "runtime_backend": method,
            "reason": None,
            "runtime_fallback": None,
            "expected_performance_impact": None,
            "detection_source": "transformers_quantizer_post_load",
        }
    result = dict(preflight)
    result["detection_source"] = "preflight_no_post_load_quantizer_state"
    return result


def extract_checkpoint_quantization_method(config: Any) -> str | None:
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return None
    for name in ("quant_method", "quantization_method"):
        value = (
            quantization_config.get(name)
            if isinstance(quantization_config, Mapping)
            else getattr(quantization_config, name, None)
        )
        value = getattr(value, "value", value)
        if value is not None and str(value).strip():
            return str(value).lower()
    return None


def checkpoint_has_quantization_config(config: Any) -> bool:
    return getattr(config, "quantization_config", None) is not None


def resolve_load_profile(
    requested_profile: str,
    config: Any,
    hardware: HardwareInfo | None = None,
    *,
    has_bitsandbytes: bool | None = None,
) -> QuantizationResolution:
    hardware = hardware or collect_hardware_info()
    native_method = extract_checkpoint_quantization_method(config)
    checkpoint_quantized = checkpoint_has_quantization_config(config)

    if requested_profile == "native" and not checkpoint_quantized:
        raise NativeQuantizationRequiredError(
            "The native profile requires a checkpoint with an embedded "
            "quantization_config."
        )

    if checkpoint_quantized:
        adapted = requested_profile not in {"native", "auto"}
        warning = None
        if adapted:
            warning = (
                f"The checkpoint is already quantized with {native_method or 'an unknown method'}. "
                f"Runtime {requested_profile} quantization will not be applied on top "
                "of the embedded weights."
            )
        return QuantizationResolution(
            requested_profile=requested_profile,
            effective_profile="native",
            checkpoint_quantized=True,
            checkpoint_quantization_method=native_method,
            runtime_quantization_method=None,
            quantization_source="checkpoint_native",
            pass_quantization_config=False,
            quantization_config=None,
            compute_dtype=None,
            profile_was_adapted=adapted,
            adaptation_reason="checkpoint_already_quantized" if adapted else None,
            warning=warning,
            bf16_dequantization_fallback_attempted=False,
            hardware=hardware,
        )

    effective_profile = "bf16" if requested_profile == "auto" else requested_profile
    if effective_profile in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if effective_profile == "bf16" else torch.float16
        return QuantizationResolution(
            requested_profile=requested_profile,
            effective_profile=effective_profile,
            checkpoint_quantized=False,
            checkpoint_quantization_method=None,
            runtime_quantization_method=None,
            quantization_source=(
                "automatic_resolution" if requested_profile == "auto" else "explicit_dtype"
            ),
            pass_quantization_config=False,
            quantization_config=None,
            compute_dtype=dtype,
            profile_was_adapted=requested_profile == "auto",
            adaptation_reason=(
                "unquantized_checkpoint_default_bf16" if requested_profile == "auto" else None
            ),
            warning=None,
            bf16_dequantization_fallback_attempted=False,
            hardware=hardware,
        )

    available = bitsandbytes_available() if has_bitsandbytes is None else has_bitsandbytes
    if not available:
        raise BitsAndBytesUnavailableError(
            f"The {effective_profile} profile requires bitsandbytes. "
            "Install it with: uv add bitsandbytes"
        )
    if effective_profile == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        compute_dtype = torch.float16
    else:
        bf16_supported = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
    return QuantizationResolution(
        requested_profile=requested_profile,
        effective_profile=effective_profile,
        checkpoint_quantized=False,
        checkpoint_quantization_method=None,
        runtime_quantization_method="bitsandbytes",
        quantization_source="runtime_bitsandbytes",
        pass_quantization_config=True,
        quantization_config=quantization_config,
        compute_dtype=compute_dtype,
        profile_was_adapted=False,
        adaptation_reason=None,
        warning=None,
        bf16_dequantization_fallback_attempted=False,
        hardware=hardware,
    )
