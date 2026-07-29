"""Secret-safe diagnostics for provider and worker operations."""

from __future__ import annotations

import re
from typing import Iterable


_SENSITIVE = re.compile(
    r"(?i)(authorization|api[-_ ]?key|bearer|token|secret|credential)"
    r"(\s*[:=]\s*|\s+)[^\s,;]+"
)
_PROVIDER_TOKEN = re.compile(
    r"\b(?:sk|gsk|xai)-[A-Za-z0-9_-]{16,}\b",
    re.IGNORECASE,
)


def redact(value: object, secrets: Iterable[str] = ()) -> str:
    """Remove known secret values and common credential-shaped fragments."""
    result = str(value)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    result = _PROVIDER_TOKEN.sub("[REDACTED]", result)
    return _SENSITIVE.sub(lambda match: f"{match.group(1)}=[REDACTED]", result)
