"""Secret-safe diagnostics for provider and worker operations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


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


class CredentialResolver:
    """Resolve credentials lazily from environment or one local JSON file."""

    def __init__(self, secrets_file: str | None = None) -> None:
        self.secrets_file = secrets_file

    def resolve(
        self,
        *,
        environment_name: str | None,
        secret_name: str | None,
    ) -> str | None:
        """Return a credential with environment taking precedence."""
        if environment_name:
            environment_value = os.environ.get(environment_name)
            if environment_value and environment_value.strip():
                return environment_value.strip()
        if not self.secrets_file or not secret_name:
            return None
        document = self._document()
        value = document.get(secret_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                f"Credential entry '{secret_name}' must be a string."
            )
        return value.strip() or None

    def available(
        self,
        *,
        environment_name: str | None,
        secret_name: str | None,
    ) -> bool:
        return (
            self.resolve(
                environment_name=environment_name,
                secret_name=secret_name,
            )
            is not None
        )

    def _document(self) -> dict[str, Any]:
        path = Path(self.secrets_file or "")
        try:
            with path.open(encoding="utf-8") as source:
                value = json.load(source)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Configured secrets file does not exist: {path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Configured secrets file cannot be read as JSON: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Configured secrets file must contain a JSON object: {path}"
            )
        return value
