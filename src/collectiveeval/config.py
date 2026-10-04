"""Experiment configuration loading and stable hashing."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def resolve_env_vars(value: Any) -> Any:
    """Resolve `${VAR}` placeholders in a nested config object."""

    if isinstance(value, dict):
        return {key: resolve_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    return value


def load_yaml_config(path: str | Path, *, resolve_env: bool = True) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping")
    return cast(dict[str, Any], resolve_env_vars(payload) if resolve_env else payload)


def stable_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def redact_secrets(config: dict[str, Any]) -> dict[str, Any]:
    secret_words = ("api_key", "secret", "password")
    secret_exact_keys = {"token", "access_token", "auth_token", "bearer_token", "refresh_token"}

    def redact(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                child_key: redact(child_value, child_key)
                for child_key, child_value in value.items()
            }
        key_lower = key.lower()
        if key_lower in secret_exact_keys or any(word in key_lower for word in secret_words):
            return "***REDACTED***"
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return cast(dict[str, Any], redact(config))
