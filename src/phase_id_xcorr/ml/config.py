"""Configuration helpers for ML workflows."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from path."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(payload).__name__}")
    return payload


def dump_yaml(payload: dict[str, Any], path: Path) -> None:
    """Write YAML mapping to path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def resolve_path(path_value: str | Path, *, base_dir: Path, repo_root: Path) -> Path:
    """Resolve path against config file directory first, then repo root."""

    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()

    local_candidate = (base_dir / p).resolve()
    if local_candidate.exists():
        return local_candidate

    return (repo_root / p).resolve()


def get_required(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    """Read required key from mapping."""

    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {where}")
    return mapping[key]


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries and return a new mapping."""

    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted_key(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set dotted key path in a nested dictionary."""

    parts = [p.strip() for p in dotted_key.split(".") if p.strip()]
    if not parts:
        raise ValueError("dotted_key must not be empty")

    cur = payload
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def parse_override_value(raw: str) -> Any:
    """Parse CLI override value from text into primitive/object types."""

    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"

    for caster in (int, float):
        try:
            return caster(text)
        except Exception:
            pass

    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        parsed = yaml.safe_load(text)
        return parsed

    if text.lower() in {"null", "none"}:
        return None

    return text


def apply_overrides(base: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply key=value dotted-path overrides to a config mapping."""

    out = copy.deepcopy(base)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value")
        key, raw = item.split("=", 1)
        set_dotted_key(out, key.strip(), parse_override_value(raw))
    return out
