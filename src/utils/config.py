"""Configuration loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Resolve a path from config relative to the project root."""
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p
