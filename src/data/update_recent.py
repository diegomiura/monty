"""Refresh recent data (thin wrapper; the primary source ships one CSV that
always contains the full history, so 'recent update' = full idempotent
rebuild from freshly downloaded raw files)."""
from __future__ import annotations

from src.data.merge import refresh_dataset


def update(config: dict | None = None, offline: bool = False):
    return refresh_dataset(config, offline=offline)
