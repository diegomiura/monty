"""Input validation for CLI entry points."""
from __future__ import annotations

import pandas as pd


def validate_team(name: str, known_teams: set[str], alias_resolver=None) -> str:
    """Return the canonical team name or raise with a helpful message."""
    if alias_resolver is not None:
        name = alias_resolver(name)
    if name in known_teams:
        return name
    from difflib import get_close_matches

    hints = get_close_matches(name, sorted(known_teams), n=5, cutoff=0.6)
    hint_txt = f" Did you mean: {', '.join(hints)}?" if hints else ""
    raise ValueError(f"Unknown team '{name}'.{hint_txt}")


def validate_probs_sum(probs: dict, keys: list[str], tol: float = 1e-6) -> None:
    s = sum(probs[k] for k in keys)
    if abs(s - 1.0) > tol:
        raise AssertionError(f"Probabilities {keys} sum to {s}, expected 1.0")


def validate_date_range(ts: pd.Timestamp, lo: str = "1872-01-01", hi: str = "2100-01-01") -> None:
    if not (pd.Timestamp(lo, tz="UTC") <= ts <= pd.Timestamp(hi, tz="UTC")):
        raise ValueError(f"Date {ts} outside plausible range [{lo}, {hi}]")
