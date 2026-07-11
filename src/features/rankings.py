"""FIFA ranking features — optional and NOT used in v1.

The core system is required to work from results/dates/teams/competition
only. Free historical FIFA-ranking dumps exist (e.g. community GitHub
mirrors) but add a second freshness dependency; they can be integrated here
later behind the same strictly-before-kickoff rule ("latest ranking
published before the match"). All models run without this module.
"""
from __future__ import annotations


def available() -> bool:
    return False
