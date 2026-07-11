"""Competition classification and pre-match tournament context."""
from __future__ import annotations

FRIENDLY_LIKE = {"Friendly", "FIFA Series"}

CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Championship",  # includes the Gold Cup era name in this dataset
    "Gold Cup",
    "Oceania Nations Cup",
}

KNOCKOUT_STAGES = {
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Quarterfinal",
    "Semi-final",
    "Semifinal",
    "Third place",
    "Match for third place",
    "Final",
    "knockout",
}


def competition_class(name: str) -> str:
    """Map a tournament name to an importance class used for Elo K and
    feature one-hots."""
    if name in FRIENDLY_LIKE:
        return "friendly"
    if name == "FIFA World Cup":
        return "world_cup"
    if "qualification" in name.lower():
        return "qualifier"
    if "Nations League" in name:
        return "nations_league"
    if name in CONTINENTAL_FINALS:
        return "continental_final"
    return "other_tournament"


def is_knockout_stage(stage: str) -> bool:
    return stage in KNOCKOUT_STAGES
