"""Club-name handling for football-data.co.uk.

This module is deliberately shaped differently from
:mod:`src.data.team_names`. For internationals the problem is many-to-one:
several sources spell the same country differently, so an alias table maps
variants onto a canonical name.

For a single-source, single-league ingest that problem does not exist.
Verified over all 33 Premier League seasons (1993-94 → 2025-26):
football-data.co.uk uses **51 distinct club names with no renames and no
spelling drift**. Every name that appears in only one or two seasons is a
genuine short top-flight spell (Barnsley 1997-98, Blackpool 2010-11,
Swindon and Oldham 1993-94, Luton 2023-24), not an alternative spelling.

So the risk worth engineering against is not "many spellings of one club" but
**silent drift**: a future season introducing ``Man Utd`` alongside the
existing ``Man United`` would create a phantom club with no history, quietly
resetting its Elo and rolling form. :func:`check_club_drift` makes that loud.

``ALIASES`` therefore exists only for *user input* at the CLI (and as the
seed of a cross-source map if a second source is ever added), never to repair
the source data.
"""
from __future__ import annotations

from difflib import get_close_matches

import pandas as pd

#: Every club appearing in the Premier League 1993-94 .. 2025-26, in
#: football-data.co.uk's own spelling. Pinned deliberately: this set is the
#: drift detector, so it must be updated consciously when a genuinely new
#: club is promoted (see check_club_drift).
KNOWN_CLUBS: frozenset[str] = frozenset({
    "Arsenal", "Aston Villa", "Barnsley", "Birmingham", "Blackburn",
    "Blackpool", "Bolton", "Bournemouth", "Bradford", "Brentford",
    "Brighton", "Burnley", "Cardiff", "Charlton", "Chelsea", "Coventry",
    "Crystal Palace", "Derby", "Everton", "Fulham", "Huddersfield", "Hull",
    "Ipswich", "Leeds", "Leicester", "Liverpool", "Luton", "Man City",
    "Man United", "Middlesbrough", "Newcastle", "Norwich", "Nott'm Forest",
    "Oldham", "Portsmouth", "QPR", "Reading", "Sheffield United",
    "Sheffield Weds", "Southampton", "Stoke", "Sunderland", "Swansea",
    "Swindon", "Tottenham", "Watford", "West Brom", "West Ham", "Wigan",
    "Wimbledon", "Wolves",
})

#: Common ways a person might type a club name, mapped to the source's
#: spelling. Used by the CLI only — never applied to the source data.
ALIASES: dict[str, str] = {
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Man Utd": "Man United",
    "MUFC": "Man United",
    "Manchester City": "Man City",
    "Man. City": "Man City",
    "MCFC": "Man City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Forest": "Nott'm Forest",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "West Bromwich": "West Brom",
    "WBA": "West Brom",
    "Sheffield Wednesday": "Sheffield Weds",
    "Sheffield Wed": "Sheffield Weds",
    "Sheff Utd": "Sheffield United",
    "Sheff United": "Sheffield United",
    "Queens Park Rangers": "QPR",
    "Newcastle United": "Newcastle",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Birmingham City": "Birmingham",
    "Derby County": "Derby",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Huddersfield Town": "Huddersfield",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Charlton Athletic": "Charlton",
    "Coventry City": "Coventry",
    "Wigan Athletic": "Wigan",
    "Crystal Palace FC": "Crystal Palace",
}


def resolve_club(name: str) -> str:
    """Map user input onto the source's spelling (identity if already exact)."""
    n = str(name).strip()
    return ALIASES.get(n, n)


def validate_club(name: str, known: set[str] | frozenset[str] | None = None) -> str:
    """Canonical club name, or raise with close-match suggestions."""
    known = set(known if known is not None else KNOWN_CLUBS)
    n = resolve_club(name)
    if n in known:
        return n
    hints = get_close_matches(n, sorted(known), n=5, cutoff=0.6)
    hint = f" Did you mean: {', '.join(hints)}?" if hints else ""
    raise ValueError(f"Unknown club '{name}'.{hint}")


def check_club_drift(df: pd.DataFrame, known: frozenset[str] = KNOWN_CLUBS) -> list[str]:
    """Names present in the data but not in :data:`KNOWN_CLUBS`.

    A non-empty result is not automatically an error — a newly promoted club
    is a legitimate new name — but it must be reviewed and pinned rather than
    absorbed silently, because an unrecognised spelling of an *existing* club
    would otherwise start from a default rating with no history.
    """
    seen = set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna())
    return sorted(seen - set(known))


def describe_drift(unknown: list[str], known: frozenset[str] = KNOWN_CLUBS) -> str:
    """Human-readable drift report naming the likely existing club for each
    unknown spelling, so a rename is easy to tell from a promotion."""
    if not unknown:
        return "no unknown club names"
    lines = [f"{len(unknown)} unrecognised club name(s):"]
    for name in unknown:
        near = get_close_matches(name, sorted(known), n=3, cutoff=0.6)
        if near:
            lines.append(f"  {name!r} — close to existing {near}; likely a RENAME, not a new club")
        else:
            lines.append(f"  {name!r} — no close match; likely a newly promoted club")
    return "\n".join(lines)
