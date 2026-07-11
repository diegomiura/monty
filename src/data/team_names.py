"""Team-name standardization.

Canonical names = the current-name convention of martj42/international_results,
which already records historical matches under the modern name for continuity
teams (verified: 'Soviet Union' matches appear as 'Russia', 'Zaïre' as
'DR Congo', 'FR Yugoslavia'/'Serbia and Montenegro' as 'Serbia').

Documented merge decisions (inherited from the source, kept deliberately):
  * Russia includes Soviet Union and CIS.
  * Serbia includes FR Yugoslavia and Serbia and Montenegro.
  * Germany includes West Germany; 'German DR' (East Germany) stays separate.
  * Czechoslovakia and Yugoslavia (pre-breakup) remain distinct teams that
    simply have no matches after dissolution.

ALIASES maps alternative spellings (openfootball, FIFA style, common user
input) onto canonical names. former_names.csv rows are added at load time so
e.g. 'Zaïre' resolves to 'DR Congo'.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ALIASES: dict[str, str] = {
    # openfootball / FIFA-style spellings
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "UAE": "United Arab Emirates",
    "West Germany": "Germany",
    "East Germany": "German DR",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
}


def load_alias_table(former_names_csv: str | Path | None = None) -> dict[str, str]:
    """Static aliases plus former-name rows from the dataset itself."""
    aliases = dict(ALIASES)
    if former_names_csv is not None and Path(former_names_csv).exists():
        fn = pd.read_csv(former_names_csv)
        for _, row in fn.iterrows():
            aliases.setdefault(row["former"], row["current"])
    return aliases


def make_resolver(aliases: dict[str, str]):
    def resolve(name: str) -> str:
        name = str(name).strip()
        return aliases.get(name, name)

    return resolve


def audit_names(df: pd.DataFrame, known: set[str]) -> list[str]:
    """Return names in df (team_a/team_b) not present in the known set."""
    seen = set(df["team_a"]) | set(df["team_b"])
    return sorted(seen - known)
