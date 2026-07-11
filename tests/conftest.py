import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def toy_matches():
    """Small synthetic canonical table: 3 teams, known dates/results,
    including a same-day pair and an extra-time knockout match."""
    rows = [
        # date, a, b, ga90, gb90, comp, neutral, et_a, et_b, pens, pa, pb, stage
        ("2020-01-01", "Alpha", "Beta", 2, 0, "Friendly", False, 0, 0, False, np.nan, np.nan, ""),
        ("2020-02-01", "Beta", "Gamma", 1, 1, "FIFA World Cup qualification", False, 0, 0, False, np.nan, np.nan, ""),
        ("2020-03-01", "Alpha", "Gamma", 0, 3, "FIFA World Cup qualification", True, 0, 0, False, np.nan, np.nan, ""),
        # same-day pair
        ("2020-04-01", "Alpha", "Beta", 1, 0, "Friendly", False, 0, 0, False, np.nan, np.nan, ""),
        ("2020-04-01", "Gamma", "Beta", 2, 2, "Friendly", False, 0, 0, False, np.nan, np.nan, ""),
        # knockout decided in extra time: 1-1 after 90, 2-1 after ET
        ("2020-05-01", "Alpha", "Gamma", 1, 1, "FIFA World Cup", True, 1, 0, False, np.nan, np.nan, "Final"),
        # shootout match: 0-0 after 90 and ET, Beta wins pens
        ("2020-06-01", "Beta", "Gamma", 0, 0, "FIFA World Cup", True, 0, 0, True, 3.0, 4.0, "Semi-final"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "date", "team_a", "team_b", "goals_a_90", "goals_b_90", "competition",
            "neutral", "goals_a_extra_time", "goals_b_extra_time", "went_to_penalties",
            "penalty_goals_a", "penalty_goals_b", "stage",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["went_to_extra_time"] = (df["goals_a_extra_time"] + df["goals_b_extra_time"] > 0) | df["went_to_penalties"]
    df["winner_90"] = np.select(
        [df["goals_a_90"] > df["goals_b_90"], df["goals_a_90"] < df["goals_b_90"]],
        ["team_a", "team_b"], default="draw",
    )
    df["goals_90_confirmed"] = True
    df["group"] = ""
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df
