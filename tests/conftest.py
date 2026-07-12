import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def rich_matches():
    """Denser synthetic canonical table: 4 teams, ~90 seeded random matches.

    Unlike `toy_matches` (7 matches, deliberately sparse), every feature
    column here has observed values. sklearn >= 1.9 with numpy >= 2.x
    raises inside HistGradientBoosting binning when a training column is
    entirely NaN, so tests that fit the GB model must use this fixture.
    """
    rng = np.random.default_rng(7)
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    comps = ["Friendly", "FIFA World Cup qualification", "UEFA Nations League"]
    rows = []
    date = pd.Timestamp("2019-01-15")
    for i in range(90):
        a, b = rng.choice(teams, size=2, replace=False)
        rows.append(
            {
                "date": date,
                "team_a": a,
                "team_b": b,
                "goals_a_90": int(rng.poisson(1.4)),
                "goals_b_90": int(rng.poisson(1.1)),
                "competition": comps[i % len(comps)],
                "neutral": bool(i % 4 == 0),
                "goals_a_extra_time": 0,
                "goals_b_extra_time": 0,
                "went_to_penalties": False,
                "penalty_goals_a": np.nan,
                "penalty_goals_b": np.nan,
                "stage": "",
            }
        )
        date += pd.Timedelta(days=int(rng.integers(7, 25)))
    df = pd.DataFrame(rows)
    df["went_to_extra_time"] = False
    df["winner_90"] = np.select(
        [df["goals_a_90"] > df["goals_b_90"], df["goals_a_90"] < df["goals_b_90"]],
        ["team_a", "team_b"], default="draw",
    )
    df["goals_90_confirmed"] = True
    df["group"] = ""
    df["match_id"] = [f"r{i}" for i in range(len(df))]
    return df


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
