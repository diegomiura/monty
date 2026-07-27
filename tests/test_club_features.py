"""Club feature builder — leakage guarantees first.

The club canonical table carries post-match statistics (shots, shots on
target, corners) on the same row as the match they describe. They are there
so that *later* matches can use them as lagged form, and the danger is
obvious: nothing about the data's shape prevents a match's own shot count
from entering its own feature row. A docstring is not a guard, so these
tests are the guard.

The strongest available check does not inspect the implementation at all.
Corrupt a match's own outcome and statistics to absurd values; if the
feature table is unchanged, no feature row could have read them. Applying
that to the *final* chronological match tests every row at once, since no
match follows it.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.club_features import (
    CLUB_FEATURE_COLUMNS,
    ClubFeatureBuilder,
    build_club_feature_table,
)
from src.utils.config import resolve


@pytest.fixture(scope="module")
def club_cfg():
    with open(resolve("config/clubs.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture
def club_matches():
    """Synthetic EPL-shaped table: 6 clubs, two seasons, with a same-day
    pair and full post-match statistics."""
    rng = np.random.default_rng(11)
    clubs = ["Arsenal", "Chelsea", "Everton", "Fulham", "Leeds", "Wolves"]
    rows, date = [], pd.Timestamp("2020-08-15")
    for i in range(90):
        h, a = rng.choice(clubs, size=2, replace=False)
        gh, ga = int(rng.poisson(1.5)), int(rng.poisson(1.2))
        rows.append({
            "date": date, "kickoff_time": pd.NaT,
            "team_a": h, "team_b": a,
            "goals_a_90": gh, "goals_b_90": ga,
            "shots_a": int(rng.integers(5, 22)), "shots_b": int(rng.integers(5, 22)),
            "shots_target_a": int(rng.integers(1, 10)), "shots_target_b": int(rng.integers(1, 10)),
            "competition": "Premier League",
            "season": "2020-21" if i < 45 else "2021-22",
            "neutral": False,
        })
        # every third match shares a date with the previous one
        if i % 3 != 0:
            date += pd.Timedelta(days=4)
    df = pd.DataFrame(rows)
    df["winner_90"] = np.select(
        [df["goals_a_90"] > df["goals_b_90"], df["goals_a_90"] < df["goals_b_90"]],
        ["team_a", "team_b"], default="draw",
    )
    df["match_id"] = [f"c{i}" for i in range(len(df))]
    return df.sort_values("date").reset_index(drop=True)


# --- leakage -------------------------------------------------------------- #

def test_no_match_sees_its_own_result_or_statistics(club_matches, club_cfg):
    """Corrupt the last match's goals and shots; every feature row, its own
    included, must be byte-identical."""
    base = build_club_feature_table(club_matches, club_cfg)

    corrupted = club_matches.copy()
    last = corrupted.index[-1]
    corrupted.loc[last, ["goals_a_90", "goals_b_90"]] = [99, 0]
    corrupted.loc[last, ["shots_a", "shots_b"]] = [999, 0]
    corrupted.loc[last, ["shots_target_a", "shots_target_b"]] = [999, 0]
    corrupted.loc[last, "winner_90"] = "team_a"

    after = build_club_feature_table(corrupted, club_cfg)
    pd.testing.assert_frame_equal(
        base[CLUB_FEATURE_COLUMNS], after[CLUB_FEATURE_COLUMNS],
        check_exact=True,
        obj="feature table changed when only the final match's own result changed",
    )


def test_same_day_matches_cannot_see_each_other(club_matches, club_cfg):
    """Two matches on one date must be built from state that excludes both."""
    counts = club_matches.groupby("date").size()
    shared = counts[counts > 1].index[0]
    same_day = club_matches.index[club_matches["date"] == shared]
    assert len(same_day) >= 2

    base = build_club_feature_table(club_matches, club_cfg)
    corrupted = club_matches.copy()
    first = same_day[0]
    corrupted.loc[first, ["goals_a_90", "goals_b_90"]] = [99, 0]
    corrupted.loc[first, ["shots_a", "shots_target_a"]] = [999, 999]
    after = build_club_feature_table(corrupted, club_cfg)

    for idx in same_day[1:]:
        pd.testing.assert_series_equal(
            base.loc[idx, CLUB_FEATURE_COLUMNS],
            after.loc[idx, CLUB_FEATURE_COLUMNS],
            check_exact=True,
            obj=f"row {idx} changed when a same-day match's result changed",
        )


def test_later_matches_do_change_when_history_changes(club_matches, club_cfg):
    """Control for the tests above: if nothing ever propagated, they would
    pass vacuously. Corrupting an EARLY match must move later rows."""
    base = build_club_feature_table(club_matches, club_cfg)
    corrupted = club_matches.copy()
    corrupted.loc[0, ["goals_a_90", "goals_b_90"]] = [9, 0]
    corrupted.loc[0, ["shots_a", "shots_target_a"]] = [40, 25]
    after = build_club_feature_table(corrupted, club_cfg)
    assert not base[CLUB_FEATURE_COLUMNS].equals(after[CLUB_FEATURE_COLUMNS]), (
        "no later feature row responded to an early result — the builder is "
        "not propagating history at all, which would make the leakage tests vacuous"
    )


def test_features_are_finite_or_explicitly_missing(club_matches, club_cfg):
    feats = build_club_feature_table(club_matches, club_cfg)
    block = feats[CLUB_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert not np.isinf(block).any(), "infinite feature value"


# --- promotion prior ------------------------------------------------------ #

def test_promoted_club_enters_below_the_league_mean(club_cfg):
    """A club with no history must not be rated average on arrival."""
    fb = ClubFeatureBuilder(club_cfg)
    for club, rating in [("Arsenal", 1700.0), ("Chelsea", 1650.0), ("Everton", 1450.0)]:
        fb.elo.ratings[club] = rating
    fb._matches_applied = 1          # the competition is already running
    mean = np.mean([1700.0, 1650.0, 1450.0])

    entry = fb.initial_rating_for("Wrexham")
    assert entry == pytest.approx(mean + club_cfg["elo"]["promoted_elo_offset"])
    assert entry < mean


def test_first_ever_club_uses_the_configured_initial_rating(club_cfg):
    fb = ClubFeatureBuilder(club_cfg)
    assert fb.initial_rating_for("Arsenal") == pytest.approx(club_cfg["elo"]["initial_rating"])


def test_founding_clubs_all_start_level(club_matches, club_cfg):
    """Every club in the dataset's first season is 'new' only because the
    data starts there — none was promoted into anything.

    Applying the promotion offset to them penalises each in turn and, since
    each penalised entry lowers the running mean, compounds: on real EPL data
    this opened a 426-point spread across the clubs of the opening day, driven
    purely by the order fixtures were listed in.
    """
    fb = ClubFeatureBuilder(club_cfg)
    first_day = club_matches[club_matches["date"] == club_matches["date"].min()]
    for _, r in first_day.iterrows():
        fb.fixture_row(r["team_a"], r["team_b"], r["date"])
    entries = [fb.elo.ratings[c] for c in fb.seen]
    assert len(entries) >= 2
    assert max(entries) - min(entries) == pytest.approx(0.0), (
        f"founding clubs entered with a {max(entries) - min(entries):.1f} point "
        "spread; entry rating depends on fixture order"
    )
    assert entries[0] == pytest.approx(club_cfg["elo"]["initial_rating"])


def test_promotion_anchor_ignores_long_departed_clubs(club_cfg):
    """A side relegated years ago keeps a frozen rating. It must not drag
    down the mean a newly promoted club is anchored to."""
    fb = ClubFeatureBuilder(club_cfg)
    now = pd.Timestamp("2024-08-01")
    fb.elo.ratings.update({"Arsenal": 1700.0, "Chelsea": 1650.0, "Swindon": 1000.0})
    fb.elo.last_played.update({
        "Arsenal": now - pd.Timedelta(days=80),
        "Chelsea": now - pd.Timedelta(days=80),
        "Swindon": now - pd.Timedelta(days=9000),      # gone for decades
    })
    fb._matches_applied = 1

    entry = fb.initial_rating_for("Wrexham", now)
    active_mean = np.mean([1700.0, 1650.0])
    assert entry == pytest.approx(active_mean + club_cfg["elo"]["promoted_elo_offset"])


def test_promoted_prior_is_applied_once_not_every_match(club_matches, club_cfg):
    """The offset is an entry condition; it must not re-apply and drag a
    club down on every appearance."""
    fb = ClubFeatureBuilder(club_cfg)
    fb.run(club_matches)
    ratings = [fb.elo.rating(c) for c in ("Arsenal", "Chelsea", "Everton")]
    assert all(np.isfinite(r) for r in ratings)
    assert max(ratings) - min(ratings) < 600, "ratings diverged implausibly"


# --- season boundary ------------------------------------------------------ #

def test_season_gap_regresses_ratings_toward_the_mean(club_cfg):
    """Squad turnover between campaigns. The summer break (~80 days) must
    trigger the configured regression; a midweek gap must not."""
    fb = ClubFeatureBuilder(club_cfg)
    fb.elo.ratings["Arsenal"] = 1800.0
    fb.elo.last_played["Arsenal"] = pd.Timestamp("2021-05-23")

    in_season = fb.elo.rating("Arsenal", pd.Timestamp("2021-05-27"))
    assert in_season == pytest.approx(1800.0)

    next_season = fb.elo.rating("Arsenal", pd.Timestamp("2021-08-14"))
    shrink = club_cfg["elo"]["inactivity_shrink"]
    expected = 1500.0 + (1800.0 - 1500.0) * (1 - shrink)
    assert next_season == pytest.approx(expected)
    assert next_season < in_season
