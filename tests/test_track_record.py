"""Grading the append-only prediction log: 90-minute convention, team
orientation, append-only enforcement, and README splicing."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics
from src.evaluation.track_record import (
    END_MARK,
    PREDICTION_COLUMNS,
    START_MARK,
    current_section,
    find_result,
    grade,
    load_log,
    render_section,
    section_equivalent,
    splice_readme,
    summarize,
    write_graded_log,
)


def _log_row(match, kickoff, pa, pd_, pb, score="1-1", xg_a=1.3, xg_b=1.1, mode="rolling"):
    return {
        "match": match,
        "kickoff": kickoff,
        "prediction_created_at": "2020-01-01T00:00:00Z",
        "prediction_cutoff": kickoff,
        "predicted_team_a_win": pa,
        "predicted_draw": pd_,
        "predicted_team_b_win": pb,
        "predicted_score": score,
        "expected_goals_a": xg_a,
        "expected_goals_b": xg_b,
        "actual_score": "",
        "actual_result": "",
        "brier_score": "",
        "log_loss": "",
        "goal_absolute_error": "",
        "correct_result": "",
        "correct_exact_score": "",
        "model_version": "0.2.0",
        "feature_version": "fv1-60",
        "latest_match_used": "2019-12-31",
        "mode": mode,
    }


@pytest.fixture
def toy_log():
    return pd.DataFrame(
        [
            # Knockout decided in extra time (1-1 at 90, 2-1 after ET).
            _log_row("Alpha v Gamma", "2020-05-01T20:00:00Z", 0.55, 0.25, 0.20),
            # Shootout match (0-0 at 90 and ET, Beta won on penalties).
            _log_row("Beta v Gamma", "2020-06-01T20:00:00Z", 0.40, 0.30, 0.30, score="0-0"),
            # Logged with the opposite orientation to the canonical table.
            _log_row("Gamma v Alpha", "2020-03-01T20:00:00Z", 0.60, 0.20, 0.20, score="3-0"),
            # Fixture that was never played.
            _log_row("Alpha v Delta", "2020-07-01T20:00:00Z", 0.50, 0.25, 0.25),
        ]
    )


def test_extra_time_match_is_graded_as_a_draw(toy_log, toy_matches):
    """Alpha beat Gamma 2-1 in extra time but the match was 1-1 at 90
    minutes; every logged probability refers to 90 minutes only."""
    g = grade(toy_log, toy_matches)
    row = g[g["match"] == "Alpha v Gamma"].iloc[0]
    assert row["graded"]
    assert row["actual_result"] == "draw"
    assert row["actual_score"] == "1-1"
    assert row["went_to_extra_time"]
    # Model favoured Alpha, so the result is wrong even though Alpha advanced.
    assert row["correct_result"] is False
    assert row["correct_exact_score"] is True


def test_shootout_goals_never_count_as_normal_goals(toy_log, toy_matches):
    g = grade(toy_log, toy_matches)
    row = g[g["match"] == "Beta v Gamma"].iloc[0]
    assert row["actual_score"] == "0-0"
    assert row["actual_result"] == "draw"
    assert row["went_to_penalties"]


def test_reversed_orientation_flips_the_actual_score(toy_log, toy_matches):
    """Canonical table has Alpha 0-3 Gamma; the log recorded 'Gamma v Alpha',
    so the actual score must be flipped to match the logged probabilities."""
    g = grade(toy_log, toy_matches)
    row = g[g["match"] == "Gamma v Alpha"].iloc[0]
    assert row["actual_score"] == "3-0"
    assert row["actual_result"] == "team_a"
    assert row["correct_result"] is True
    assert row["correct_exact_score"] is True


def test_unplayed_fixture_stays_ungraded(toy_log, toy_matches):
    g = grade(toy_log, toy_matches)
    row = g[g["match"] == "Alpha v Delta"].iloc[0]
    assert not row["graded"]
    assert pd.isna(row["actual_score"]) or row["actual_score"] == ""
    assert summarize(g)["n_ungraded"] == 1


def test_no_match_within_date_tolerance(toy_matches):
    """A pair that exists but on a far-away date must not be force-matched."""
    assert find_result("Alpha v Gamma", "2021-05-01T20:00:00Z", toy_matches) is None


def test_per_match_metrics_agree_with_metrics_module(toy_log, toy_matches):
    g = grade(toy_log, toy_matches)
    graded = g[g["graded"]]
    p = graded[["predicted_team_a_win", "predicted_draw", "predicted_team_b_win"]].to_numpy(float)
    y = np.array([{"team_a": 0, "draw": 1, "team_b": 2}[r] for r in graded["actual_result"]])
    assert graded["log_loss"].mean() == pytest.approx(metrics.log_loss(y, p), abs=1e-6)
    assert graded["brier_score"].mean() == pytest.approx(metrics.brier_multiclass(y, p), abs=1e-6)


def test_summary_counts(toy_log, toy_matches):
    s = summarize(grade(toy_log, toy_matches))
    assert s["n"] == 3
    assert s["correct_results"] == 1        # only the reversed-orientation row
    assert s["exact_scores"] == 3
    assert s["uniform_log_loss"] == pytest.approx(np.log(3), abs=1e-4)


def test_write_refuses_to_change_predictions(tmp_path, toy_log, toy_matches):
    path = tmp_path / "log.csv"
    toy_log.to_csv(path, index=False)
    g = grade(toy_log, toy_matches)
    g.loc[0, "predicted_team_a_win"] = 0.99          # simulate a bug
    with pytest.raises(ValueError, match="append-only"):
        write_graded_log(g, path)


def test_write_back_preserves_predictions_and_is_idempotent(tmp_path, toy_log, toy_matches):
    path = tmp_path / "log.csv"
    toy_log.to_csv(path, index=False)
    before = pd.read_csv(path)[PREDICTION_COLUMNS]

    write_graded_log(grade(load_log(path), toy_matches), path)
    once = pd.read_csv(path)
    pd.testing.assert_frame_equal(once[PREDICTION_COLUMNS], before)
    assert once.loc[0, "actual_score"] == "1-1"

    # Re-grading an already-graded log must not drift.
    write_graded_log(grade(load_log(path), toy_matches), path)
    twice = pd.read_csv(path)
    pd.testing.assert_frame_equal(twice, once)


def test_column_order_survives_grading(tmp_path, toy_log, toy_matches):
    """predict.py appends with a fixed DictWriter field order; reordering the
    header on write would misalign every future appended prediction."""
    from predict import LOG_FIELDS

    path = tmp_path / "log.csv"
    toy_log.to_csv(path, index=False)
    before = list(pd.read_csv(path).columns)
    assert before == LOG_FIELDS, "fixture must mirror predict.py's schema"

    write_graded_log(grade(load_log(path), toy_matches), path)
    assert list(pd.read_csv(path).columns) == LOG_FIELDS


def test_render_and_splice_is_idempotent(toy_log, toy_matches):
    section = render_section(grade(toy_log, toy_matches))
    assert START_MARK in section and END_MARK in section
    assert "Alpha v Gamma" in section
    assert "Alpha v Delta" not in section        # ungraded rows are not shown

    readme = "# Title\n\nintro\n\n## Contents\n\nbody\n"
    once = splice_readme(readme, section)
    assert splice_readme(once, section) == once
    assert once.count(START_MARK) == 1


def test_render_limit(toy_log, toy_matches):
    section = render_section(grade(toy_log, toy_matches), limit=1)
    assert section.count("| rolling |") == 1


def test_empty_log_renders_placeholder(toy_log, toy_matches):
    g = grade(toy_log.iloc[[3]], toy_matches)      # only the unplayed fixture
    section = render_section(g)
    assert "No logged prediction has a completed result yet" in section
    assert summarize(g)["n"] == 0


def test_section_equivalent_ignores_timestamp(toy_log, toy_matches):
    g = grade(toy_log, toy_matches)
    a = current_section(render_section(g))
    b = current_section(render_section(g).replace("2026-", "2027-"))
    assert section_equivalent(a, b)
