import numpy as np

from src.prediction.score_matrix import build_matrix, markets, outcome_probs, top_scorelines


def test_matrix_sums_to_one():
    m, tail = build_matrix(1.4, 1.1, max_goals=10)
    assert abs(m.sum() + tail - 1.0) < 1e-9
    assert tail < 1e-3


def test_matrix_grows_for_high_lambdas():
    m, tail = build_matrix(5.5, 5.0, max_goals=8, max_tail=1e-3)
    assert tail < 1e-3
    assert m.shape[0] > 9  # grid was enlarged


def test_outcome_probs_sum_to_one():
    m, _ = build_matrix(1.5, 1.2)
    p = outcome_probs(m)
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[0] > p[2]  # higher lambda favors team A


def test_markets_consistent():
    m, _ = build_matrix(1.3, 1.0)
    grid = m / m.sum()
    mk = markets(grid, 1.3, 1.0)
    assert abs(mk["over_2_5"] + mk["under_2_5"] - 1.0) < 1e-9
    assert mk["no_goal"] <= mk["under_2_5"]
    np.testing.assert_allclose(
        mk["team_a_scores_first"] + mk["team_b_scores_first"] + mk["no_goal"], 1.0, atol=1e-9
    )


def test_top_scorelines_sorted():
    m, _ = build_matrix(1.4, 1.1)
    tops = top_scorelines(m, 5)
    assert len(tops) == 5
    probs = [t["probability"] for t in tops]
    assert probs == sorted(probs, reverse=True)
