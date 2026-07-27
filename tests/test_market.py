"""Closing-odds de-vigging and the market baseline."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.market import (
    MarketBaseline,
    devig_proportional,
    devig_shin,
    implied_probabilities,
    market_probabilities,
    overround,
    priced_mask,
)

# A realistic EPL line: clear favourite, mid, and a near coin-flip.
ODDS = np.array([[1.50, 4.20, 6.50], [2.10, 3.40, 3.60], [4.50, 3.80, 1.80]])


def _df(odds, source="pinnacle"):
    return pd.DataFrame({
        "odds_close_a": odds[:, 0],
        "odds_close_draw": odds[:, 1],
        "odds_close_b": odds[:, 2],
        "odds_source": [source] * len(odds),
    })


def test_raw_implied_probabilities_exceed_one():
    """Published odds carry the bookmaker's margin; they are not
    probabilities until it is removed."""
    assert (overround(ODDS) > 1.0).all()
    assert implied_probabilities(np.array([[2.0, 4.0, 4.0]])).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("devig", [devig_proportional, devig_shin])
def test_devigged_probabilities_sum_to_one(devig):
    p = devig(ODDS)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all() and (p < 1).all()


def test_shin_shifts_probability_toward_favourites():
    """Bookmakers load proportionally more margin onto longshots, so
    dividing uniformly leaves longshots too high and favourites too low.
    Shin corrects in that direction."""
    pp, ps = devig_proportional(ODDS), devig_shin(ODDS)
    for i in range(len(ODDS)):
        fav = int(np.argmax(pp[i]))
        dog = int(np.argmin(pp[i]))
        assert ps[i][fav] > pp[i][fav], f"row {i}: Shin did not raise the favourite"
        assert ps[i][dog] < pp[i][dog], f"row {i}: Shin did not lower the longshot"


def test_fair_book_is_unchanged_by_either_method():
    """With no margin there is nothing to remove."""
    fair = np.array([[2.0, 4.0, 4.0]])
    assert np.allclose(devig_proportional(fair), [[0.5, 0.25, 0.25]])
    assert np.allclose(devig_shin(fair), [[0.5, 0.25, 0.25]], atol=1e-6)


def test_unpriced_rows_yield_nan_not_a_guess():
    df = _df(ODDS)
    df.loc[1, "odds_close_draw"] = np.nan
    p = market_probabilities(df)
    assert np.isfinite(p[0]).all()
    assert np.isnan(p[1]).all()
    assert np.isfinite(p[2]).all()


def test_priced_mask_can_restrict_to_one_bookmaker():
    """Pinnacle prices EPL at 2.40% overround and Bet365 at 5.65%; pooling
    a sharp and a soft market would bias the reference."""
    df = pd.concat([_df(ODDS, "pinnacle"), _df(ODDS, "bet365")], ignore_index=True)
    assert priced_mask(df).sum() == 6
    assert priced_mask(df, "pinnacle").sum() == 3
    assert priced_mask(df, "bet365").sum() == 3


def test_market_baseline_needs_no_training():
    m = MarketBaseline("proportional").fit(_df(ODDS), np.array([0, 1, 2]))
    p = m.predict_proba(_df(ODDS))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_unknown_devig_method_raises():
    with pytest.raises(ValueError, match="unknown devig method"):
        market_probabilities(_df(ODDS), "magic")
