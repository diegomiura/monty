# Premier League — chronological validation results

Generated 2026-07-26. Reproduce with:

```bash
python backtest_clubs.py
```

**Data:** 12,704 Premier League matches, 1993-08-14 → 2026-05-24, from
football-data.co.uk (see [CLUB_DATA_AUDIT.md](CLUB_DATA_AUDIT.md)).
**Validation:** expanding window by season — train on everything before
season S, predict S, step forward. 26 folds, 9,880 matches evaluated
(2000-01 onward, where shot data begins). No random splitting, no future
data, no tuning against the evaluation seasons.

## All 9,880 validated matches

| model | log loss | Brier | RPS | accuracy | ECE |
|:--|--:|--:|--:|--:|--:|
| **logistic** | **0.9788** | 0.5824 | 0.1986 | 52.8% | 0.019 |
| elo | 0.9845 | 0.5863 | 0.2005 | 52.8% | 0.016 |
| higher_elo | 1.0188 | 0.6101 | 0.2116 | 52.5% | 0.031 |
| gradient_boosting | 1.0267 | 0.6059 | 0.2066 | 51.4% | 0.055 |
| frequency | 1.0651 | 0.6434 | 0.2283 | 45.7% | 0.004 |

## Against the closing line

Restricted to the 5,150 matches priced by Pinnacle — one consistent market
rather than a blend of a sharp book and a soft one.

| model | log loss | Brier | RPS | accuracy | ECE |
|:--|--:|--:|--:|--:|--:|
| **market (closing line)** | **0.9523** | 0.5633 | 0.1930 | 55.2% | 0.012 |
| logistic | 0.9704 | 0.5755 | 0.1984 | 53.9% | 0.022 |
| gradient_boosting | 0.9786 | 0.5803 | 0.2002 | 53.1% | 0.020 |
| elo | 0.9787 | 0.5813 | 0.2009 | 54.0% | 0.022 |
| higher_elo | 1.0135 | 0.6060 | 0.2123 | 53.9% | 0.035 |
| frequency | 1.0690 | 0.6464 | 0.2325 | 44.7% | 0.014 |

**The model loses to the market by 0.018 log loss.** That is the expected
and correct outcome: the closing line aggregates team news, lineups, and
money that a goals-and-shots model cannot see. A 1.9% gap on a
free-data model is a good result — and the gap is the honest headline, not
the fact that logistic beats four other things we also built.

Calibration is the more encouraging number: ECE 0.022 against the market's
0.012, on a model that never sees a price.

## Honest observations

**Gradient boosting is worse than plain Elo overall** (1.0267 vs 0.9845
across all matches). The per-fold numbers show why: it is badly
mis-calibrated in the early seasons — 1.17 to 1.25 log loss in 2000-05,
where training data is thin and shot features are largely absent — and only
converges to the logistic model from about 2006-07. Restricted to the
priced subset (which is 2015-16 onward, so all of it is the converged era)
it draws level with Elo. The decision rules say to report this rather than
quietly drop the fold range that makes it look bad.

**No model beats "higher Elo wins" on accuracy by much** (52.8% vs 52.5%).
Accuracy is close to useless here; the probabilistic metrics separate the
models cleanly and the accuracy column does not.

**The de-vigging choice does not matter at this overround.** Proportional
and Shin give a market log loss of 0.9523 and 0.9522 respectively — a
difference of 0.0001. Shin is slightly better calibrated (ECE 0.0103 vs
0.0119). This was expected to matter more; on Pinnacle's tight 2.40% margin
there is simply not enough vig for the two assumptions to diverge. It would
matter more on a soft book — Bet365's EPL overround is 5.65%.

## What has not been done

* **No ensemble, no calibration layer.** The international pipeline has
  both; neither has been fitted for clubs yet. Both should narrow the gap
  to the market.
* **No Poisson/scoreline model**, so no expected goals, scoreline
  probabilities, over/under or BTTS for clubs yet — only the 1X2 outcome.
* **Hyperparameters are inherited, not tuned.** `config/clubs.yaml` carries
  documented starting values. The tuning seasons (2000-01 to 2004-05) are
  reserved for that and have not been used.
* **Odds remain a benchmark only.** `market.use_market_features` is off,
  and turning it on would need a validation gate of its own.
