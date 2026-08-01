# Complementarity — does LeadX catch fraud XGBoost misses? (v0.2-E, 2026-07)

For a team already running XGBoost the decision is not "is LeadX better" but
**"does adding LeadX catch cases XGBoost misses"**. Measured on the temporal
setup (IEEE fraud, train past → score future, 3 rolling folds × 2 PU draws).

Two incumbents are compared:
- `xgbpu` — XGBoost trained on **confirmed** frauds vs everything-else-as-negative.
  **This is the realistic incumbent** (what a fraud team actually has).
- `xgboracle` — XGBoost with *all* true labels including the frauds nobody
  reported. Not obtainable in production; upper reference only.

## Overlap of the review queue is genuinely low

| ratio | vs | overlap@0.1% | overlap@0.5% |
|---|---|---|---|
| natural 3.5% | xgbpu | 0.22 | 0.46 |
| natural 3.5% | xgboracle | 0.25 | 0.57 |
| **1:1,000** | **xgbpu** | **0.11** | **0.12** |
| 1:1,000 | xgboracle | 0.16 | 0.15 |

At ultra imbalance the two models agree on barely **one in eight** of the flagged
transactions. They are ranking on genuinely different evidence (density around
confirmed frauds vs learned decision boundary).

## What that buys, at a *fixed* review budget

Rank-average ensemble vs the incumbent alone (paired, 6 obs):

| ratio | vs | budget 0.1% | budget 0.5% |
|---|---|---|---|
| natural | xgbpu | +3.0% (6/6) | −0.9% (2/6) |
| natural | xgboracle | +1.4% (4/6) | −2.3% (0/6) |
| **1:1,000** | **xgbpu** | **+21.1% (5/6)** | +2.2% (3/6) |
| 1:1,000 | xgboracle | −16.2% (1/6) | −5.0% (2/6) |

**The one cell that clearly pays: ultra imbalance + tight budget + the realistic
incumbent** — ensemble recall@0.1% is +21% over XGBoost-PU in 5 of 6 folds. That
is precisely the production shape (rare fraud, small review team, labels only
for confirmed cases).

Everywhere else a naive rank average is neutral or harmful: against the
full-label oracle it *dilutes* the stronger model (−16%). Low overlap is a
necessary condition for ensembling, not a sufficient one.

## If you can afford more review capacity

Reviewing both lists (union at 2× budget) is where the disagreement shows most:

| ratio | vs | single (incumbent) | union@2k | gain |
|---|---|---|---|---|
| natural | xgbpu | 0.121 | 0.171 | **+41%** |
| 1:1,000 | xgbpu | 0.149 | 0.237 | **+59%** |

At natural imbalance and 0.5% budget, **146 of LeadX's catches per fold are
frauds XGBoost-PU's list misses** (and 171 the other way) — the models are close
to interchangeable in *volume* but far apart in *which* cases they find.

## Caveats

- At 1:1,000 the absolute counts are tiny (3–5 unique catches per fold) — the
  +21% is a real pattern (5/6) but rests on few events.
- 6 observations, one dataset, overlapping rolling windows; replication, not a
  significance test.
- Only one combination rule tested (rank average). The union@2k gains suggest a
  smarter combiner (budget split, cascade, stacked ranker) could capture more at
  fixed capacity — untested.

## Practical reading

1. LeadX and XGBoost find **different frauds** — the complementarity premise holds.
2. Against a realistic PU-trained XGBoost at ultra imbalance and a tight queue,
   a plain rank-average ensemble already adds ~20% recall.
3. A fixed-budget ensemble is *not* automatically better elsewhere; when the
   incumbent is much stronger, averaging hurts. Combine deliberately, don't just
   add scores.

## Reproduce
```bash
cd sandbox/data_science/leadx
uv run --with xgboost,scipy python outputs/complementarity_benchmark.py
#   → outputs/complementarity_raw.csv (source of truth)
```
