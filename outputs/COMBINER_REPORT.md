# Combiner search — spending one review budget across two models (v0.2-G, 2026-07)

Complementarity was established (overlap 0.11–0.12 at ultra imbalance; union at
2× capacity +41–59%). This asks the operational question: **at the same review
budget, which rule converts that disagreement into catches?**

Setting identical to the complementarity run: IEEE fraud, temporal (train 250k →
next 70k), 3 rolling folds × 2 PU draws, natural (~3.5%) and 1:1,000. Incumbent
`xgb_only` = XGBoost trained on confirmed frauds only (what a fraud team has).
Primary metric Recall@0.1%, pre-registered gate: median > 0 **and ≥5/6 wins** at
1:1,000, and no regression worse than 5% at natural.

## Result — 1:1,000, Recall@0.1% (primary), paired vs `xgb_only`

| combiner | recall | vs xgb_only | gate |
|---|---|---|---|
| **cascade_leadx** (LeadX shortlists 10k → XGBoost re-ranks) | .0944 | **+22.5% (6/6)** | ✅ **pass** |
| cascade_xgb (XGBoost shortlists → LeadX re-ranks) | .0965 | +29.2% (5/6) | ❌ regresses −5.7% at natural |
| rank_avg | .0965 | +21.1% (5/6) | ✅ pass |
| wavg_0.3 (0.7·xgb + 0.3·leadx) | .0944 | +19.1% (5/6) | ✅ pass |
| rrf | .0965 | +23.6% (4/6) | ❌ wins |
| split_0.25 / split_0.5 | .0877 | +12.5% (3/6) | ❌ wins |
| leadx_only | .0877 | +18.2% (3/6) | ❌ wins |

At the wider budget (Recall@0.5%, 1:1,000) the same direction holds — cascade_xgb
+19.6% (5/6), split_0.25 +26.3% (4/6), rrf +23.6% (4/6).

At **natural imbalance nothing meaningful is available**: the best is +3.7% at
Recall@0.1% and several combiners go negative at Recall@0.5%. The combiner value
is ultra-imbalance-specific, consistent with every other result in this project.

## Two things this overturned

1. **My prediction was wrong.** I argued `split` (reserve α·k slots for LeadX)
   should win, reasoning that LeadX's unique catches were being "pushed out of
   the queue". It was the *worst* fusion variant at the primary metric
   (+12.5%, 3/6). Reserving capacity blindly wastes slots on LeadX's weak tail;
   the disagreement is not uniform across the ranking.
2. **`rank_avg` was not useless.** The complementarity report framed it as
   capturing "almost none" of the gap; at 1:1,000 / Recall@0.1% it reproduces
   +21.1% (5/6) exactly — the earlier framing over-generalised from the wider
   budget and the natural-imbalance cells.

## Why `cascade_leadx` makes sense — and why it matters operationally

LeadX casts the **broad net** (density around confirmed frauds — good recall,
blunt at the very top); XGBoost is **sharp at ordering** but misses cases its
boundary never learned. Letting LeadX choose the candidate pool and XGBoost sort
it plays to both. It is also the only combiner that won **6/6**.

Operationally this is the cheapest arrangement at scale: LeadX scores the whole
population fast (≈320k rows/s measured), and XGBoost only ever runs on the
shortlist (10× the review budget), not on every row. For a 500M-transaction
week that is the difference between scoring 500M rows with a tree ensemble and
scoring ~a few hundred thousand.

```python
# cascade_leadx, fixed review budget k
cand = np.argsort(-leadx_scores)[:10 * k]      # LeadX shortlists (whole population)
queue = cand[np.argsort(-xgb_scores[cand])[:k]] # XGBoost re-ranks the shortlist
```

## Caveats
- 6 observations per cell, one dataset, overlapping rolling windows; the gate is
  a replication bar (≥5/6), not a significance test
- shortlist depth fixed at 10× budget — not swept
- `cascade_xgb` scored higher at the primary metric but failed the
  natural-imbalance guard; with more data it may be the better rule
- absolute recall stays modest (~.09 at 0.1% budget): this improves the queue,
  it does not solve fraud detection

## Reproduce
```bash
cd sandbox/data_science/leadx
uv run --with xgboost,scipy python outputs/combiner_benchmark.py
#   → outputs/combiner_raw.csv (source of truth)
```
