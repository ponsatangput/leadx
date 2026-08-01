# Temporal fraud rehearsal — train on the past, score the future (v0.2-D, 2026-07)

First measurement of **temporal generalisation** in this project. Every earlier
benchmark used random / feature-group splits inside one period; production
(weekly payment fraud) trains on the past and scores the future. Also the second
categorical dataset for `cat_engine="minhash"`.

## Design (pre-registered in `temporal_fraud_benchmark.py`)

IEEE-CIS fraud sorted by `TransactionDT`; 3 rolling origins (train 250k rows →
test the next 70k); PU with 50% of the train window's frauds confirmed as seeds
(rest unlabeled), 2 draws per fold; natural (~3.5%) and downsampled 1:1,000.

The counterfactual that isolates drift: each fold is run **both** as *temporal*
(train = earlier rows) and as *random* (same counts, same rows pool, shuffled).
`drift = (temporal − random)/random`, paired per (fold, draw).

## Result 1 — LeadX survives temporal shift, and drifts less than XGBoost

Paired drift advantage of LeadX over XGBoost-PU (percentage points; positive =
LeadX degrades less), 6 observations:

| ratio | PR-AUC | Recall@0.5% | Recall@0.1% |
|---|---|---|---|
| natural (3.5%) | **+5.1 pp (5/6)** | +3.1 pp (4/6) | +1.3 pp (4/6) |
| **1:1,000** | **+27.3 pp (5/6)** | **+26.7 pp (5/6)** | **+51.0 pp (5/6)** |

At 1:1,000 XGBoost loses about half its quality when the window moves
(PR −49.8%, Recall@0.5% −37.8%, Recall@0.1% −50.7% median), while LeadX's
recall is roughly flat (+9.4% / +9.1% median, i.e. no systematic degradation)
and its PR drop is smaller (−39.6%). **Drift robustness relative to trees grows
with imbalance severity** — the direction that matters for production.

## Result 2 — but LeadX is *not* uniquely drift-robust: LogReg is too

Paired drift advantage of LeadX over LogReg-PU (positive = LeadX better):

| ratio | PR-AUC | Recall@0.5% | Recall@0.1% |
|---|---|---|---|
| natural | −7.8 pp (2/6) | −6.1 pp (2/6) | −1.5 pp (2/6) |
| 1:1,000 | −75.2 pp (2/6) | −3.2 pp (2/6) | +7.2 pp (3/6) |

**Correction of an earlier claim.** A one-fold smoke run showed LogReg collapsing
(Recall@0.1% −60%) and that was written up here as a finding. It **did not
replicate**: with 6 observations LogReg is as drift-robust as LeadX or better.
The mechanistic story spun around it ("density ratio re-calibrates, linear
coefficients don't") is unsupported and withdrawn.

## Result 3 — absolute temporal accuracy (median over folds×draws)

| ratio | model | PR-AUC | Recall@0.5% |
|---|---|---|---|
| natural | xgboost_oracle (full labels) | .578 | .125 |
| natural | xgboost_pu_naive | **.531** | **.121** |
| natural | logreg_pu_naive | .300 | .112 |
| natural | leadx_pu | .301 | .108 |
| 1:1,000 | xgboost_oracle (full labels) | .052 | **.175** |
| 1:1,000 | **leadx_pu** | **.045** | .149 |
| 1:1,000 | xgboost_pu_naive | .043 | .149 |
| 1:1,000 | logreg_pu_naive | .018 | .167 |

Paired on the temporal split: at 1:1,000 LeadX beats XGBoost-PU on PR (+18%,
4/6) and LogReg on PR (+76%, 4/6), but **loses Recall@0.5%** to both (−12%,
−13%). At natural imbalance XGBoost dominates outright, as in every earlier
benchmark.

## Caveats (these matter)

- **Variance is large.** Per-fold PR drift at 1:1,000 ranges −55%…+48% (LeadX),
  −86%…+159% (LogReg), −91%…−39% (XGBoost). 6 observations, one dataset:
  replication, not a significance test. Only the LeadX-vs-XGBoost sign is
  consistent (5/6 across all three metrics).
- Absolute values at 1:1,000 are tiny (PR .02–.05), so ratios are unstable.
- Rolling windows overlap (stride 100k < train 250k), so folds are correlated.
- IEEE fraud spans ~6 months of one payment ecosystem; real drift in another
  system may be faster or structurally different.

## What this means for a real weekly-fraud pilot

Passed: LeadX does not collapse when scoring a future window, and it is the
most tree-robust option measured — at ultra imbalance XGBoost loses ~half its
quality across the time boundary while LeadX's recall holds.
Not established: superiority over a simple linear PU baseline (comparable drift,
better PR but worse broad recall), and anything beyond one dataset.

## Reproduce

```bash
cd sandbox/data_science/leadx
uv run --with xgboost python outputs/temporal_fraud_benchmark.py
#   → outputs/temporal_fraud_raw.csv (source of truth)
```
