# MI feature weighting — gate failed, but a useful rule fell out (v0.2-F, 2026-07)

Hypothesis: `feature_weighting="contrast"` only sees a *first moment*
(|mean(seeds) − mean(bg)|). Mutual information between a feature and "is a
confirmed seed" sees any distributional difference, is PU-native, and measures
the marginal relevance that LeadX's linear projections can actually use — so it
should resist noise dilution better.

Setting: IEEE fraud, temporal (train 250k → next 70k), 3 rolling folds × 2 PU
draws, 1:1,000, numeric features only (so categorical tables don't dilute the
weighting effect), with and without **+60 injected pure-noise features**.

## Verdict: the pre-registered gate FAILED

Gate: *`mi` beats `contrast` on Recall@0.5% (paired median > 0, ≥4/6 wins) in
the +noise setting.* Result: **median 0.0%, 2/6 → fail.** MI is not shipped.

Paired vs `contrast` (6 observations each):

| | PR-AUC | Recall@0.5% | Recall@0.1% |
|---|---|---|---|
| **no noise** — mi | +14.3% (5/6) | −2.9% (1/6) | +11.7% (4/6) |
| **no noise** — mi_mrmr | +11.7% (5/6) | 0.0% (2/6) | +3.6% (3/6) |
| **no noise** — none | **+22.8% (5/6)** | 0.0% (0/6) | +4.5% (3/6) |
| **+60 noise** — mi | −4.8% (2/6) | 0.0% (2/6) | **+17.1% (4/6)** |
| **+60 noise** — mi_mrmr | −3.5% (3/6) | −2.3% (2/6) | 0.0% (2/6) |
| **+60 noise** — none | −33.5% (1/6) | −15.0% (1/6) | −20.0% (0/6) |

The only consistent MI advantage is at the **tightest shortlist** (Recall@0.1%:
+11.7% and +17.1%, 4/6 in both settings) — a coherent pattern, but 4/6 is weak
evidence and it was not the pre-registered primary metric. Reporting it as a
lead, not a result.

The redundancy penalty (`mi_mrmr`) tracks plain `mi` and adds nothing
measurable. **Its first run was a bug, not a finding**: `np.corrcoef` on a
20k subsample hit columns that were constant *in that subsample*, produced NaNs,
and every weight became NaN (recall exactly 0.000). Fixed with `nan_to_num`; the
numbers above are post-fix.

## The finding worth keeping: weighting is conditional on feature hygiene

Compare `none` against `contrast` across the two settings:

| | PR-AUC | Recall@0.5% | Recall@0.1% |
|---|---|---|---|
| no noise: none vs contrast | **+22.8% (5/6)** | 0.0% | +4.5% |
| +60 noise: none vs contrast | **−33.5% (1/6)** | −15.0% | −20.0% (0/6) |

**When every feature carries signal, weighting costs you; when irrelevant
features are present, it pays — a lot.** Feature *selection* and feature
*weighting* are partial substitutes, so "select the top-N features **and** turn
on contrast" is not automatically right — it has to be measured on the actual
feature set.

## Caveats
- one dataset, 6 observations per cell, overlapping rolling windows
- injected noise is i.i.d. Gaussian — real irrelevant features are correlated
  with each other and with signal, which is a harder case for any weighting
- MI used fixed 8 quantile bins with a 2-fold conservative (min) estimate; other
  binning / estimators were not swept

## Reproduce
```bash
cd sandbox/data_science/leadx
uv run python outputs/mi_weighting_experiment.py
#   → outputs/mi_weighting_raw.csv (source of truth)
```
