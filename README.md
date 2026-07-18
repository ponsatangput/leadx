# LeadX

**Rare-event retrieval from a handful of confirmed positives** — an LSH /
random-projection-tree ensemble with density-ratio scoring and a reusable
hashing backbone. Pure batch operations (NumPy + Polars); validated up to
10M rows on a laptop.

> **Status: v0.1 research preview.** This library ships with an unusually
> adversarial benchmark suite: every performance claim links to a raw CSV
> artifact, paired per-draw analysis is used instead of ratio-of-medians, and
> the docs list what has **not** been proven yet. Headline accuracy results
> are labeled *preliminary* until pre-registered multi-dataset validation
> lands (planned for v0.2). Read [`outputs/`](outputs/README.md) before
> quoting any number.

Full documentation is currently in Thai — see **[README.th.md](README.th.md)**
and **[DESIGN_NOTES.md](DESIGN_NOTES.md)** (design history including every
measured-and-rejected idea). English translation is in progress.

## The problem it targets

You have millions of unlabeled rows and only a few *confirmed* positives
(fraud cases an analyst verified, customers who actually converted, leads that
closed). You want a ranked review queue. Standard supervised training does not
apply directly without labeled negatives; PU conversions of strong learners
degrade sharply when confirmed seeds number in the tens.

LeadX scores each hash bucket by a smoothed density ratio
`lift = P(bucket | seeds) / P(bucket | population)` averaged across K tables —
an approximate soft-kNN that is naturally positive-unlabeled and
imbalance-invariant (the class prior never enters the formula).

## Install

```bash
pip install leadx
```

## Quickstart — PU retrieval

```python
import polars as pl
from leadx import LeadX

# Flagship config (opt-in, task-specific geometry — see docs for trade-offs)
clf = LeadX(mode="pu", tune=False, n_bits=10, n_tables=80,
            score_type="lift", agg="log", feature_weighting="contrast",
            engine="rpforest", hier_depths="auto")
clf.fit(X_confirmed_positives, background=X_population)
scores = clf.score_samples(X_population)   # ranked review queue
```

## Quickstart — build once, many tasks

```python
from leadx import LeadXBackbone

bb = LeadXBackbone(n_bits=16, n_tables=40).fit(X_population)  # hash once
task = bb.fit_task(mode="pu", seed_rows=confirmed_idx, background="cached",
                   score_type="lift", agg="log", hier_depths="auto")
ranking = task.score_cached()          # score the fitted population, no re-hash

codes = bb.transform(X_new_batch)      # hash a new batch once...
s1 = task.score_codes(codes)           # ...reuse the codes across many tasks
```

Measured at 10M rows x 100 features (24GB laptop): backbone 51s once,
task-head construction ~0.04s per task, full-population scoring ~31s per task.
Task-head cost is the reuse win; end-to-end totals depend on scoring volume —
see [`outputs/SCALE_VALIDATION_REPORT.md`](outputs/SCALE_VALIDATION_REPORT.md).

## When to use it (honest version)

| Situation | Use LeadX? |
|---|---|
| ~20 confirmed positives, millions unlabeled, need a review queue | **Preliminary yes** — on Criteo conversion (0.29% observed rate, 20 seeds) task-specific LeadX beat a linear-PU envelope by +20.4% paired PR-AUC, 8/10 draws (p=.055, one split — exploratory) |
| Hundreds of confirmed positives | Probably not — linear/bagged PU baselines catch up (+1.7%, CI crosses zero at 100 seeds) |
| Many labelings over one population, task-head cost is the bottleneck | Yes (architecture) — frozen backbone gives ~ms task heads; accuracy parity with linear PU, not a win |
| Fully labeled two-class data | No — use gradient boosting |
| Need a model that deploys as plain lookup tables | Yes — scoring is bucket lookup (rpforest adds a tree traversal) |

## What's inside

- **Two code-generation engines**: global hyperplanes with tie-aware median
  thresholds (SimHash-style, single matmul) and `engine="rpforest"` — a
  balanced random-projection-tree forest (per-node random direction + local
  median split; approximately equal-mass buckets on the fit sample).
- **Lift scoring** with hierarchical shrinkage (`hier_depths`): sparse deep
  buckets inherit evidence from their prefix ancestors — helps most when
  seeds number in the tens.
- **Reusable backbone**: population hashed once; per-task cost is a group-by.
  Background statistics are memoized per configuration.
- **Auto-tuner**: one hash at max resolution evaluates the whole
  (bits x tables x scoring) grid via prefix masking, cross-fitted to avoid
  OOF leakage; evaluates both engines and picks per dataset.
- **136 tests**, benchmark provenance (SHA-256 checksums, versions, git state)
  on every artifact.

## Benchmarks and evidence policy

All numbers live in [`outputs/`](outputs/README.md) with raw CSVs:

- [`CRITEO_UPLIFT_REPORT.md`](outputs/CRITEO_UPLIFT_REPORT.md) — external
  check on Criteo Uplift v2.1 (13.98M rows), few-seed retrieval, paired
  analysis, includes the negative results (frozen backbone is not an accuracy
  moat; the edge fades at 100 seeds).
- [`IMBALANCE_FOCUS_REPORT.md`](outputs/IMBALANCE_FOCUS_REPORT.md) — rarity
  stress tests on Home Credit / IEEE Fraud vs PU-naive baselines and a
  full-label oracle.
- [`SCALE_VALIDATION_REPORT.md`](outputs/SCALE_VALIDATION_REPORT.md) — 1M-10M
  row timing/memory decomposition.
- [`DESIGN_NOTES.md`](DESIGN_NOTES.md) — every idea that was measured and
  **rejected** (9 so far), with numbers.

Known limits, stated plainly: single fixed splits so far; strong PU baselines
(Elkan-Noto, original Bagging-PU) not yet compared; fully-labeled tuned
gradient boosting still wins when labels are abundant; rare-event accuracy and
frozen-backbone reuse are proven **separately**, not yet together.

## Roadmap (v0.2)

1. Pre-registered validation: >=2 independent datasets, multiple splits,
   strong PU baselines.
2. Frozen-geometry accuracy: contrast weighting at scoring time
   (per-table task weights) and retrieve-then-rerank on backbone candidates.
3. English documentation.

## License & acknowledgments

MIT. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
