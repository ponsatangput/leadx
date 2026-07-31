# LeadX

[![PyPI](https://img.shields.io/pypi/v/leadx.svg)](https://pypi.org/project/leadx/)
[![CI](https://github.com/ponsatangput/leadx/actions/workflows/test.yml/badge.svg)](https://github.com/ponsatangput/leadx/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
| ~20 confirmed positives, millions unlabeled, need a review queue | **Unconfirmed** — the original fixed-split +20.4% PR-AUC edge over linear PU shrank to **+1.8% [−8.5%, +9.3%]** against an envelope including Bagging-SVM across 5 new Criteo splits. Recall@0.5% was +3.7% [−1.0%, +11.9%] |
| Hundreds of confirmed positives | Probably not — linear/bagged PU baselines catch up (+1.7%, CI crosses zero at 100 seeds) |
| Many labelings over one population, task-head cost is the bottleneck | Yes (architecture) — frozen backbone gives ~ms task heads; accuracy parity with linear PU, not a win |
| Fully labeled extreme imbalance | Usually no at 1:338 and 1:1,000. **Preliminary crossover at 1:10,000**: RPForest paired PR +3.3% and Recall@0.5% +0.5 pp, both 2/3 splits with intervals crossing zero; tight Recall@0.01% still lost |
| Need a model that deploys as plain lookup tables | Yes — scoring is bucket lookup (rpforest adds a tree traversal) |

## Where it fits — proven vs. hypothesis

**Proven (raw-CSV backed):**
- **Build once, reuse cheaply.** Hash a 10M-row population once, then each new
  labeling is a ~0.04s task head — no retraining
  ([`SCALE_VALIDATION_REPORT.md`](outputs/SCALE_VALIDATION_REPORT.md)).
- **Deploys as plain lookup tables** (no model server); the lift score is
  imbalance-invariant by construction (the class prior never enters the formula).
- The reuse win is **speed/architecture, not accuracy** — a frozen backbone sits
  at *parity* with linear PU, not ahead of it.

**Hypothesis being tested in v0.2 (not a claim yet):**
- That LeadX gains an *accuracy* edge specifically at **extreme imbalance
  (~1:10,000) in PU mode** with many confirmed seeds — the regime it was
  originally built for (10–20M accounts, ~1–2k confirmed positives).
- Today's only evidence there is **preliminary and fully-supervised** (broad
  Recall@0.5% competitive on 2/3 Criteo splits, CI crossing zero; tight
  Recall@0.01% still lost). **The PU version of this regime is untested** —
  which is exactly what v0.2 measures before any positioning claim is made.

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
- **138 tests**, benchmark provenance (SHA-256 checksums, versions, git state)
  on every artifact.

## Benchmarks and evidence policy

All numbers live in [`outputs/`](outputs/README.md) with raw CSVs:

- [`CRITEO_MULTISPLIT_GATE_REPORT.md`](outputs/CRITEO_MULTISPLIT_GATE_REPORT.md)
  — newest Criteo gate: 5 fresh group splits, 50 paired observations, and a
  Bagging-SVM baseline; PR accuracy parity, with an unconfirmed recall signal.
- [`CRITEO_SUPERVISED_IMBALANCE_REPORT.md`](outputs/CRITEO_SUPERVISED_IMBALANCE_REPORT.md)
  — isolates imbalance from label scarcity by giving every model full labels;
  LeadX loses at the natural 1:338 ratio.
- [`CRITEO_ULTRA_IMBALANCE_REPORT.md`](outputs/CRITEO_ULTRA_IMBALANCE_REPORT.md)
  — 10M-row, fully-labeled stress test at 1:1,000 and 1:10,000; no edge at
  1:1,000, preliminary PR/broad-shortlist crossover at 1:10,000.
- [`CRITEO_UPLIFT_REPORT.md`](outputs/CRITEO_UPLIFT_REPORT.md) — original
  fixed-split Criteo study and 13.98M-row scale replication; retained as the
  exploratory predecessor to the multi-split gate.
- [`IMBALANCE_FOCUS_REPORT.md`](outputs/IMBALANCE_FOCUS_REPORT.md) — rarity
  stress tests on Home Credit / IEEE Fraud vs PU-naive baselines and a
  full-label oracle.
- [`SCALE_VALIDATION_REPORT.md`](outputs/SCALE_VALIDATION_REPORT.md) — 1M-10M
  row timing/memory decomposition.
- [`DESIGN_NOTES.md`](DESIGN_NOTES.md) — every idea that was measured and
  **rejected** (9 so far), with numbers.

Known limits, stated plainly: Criteo now has multi-split validation but no
independent-dataset confirmation; the Bagging-SVM paper structure is compared
with fixed C, but exact reference/modern PU implementations are not; fully
labeled tuned gradient boosting still wins when labels are abundant; rare-event
accuracy and frozen-backbone reuse are proven **separately**, not together.

## Roadmap (v0.2)

1. **Test the hypothesized niche** — extreme imbalance (~1:10,000) + PU +
   scale/reuse — in **PU mode**. Today's 1:10,000 evidence is fully-supervised
   and preliminary; the PU version matching the intended use case is untested.
   Plus pre-registered validation on >=2 independent datasets and
   modern/reference PU baselines (multi-split Criteo + Bagging-SVM is done; its
   20-seed PR gate failed).
2. Frozen-geometry accuracy: contrast weighting at scoring time
   (per-table task weights) and retrieve-then-rerank on backbone candidates.
3. English documentation.

## License & acknowledgments

MIT. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
