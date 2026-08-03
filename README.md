# LeadX

[![PyPI](https://img.shields.io/pypi/v/leadx.svg)](https://pypi.org/project/leadx/)
[![CI](https://github.com/ponsatangput/leadx/actions/workflows/test.yml/badge.svg)](https://github.com/ponsatangput/leadx/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Rare-event retrieval from a handful of confirmed positives** — an LSH /
random-projection-tree ensemble with density-ratio scoring and a reusable
hashing backbone. Pure batch operations (NumPy + Polars); validated up to
10M rows on a laptop.

> **Status: v0.4 research preview.** This library ships with an unusually
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

## Fraud / extreme-imbalance quickstart

The regime this is built for: millions of rows, a few thousand **confirmed**
frauds, everything else unlabeled, and a review team that can only look at the
top fraction of a queue.

```python
from leadx import LeadXClassifier

# all-numeric feature table (categoricals already encoded upstream) — the
# common case. rare_pu falls back to cat_engine="onehot" and tells you so.
m = LeadXClassifier(profile="rare_pu")
m.fit(X_train, y_train,                    # y = 1 confirmed fraud, 0 unlabeled
      sample_weight=w_train)               # optional: weight hard cases higher
queue = m.review_queue(X_next_week, frac=0.001)     # today's investigation list
```

If some columns really are high-cardinality IDs (account, merchant, device),
pass a dataframe and name them — MinHash gives Jaccard locality over them
instead of one-hot blow-up:

```python
m = LeadXClassifier(profile="rare_pu",
                    categorical_columns=["account_id", "merchant_id", "device"])
m.fit(df_train, y_train)
```

Select the configuration on a **later time window**, not by random CV:

```python
m.fit(X_week1, y_week1, eval_set=(X_week2, y_week2),
      param_grid={"n_bits": [8, 10, 12], "engine": ["hyperplane", "rpforest"]},
      eval_metric="recall@1000")     # also: recall@0.5%, precision@K, pr_auc, f1
```

Already running XGBoost? The measured best use is a **cascade**, not a
replacement — LeadX shortlists the whole population cheaply, XGBoost re-ranks
the shortlist (+22.5% Recall@0.1%, 6/6 folds at 1:1,000; see
[`COMBINER_REPORT.md`](outputs/COMBINER_REPORT.md)):

```python
cand  = m.review_queue(X_next, k=10 * budget)          # LeadX over everything
queue = cand[np.argsort(-xgb.predict_proba(X_next[cand])[:, 1])[:budget]]
```

**Three things to know**

- `predict_proba` in PU mode is a **ranking score rescaled to [0, 1]**, not a
  calibrated probability — pass `prior=<prevalence>` for a SCAR-style
  `clip(prior * lift, 0, 1)`, or use `decision_function` (raw lift) for
  cascades and ensembles.
- `y == 0` means *unlabeled*, not *negative*.
- `sample_weight` weights the confirmed positives (the numerator); the
  background population still counts one row each.

## Quickstart — XGBoost-shaped API (recommended)

```python
from leadx import LeadXClassifier

m = LeadXClassifier(profile="rare_pu")     # measured flagship config
m.fit(X, y)                                # y = 1 confirmed positive, 0 UNLABELED
                                           # X: pandas / polars / numpy / pyarrow
scores = m.predict_proba(X_next_week)[:, 1]
queue  = m.review_queue(X_next_week, frac=0.001)   # indices to investigate
```

`y == 0` means **unlabeled, not negative** — the whole of `X` becomes the
background population (the positive-unlabeled setting fraud and lead problems
actually live in). Pass `task="two_class"` when zeros are confirmed negatives.

For a *temporal* problem, select the configuration on a later window rather than
by random cross-validation:

```python
m.fit(X_week1, y_week1, eval_set=(X_week2, y_week2),
      param_grid={"n_bits": [8, 10, 12], "engine": ["hyperplane", "rpforest"]},
      eval_metric="recall@0.5%")
```

Other conveniences: `predict`, `decision_function` (raw lift — use this for
cascades/ensembles), `feature_importances_`, `save_model`/`load_model`,
`get_params`/`set_params` (works inside scikit-learn pipelines), and
`background_policy="frozen"|"refresh"` to choose whether the density denominator
is fixed at fit time or re-estimated from each scored batch.

## Quickstart — core API (explicit PU contract)

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
| **Ultra-extreme imbalance (~1:10,000) in PU mode** | ✅ **Measured (Criteo 10M, 3 splits×3 draws)** — tree methods collapse here: LeadX beats **XGBoost-PU +325% PR / +208% Recall@0.5% (9/9)**, and beats XGBoost with *full labels* +240% PR (9/9). But **ties a strong linear-PU baseline** (LogReg slightly better on PR) — LeadX is top-tier, not uniquely best. Edge is ultra-specific (gone by 1:1,000). One dataset |
| Fully labeled extreme imbalance (supervised) | Usually no at 1:338 and 1:1,000; preliminary crossover at 1:10,000 (see reports) |
| Scoring a *future* window (weekly fraud queues) | ⚠️ **Measured once** — LeadX doesn't collapse across a time boundary and degrades less than XGBoost at 1:1,000 (+27 pp, 5/6 folds), but a linear PU baseline is equally drift-robust; one dataset, high variance ([`TEMPORAL_FRAUD_REPORT.md`](outputs/TEMPORAL_FRAUD_REPORT.md)) |
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

**Measured in v0.2 — the ultra-imbalance PU regime:**
- At **~1:10,000 in PU mode** (Criteo 10M, 3 splits × 3 draws), gradient-boosted
  **trees collapse**: LeadX beats XGBoost-PU **+325% PR-AUC / +208% Recall@0.5%
  (9/9)**, and beats XGBoost with *full labels* +240% PR (9/9).
- Honest limit: LeadX **ties a strong linear-PU baseline** (LogReg is slightly
  better on PR) — so the differentiator here is the reuse/scale architecture on
  top of top-tier accuracy, not accuracy alone. Edge is ultra-specific (gone by
  1:1,000). One dataset ([`CRITEO_PU_ULTRA_REPORT.md`](outputs/CRITEO_PU_ULTRA_REPORT.md)).

## What's inside

- **Two code-generation engines**: global hyperplanes with tie-aware median
  thresholds (SimHash-style, single matmul) and `engine="rpforest"` — a
  balanced random-projection-tree forest (per-node random direction + local
  median split; approximately equal-mass buckets on the fit sample). Numeric
  and categorical are hashed by whichever engine fits and their bucket codes are
  concatenated — the scorer is engine-agnostic, so features of any type compose.
- **Categorical handling** (`cat_engine="minhash"`): MinHash gives Jaccard
  locality over `{column=value}` tokens — arbitrary cardinality, no one-hot
  blow-up, ID-like integers treated as tokens instead of z-scored. On IEEE
  fraud (PU, 1:1,000, 5 paired draws) it beat the one-hot encoder **+56% PR-AUC
  / +37% Recall@0.5%, 5/5 draws** — and one-hot was *hurting* vs numeric-only
  (0/5). Preliminary (one dataset). Default `"onehot"` is unchanged.
- **Lift scoring** with hierarchical shrinkage (`hier_depths`): sparse deep
  buckets inherit evidence from their prefix ancestors — helps most when
  seeds number in the tens.
- **Reusable backbone**: population hashed once; per-task cost is a group-by.
  Background statistics are memoized per configuration.
- **Auto-tuner**: one hash at max resolution evaluates the whole
  (bits x tables x scoring) grid via prefix masking, cross-fitted to avoid
  OOF leakage; evaluates both engines and picks per dataset.
- **177 tests**, benchmark provenance (SHA-256 checksums, versions, git state)
  on every artifact.

## Benchmarks and evidence policy

All numbers live in [`outputs/`](outputs/README.md) with raw CSVs:

- [`COMBINER_REPORT.md`](outputs/COMBINER_REPORT.md) — how to spend one review
  budget across LeadX + XGBoost: a **cascade** (LeadX shortlists, XGBoost
  re-ranks) beats XGBoost alone by +22.5% Recall@0.1% (6/6 folds) at 1:1,000,
  and is the cheapest arrangement at scale. No meaningful gain at natural
  imbalance.
- [`COMPLEMENTARITY_REPORT.md`](outputs/COMPLEMENTARITY_REPORT.md) — LeadX vs
  XGBoost flag *different* transactions (overlap 0.11–0.12 at ultra imbalance);
  a rank-average ensemble adds +21% recall@0.1% over a PU-trained XGBoost
  (5/6 folds), but is neutral-to-harmful elsewhere.
- [`TEMPORAL_FRAUD_REPORT.md`](outputs/TEMPORAL_FRAUD_REPORT.md) — train on the
  past, score the future (IEEE fraud, rolling windows): LeadX survives temporal
  shift and drifts less than XGBoost (+27 pp at 1:1,000, 5/6), but is *not* more
  drift-robust than a linear PU baseline.
- [`CRITEO_PU_ULTRA_REPORT.md`](outputs/CRITEO_PU_ULTRA_REPORT.md) — 1:10,000 PU
  (production regime): trees collapse, LeadX beats XGBoost 9/9, ties linear-PU.
- [`CATEGORICAL_MINHASH_REPORT.md`](outputs/CATEGORICAL_MINHASH_REPORT.md) —
  MinHash categorical handling beats one-hot +56% (5/5) on IEEE fraud.

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

1. **Ultra-imbalance PU niche** — **measured** (1:10,000, Criteo 10M): beats
   XGBoost 9/9, ties linear-PU ([`CRITEO_PU_ULTRA_REPORT.md`](outputs/CRITEO_PU_ULTRA_REPORT.md)).
   Still needed: >=2 independent datasets, and a specialised PU baseline (not
   just naive unlabeled-as-negative).
2. Frozen-geometry accuracy: contrast weighting at scoring time
   (per-table task weights) and retrieve-then-rerank on backbone candidates.
3. Categorical handling via MinHash — **done** (`cat_engine="minhash"`),
   validated on one dataset; needs a second dataset + backbone-reuse integration.
4. English documentation.
5. XGBoost-shaped `LeadXClassifier` — **done** (v0.3–v0.4): `fit(X, y)` PU
   semantics, pandas/numpy/pyarrow input, `eval_set` for temporal selection,
   `sample_weight`, all-numeric fallback, richer metrics, sklearn compat.

## License & acknowledgments

MIT. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
