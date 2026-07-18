# LeadX Real-Data Pilot Benchmark

Run date: 2026-07-12

## Protocol

- Fixed LeadX configuration: median hashing, unsupervised projections, type-aware categorical bits, 10 tables, 10–12 bits.
- Models: two-class `prop`, two-class `lift`, PU `lift`, and HistGradientBoosting.
- Random stratified holdout: Telco, Adult, Home Credit.
- Temporal holdout: Hotel and IEEE fraud.
- Direct identifiers and known outcome leakage were removed (`customerID`, `SK_ID_CURR`, hotel reservation status/date).
- Large datasets were sampled reproducibly: Home Credit 150k rows; fraud 300k rows and 153 selected transaction features.
- HistGradientBoosting used the same fitted `FeatureEncoder` representation as LeadX.

## Main results

LeadX values are the better `prop`/`lift` result for the displayed ranking metric.

| Dataset | Test rows | Positive rate | LeadX ROC-AUC | LeadX PR-AUC | LeadX P@1% | LeadX lift@1% | LeadX fit | HistGB ROC-AUC | HistGB PR-AUC | HistGB P@1% | HistGB lift@1% | HistGB fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Telco churn | 2,114 | 26.54% | 0.806 | 0.579 | 77.27% | 2.91x | 0.09s | 0.836 | 0.639 | 86.36% | 3.25x | 0.71s |
| Adult income | 9,769 | 24.09% | 0.862 | 0.658 | 80.61% | 3.35x | 0.28s | 0.928 | 0.829 | 100.00% | 4.15x | 1.25s |
| Hotel cancellation | 35,817 | 39.33% | 0.772 | 0.727 | 100.00% | 2.54x | 0.82s | 0.905 | 0.863 | 100.00% | 2.54x | 5.77s |
| Home Credit default | 45,001 | 8.07% | 0.657 | 0.148 | 27.49% | 3.41x | 1.25s | 0.751 | 0.234 | 44.57% | 5.52x | 5.66s |
| IEEE fraud | 90,000 | 3.44% | 0.795 | 0.267 | 63.22% | 18.38x | 1.97–2.33s | 0.898 | 0.509 | 85.44% | 24.85x | 14.89s |

## PU confirmed-seed sensitivity

The backbone is fitted on the unlabeled population. Only the listed fraction of training positives is exposed as confirmed seeds.

| Dataset | Confirmed fraction | Confirmed seeds | ROC-AUC | PR-AUC | Precision@1% | Lift@1% |
|---|---:|---:|---:|---:|---:|---:|
| Home Credit | 10% | 848 | 0.595 | 0.122 | 21.95% | 2.72x |
| Home Credit | 25% | 2,119 | 0.631 | 0.135 | 23.73% | 2.94x |
| Home Credit | 50% | 4,238 | 0.643 | 0.141 | 26.83% | 3.32x |
| Home Credit | 100% | 8,476 | 0.657 | 0.148 | 27.49% | 3.41x |
| IEEE fraud | 10% | 735 | 0.771 | 0.208 | 49.00% | 14.25x |
| IEEE fraud | 25% | 1,837 | 0.784 | 0.236 | 55.78% | 16.22x |
| IEEE fraud | 50% | 3,674 | 0.790 | 0.253 | 59.00% | 17.16x |
| IEEE fraud | 100% | 7,348 | 0.795 | 0.264 | 61.22% | 17.80x |

## Interpretation

1. HistGradientBoosting remains the accuracy winner whenever complete positive and negative labels exist.
2. LeadX fits roughly 4–8x faster in this pilot and produces useful high-lift shortlists.
3. The strongest evidence for LeadX is PU fraud ranking: 735 confirmed fraud seeds still produce 49% precision in the top 1%, versus a 3.44% base rate.
4. PU lift with all positives exactly matches two-class lift here, as expected from the density-ratio formulation when the background is the full training population.
5. `lift` is not universally better than `prop`; select using the operational metric, especially Precision@K/Recall@K rather than ROC-AUC alone.

## Vector-capacity sweep on fraud PU

Grid: vectors per set (`n_bits`) = 8/10/12/14/16; sets (`n_tables`) = 10/20/40.
The maximum 16×40 backbone was hashed once and all configurations used nested prefixes.

| Confirmed positives | Baseline result | Best result | PR-AUC change | Precision@1% change | Lift@1% change |
|---|---|---|---:|---:|---:|
| 10% (735 seeds) | 12 bits × 10 tables: PR 0.206, P@1% 50.0% | **10 bits × 40 tables:** PR 0.251, P@1% 61.9% | +22.0% | +11.9 pp | 14.54x → 18.00x |
| 25% (1,837 seeds) | 12 bits × 10 tables: PR 0.225, P@1% 52.7% | **10 bits × 40 tables:** PR 0.267, P@1% 63.9% | +18.5% | +11.2 pp | 15.32x → 18.58x |

Increasing tables consistently reduced ensemble variance. Increasing bits beyond 10 hurt because
bucket resolution outpaced seed coverage: join miss rate rose from approximately zero at 10 bits
to 1.6% at 14 bits and 5.3% at 16 bits. For low-seed PU, capacity should therefore come primarily
from more tables, not finer buckets.

## Extreme-imbalance stress test

Fraud positives were downsampled in both train and temporal test populations to 3%, 1%, 0.3%,
and 0.1%. LeadX received only positive seeds plus unlabeled background. HistGB received complete
positive and negative labels; both plain and class-weight-balanced variants were tested.

| Positive rate | Train positives | LeadX PR-AUC | HistGB plain PR-AUC | HistGB balanced PR-AUC | LeadX P@0.1% | HistGB plain P@0.1% |
|---:|---:|---:|---:|---:|---:|---:|
| 3.0% | 6,268 | 0.258 | **0.476** | 0.457 | 81.1% | **95.6%** |
| 1.0% | 2,047 | 0.148 | **0.306** | 0.285 | 64.8% | **86.4%** |
| 0.3% | 610 | 0.064 | **0.132** | 0.119 | 28.4% | **39.8%** |
| 0.1% | 203 | 0.044 | **0.064** | 0.034 | 6.9% | **11.5%** |

No real-data crossover against plain HistGB was observed down to 0.1% prevalence. LeadX beat the
balanced HistGB variant on PR-AUC at 0.1%, but not the stronger unweighted HistGB. The gap narrowed
at the lowest prevalence, yet this single split has only 87 positive test cases and should not be
treated as a stable win/loss boundary. Extreme imbalance alone is therefore not a demonstrated moat;
the supported advantage remains learning from positive-only labels and fast reusable retrieval.

## Limitations

- LeadX used a fixed configuration, not per-dataset tuning.
- HistGradientBoosting shared LeadX's encoder; stronger task-specific categorical processing may improve it further.
- Fraud used 153 selected transaction features and did not join the identity table.
- PU seed sampling is random among known positives; real confirmation bias can reduce performance.
- This is an offline holdout benchmark, not calibration, drift, or production monitoring validation.
