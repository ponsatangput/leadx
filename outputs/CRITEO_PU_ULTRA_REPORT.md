# Criteo PU ultra-imbalance — trees collapse, LeadX holds (v0.2-B, 2026-07)

Turns the README's **hypothesis** (accuracy edge at ~1:10,000 in PU mode — the
production regime) into a measured result. Pre-registered protocol locked in
`criteo_pu_ultra_benchmark.py` before the full run.

## Protocol
- Criteo Uplift v2.1, uniform 10M sample, conversion target
- imbalance: downsample positives to **1:10,000** (and 1:1,000 for contrast)
- 3 feature-group hash splits × 3 PU seed draws (genuine PU: **50% of train
  positives confirmed**, the rest left UNLABELED — allseeds would collapse
  PU-naive into supervised)
- models: `leadx_pu` (rpforest b10 K80 lift/log hier contrast, PU),
  `logreg_pu_naive`, `xgboost_pu_naive`, `xgboost_oracle` (full labels, ceiling)
- paired per (split, draw); oracle paired per split. NOT a significance test.

## Result — 1:10,000 (production regime), leadx_pu paired (9 obs)

| leadx vs | PR-AUC | Recall@0.5% | Recall@0.1% |
|---|---|---|---|
| **xgboost_pu_naive** | **+325% (9/9)** | **+208% (9/9)** | **+183% (9/9)** |
| **xgboost_oracle** (full labels) | **+240% (9/9)** | **+123% (9/9)** | **+62% (9/9)** |
| logreg_pu_naive | −10% (3/9) | +4% (6/9) | −18% (0/9) |

median absolute (1:10,000): PR — logreg .0074, leadx .0067, xgb-oracle .0021,
xgb-pu .0017; Recall@0.5% — leadx .403, logreg .402, xgb-oracle .181, xgb-pu .129.

## Reading

1. **Tree-based methods collapse at ultra-extreme imbalance PU.** XGBoost-PU-naive
   drops to PR .0017 / Recall@0.5% .13 — LeadX beats it 9/9 on every metric
   (+325% PR). Even XGBoost with **full labels** loses to LeadX-with-half-the-
   positives 9/9 (+240% PR). The ~700 positives among 7M train rows are too few
   for tree splits regardless of the labelling.
2. **Density-ratio (LeadX) and linear (LogReg) both hold** and are ~tied. LeadX
   does **not** beat LogReg (LogReg slightly better on PR and Recall@0.1%). So the
   honest niche is *"trees fail here; LeadX and linear don't, and match each
   other"* — not "LeadX is uniquely best."
3. **Edge is ultra-specific.** At 1:1,000 the advantage over XGBoost is mostly
   gone (PR +2%, 5/9) — trees recover once positives are 10× denser.

## Honest positioning (measured, replaces the hypothesis)

> At ultra-extreme imbalance (~1:10,000) in PU mode, gradient-boosted trees
> (XGBoost) collapse — even with full labels. LeadX beats XGBoost by ~3–4× on
> PR-AUC and ~2–3× on recall (9/9 splits×draws), and ties a strong linear-PU
> baseline. LeadX's differentiator in this regime is therefore its reuse/scale
> architecture on top of top-tier accuracy, not accuracy alone.

## Caveats
- one dataset (Criteo, dense 12-feature), 3 splits × 3 draws — replication, not
  a significance test; no independent-dataset confirmation
- LeadX config is task-specific (contrast + rpforest fit per task), not the
  frozen reusable backbone — rare-event accuracy and reuse still proven separately
- `xgboost_pu_naive` is the naive unlabeled-as-negative PU conversion (what people
  reach for), not a specialised PU tree method

## Reproduce
```bash
cd sandbox/data_science/leadx
uv run --with xgboost python outputs/criteo_pu_ultra_benchmark.py
#   → outputs/criteo_pu_ultra_raw.csv (source of truth)
```
