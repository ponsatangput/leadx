# Criteo Uplift v2.1 — rare-outcome / reusable-backbone benchmark (2026-07-18)

> **Newer evidence (2026-07-21):** the 5-split gate with Bagging Linear SVM
> reduces the 20-seed PR effect to +1.8% [−8.5%, +9.3%]. Read
> [`CRITEO_MULTISPLIT_GATE_REPORT.md`](CRITEO_MULTISPLIT_GATE_REPORT.md) first.
> The report below is retained as the original fixed-split/scale study.

## คำตัดสิน

| สมมติฐาน | ผลจากรอบนี้ |
|---|---|
| LeadX เด่นเมื่อ rare event มากและมี confirmed seeds น้อยมาก | **มีหลักฐานรองรับเฉพาะ task-specific contrast ที่ 20 seeds**: conversion PR-AUC paired median +20.4% เทียบ linear-PU envelope, ชนะ 8/10 draws |
| ความได้เปรียบยังอยู่เมื่อ seeds เพิ่ม | **ไม่รองรับ**: ที่ 100 seeds contrast RPForest เหลือ +1.7%, CI คร่อมศูนย์; recall@0.5% ติดลบ |
| สร้าง geometry ครั้งเดียวแล้วใช้หลายโจทย์ได้ | **รองรับในเชิงระบบจาก joint pilot**: invocation เดียวสร้าง frozen backbone แล้ว fit conversion/visit heads ได้ราว 0.03s/task; 10-draw accuracy runs แยก invocation และ reconstruct geometry เดิม |
| Frozen/reusable geometry แม่นกว่า baseline พร้อมกัน | **ไม่รองรับ**: conversion 20 seeds สูสี, visit 20 seeds สูสีเฉพาะ PR แต่ recall แพ้; ที่ 100 seeds แพ้ชัด |
| LeadX เร็วกว่า end-to-end บนข้อมูลนี้ | **ไม่รองรับ**: 12 dense features เป็นสนามที่ linear PU fit/score เร็วกว่ามาก แม้ LeadX task-head จะถูกมาก |

ดังนั้นจุดขายที่ปลอดภัยตอนนี้คือ:

> เมื่อมี confirmed positives เพียงประมาณ 20 ตัว task-specific LeadX มีสัญญาณว่า
> จัดอันดับ rare conversions ได้ดีกว่า selected PU baselines; ความได้เปรียบนี้
> ไม่ได้เกิดจาก imbalance อย่างเดียว และยังไม่เกิดพร้อมกับ frozen-backbone reuse

ห้ามย่อเป็น “traditional algorithms fail แต่ LeadX ทำงานได้” เพราะ linear PU
ยังทำงานได้ดี และ full-label oracle ยังสูงกว่า LeadX ทุก run ในรายงานนี้

## ข้อมูลและ estimand

- แหล่งข้อมูล: [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), v2.1 ที่แก้ advertiser-identification leak ของ v1 จาก [official Criteo Hugging Face repository](https://huggingface.co/datasets/criteo/criteo-uplift)
- ไฟล์จริง 13,979,592 แถว, 12 dense anonymized features (`f0..f11`)
- observed conversion rate = 40,774 / 13,979,592 = **0.291668%**; visit = **4.6992%**
- SHA-256 = `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- ใช้ `f0..f11` เท่านั้น; ตัด `treatment`, `conversion`, `visit`, `exposure` ออกจาก predictors
- License: CC BY-NC-SA 4.0

นี่คือ **rare-outcome retrieval จาก simulated confirmed seeds** ไม่ใช่ uplift/ITE
benchmark และ dataset เดิมไม่ได้เป็น native PU dataset. Criteo ระบุว่าข้อมูลถูก
non-uniformly subsample เพื่อ privacy จึงเรียก 0.291668% ได้เพียง observed rate
ของ release นี้ ไม่ใช่ natural/production prevalence. Dataset ไม่มี timestamp
จึงไม่มีหลักฐาน temporal robustness

Population ใน benchmark คือทุกแถว: pool randomized treatment/control ตาม mixture
85/15 ของ release และไม่ใช้ `treatment` เป็น predictor. Estimand จึงเป็น conversion
retrieval marginal over mixture นี้ ไม่ใช่ treated/control propensity หรือ
counterfactual treatment effect

## Protocol

- Development + main evaluation: uniform sample 1,000,000 rows โดยไม่ resample label
- Feature-hash group split 60/20/20; feature vectors ที่เหมือนกันทุกคอลัมน์อยู่ split เดียวกัน
- Main test: train 599,242 / test 200,306; conversion positives 1,760 / 594
- Seed budgets 20 และ 100; 10 fixed paired draws; budget 20 เป็น prefix ของ budget 100
- LeadX config คงที่: `b10 K80`, lift/log, hierarchical depths auto
- Frozen arms: unsupervised hyperplane และ RPForest geometry ที่ reuse ข้าม task
- Task-specific arms: contrast-weighted hyperplane และ RPForest ซึ่ง rebuild ต่อ seed task
- Baselines:
  - balanced Logistic Regression บน seeds-vs-unlabeled (`logreg_pu_naive`)
  - 30-model bagged balanced Logistic Regression, แต่ละ model ใช้ all seeds + 1,000 random unlabeled
  - balanced HistGradientBoosting บน seeds-vs-unlabeled
  - full-label balanced HistGradientBoosting เป็น oracle reference
- Primary metric: PR-AUC; operational metric: recall เมื่อ review top 0.5% ของ test population
- Paired delta เทียบ baseline ที่ดีกว่าระหว่าง LogReg และ Bagged-LogReg **ใน draw เดียวกันและ metric เดียวกัน**
- 95% bootstrap interval resample เฉพาะ 10 seed draws; ไม่ครอบคลุม split/dataset/model variance

Bagged-LogReg เป็น logistic adaptation ของแนวคิด
[Bagging-PU ของ Mordelet–Vert](https://www.sciencedirect.com/science/article/abs/pii/S0167865513002432)
ไม่ใช่ implementation เดิมที่ใช้ SVM และไม่ใช่ตัวแทน SOTA PU ทั้งหมด

Main evaluation เป็น test groups ที่แยกจาก validation ตอนพัฒนา แต่รายงานทั้งหมด
ยังถือเป็น **exploratory**: fixed split เดียว, config ถูกพัฒนาภายในโปรเจกต์เดียวกัน,
และ full-scale run ด้านล่าง reuse split เดิม จึงไม่ใช่ independent confirmation
แม้ PR arm หลักชนะ 8/10 draws, exact one-sided sign-test p=.0547 และมีหลาย
arms/metrics ที่สำรวจ จึงไม่ถือว่าเป็น confirmatory significance test

## ผล conversion บน 1M sample — test groups, 10 paired draws

ค่าด้านล่างคือ median ข้าม draws; base rate ใน test = 0.2965%

| model | geometry | 20 seeds PR / R@0.5% | 100 seeds PR / R@0.5% |
|---|---|---:|---:|
| **LeadX contrast RPForest** | task-specific | **.1533 / .3468** | **.1721 / .3510** |
| LeadX contrast hyperplane | task-specific | .1475 / .3359 | .1677 / .3502 |
| LeadX frozen RPForest | reusable | .1400 / .3199 | .1615 / .3418 |
| LeadX frozen hyperplane | reusable | .1368 / .3106 | .1584 / .3165 |
| Bagged-LogReg PU | PU baseline | .1281 / .3190 | .1674 / **.3594** |
| LogReg PU-naive | PU baseline | .1238 / .3140 | .1685 / .3577 |
| HistGB PU-naive | PU baseline | .0418 / .1608 | .0668 / .2012 |
| HistGB full-label oracle | reference | .2038 / .3956 | same fit |

### Paired comparison กับ stronger linear-PU baseline ต่อ draw

| seeds | LeadX arm | paired Δ PR-AUC [95% draw-bootstrap] | wins | paired Δ R@0.5% [95%] | wins |
|---:|---|---:|---:|---:|---:|
| 20 | **contrast RPForest** | **+20.4% [+2.5, +31.4]** | **8/10** | +9.6% [−1.4, +16.6] | 7/10 |
| 20 | contrast hyperplane | **+16.9% [+2.5, +27.3]** | 8/10 | +6.0% [−1.5, +11.5] | 7/10 |
| 20 | frozen RPForest | +0.3% [−3.5, +18.2] | 5/10 | +4.2% [−18.0, +8.9] | 6/10 |
| 20 | frozen hyperplane | +2.8% [−9.3, +16.5] | 5/10 | +0.9% [−23.0, +5.3] | 5/10 |
| 100 | contrast RPForest | +1.7% [−3.1, +6.9] | 6/10 | −2.5% [−7.7, +1.7] | 2/10 (+2 ties) |
| 100 | contrast hyperplane | −2.6% [−7.6, +5.5] | 4/10 | −3.3% [−7.6, −1.0] | 1/10 |
| 100 | frozen RPForest | −4.3% [−14.9, +1.7] | 3/10 | −5.8% [−11.4, 0.0] | 2/10 (+1 tie) |
| 100 | frozen hyperplane | −6.5% [−17.3, −2.0] | 1/10 | −14.1% [−21.8, −7.3] | 0/10 |

หลักฐานที่ชัดคือ **PR-AUC ที่ 20 seeds + task-specific contrast**. Recall@0.5%
มี positive point estimate แต่ interval ยังคร่อมศูนย์ จึงห้ามเขียนว่าชนะครบทุก metric

## Reuse ข้าม outcome: visit บน backbone เดิม

รอบนี้ปิด task-specific arms แล้วทดสอบเฉพาะ frozen geometry กับ baselines เพื่อถามว่า
“geometry เดิมใช้โจทย์ใหม่ได้ดีแค่ไหน” test visit base rate = 4.7447%
ตาราง accuracy ใช้ standalone 10-draw run ซึ่ง rebuild geometry เดิมแบบ deterministic;
joint pilot แยกต่างหากตรวจแล้วว่า backbone object หนึ่งตัว fit ทั้ง conversion และ
visit heads ได้โดยไม่ re-hash population (`criteo_uplift_pilot_raw.csv`, 3 draws)

| model | 20 seeds PR / R@0.5% | 100 seeds PR / R@0.5% |
|---|---:|---:|
| LeadX frozen hyperplane | .3376 / .0512 | .3920 / .0572 |
| LeadX frozen RPForest | .3222 / .0513 | .3827 / .0574 |
| **Bagged-LogReg PU** | **.3383 / .0615** | **.4524 / .0705** |
| LogReg PU-naive | .3226 / .0592 | .4499 / **.0708** |
| HistGB PU-naive | .1581 / .0287 | .2361 / .0377 |
| HistGB full-label oracle | .5182 / .0782 | same fit |

- 20 seeds: frozen hyperplane PR paired −2.4% [−11.3, +5.3], 4/10 wins;
  recall −17.0% [−23.4, −6.9], 1/10 wins
- 100 seeds: frozen hyperplane PR −10.7% [−16.8, −9.1], 0/10 wins;
  recall −17.2% [−22.1, −14.7], 0/10 wins

สรุป: reuse API/geometry ใช้งานได้จริง แต่ frozen geometry ปัจจุบัน **ไม่ใช่ accuracy moat**
ข้าม task. ที่ 20 seeds PR สูสีได้ แต่ recall shortlist แพ้ baseline ชัด

## Full 13.98M-row scale replication — conversion, 20 seeds, 3 draws

- train 8,385,418 / test 2,795,112; test conversion rate 0.2932%
- exact-feature duplicate row fraction = 15.17%; group split จึงจำเป็น

| model | median PR | median R@0.5% |
|---|---:|---:|
| LeadX contrast RPForest | **.1606** | **.3606** |
| LeadX contrast hyperplane | .1482 | .3575 |
| LeadX frozen RPForest | .1333 | .3350 |
| LeadX frozen hyperplane | .1324 | .3464 |
| Bagged-LogReg PU | .1265 | .3120 |
| LogReg PU-naive | .1118 | .2854 |
| HistGB PU-naive | .0244 | .1063 |
| HistGB full-label oracle | .2294 | .4150 |

อย่าใช้ `.1606/.1265 − 1 = +27%` เพราะเป็น ratio-of-medians จากคนละ draws.
Paired contrast-RPForest เทียบ baseline envelope ได้ PR **+3.1%**, ชนะ 3/3;
R@0.5% **+15.6%**, ชนะ 3/3; แต่ tightest R@0.1% **−6.5%**, ชนะเพียง 1/3.
สาม draws น้อยเกินสรุปความเสถียร (แม้ชนะ 3/3, one-sided sign-test p=.125)
จึงใช้ผลนี้ได้เพียงว่า direction บาง metric ยังอยู่เมื่อ scale ใหญ่ขึ้น

## เวลา — full dataset

| arm | shared setup | fit/task-head ต่อ task | score 2.80M rows | คำอ่าน |
|---|---:|---:|---:|---|
| frozen hyperplane | 68.4s (build + eval hash + bg stats) | **0.030s** | 16.3s | task head ถูกมาก แต่ scoring ยังแพง |
| frozen RPForest | 176.0s | **0.039s** | 17.5s | setup แพงกว่าโดยไม่สร้าง frozen accuracy win |
| contrast hyperplane | — | 60.0s | 19.8s | rebuild ต่อ task |
| contrast RPForest | — | 143.5s | 43.9s | arm แม่นสุด แต่ไม่ใช่ reuse |
| Bagged-LogReg PU | 1.65s shared encoding | 0.067s | **0.559s** | end-to-end เร็วกว่า LeadX บน 12 features |
| LogReg PU-naive | 1.65s shared encoding | 3.93s | **0.017s** | scoring เร็วมาก |
| HistGB PU-naive | 1.65s shared encoding | 22.2s | 2.09s | accuracy แพ้ใน few-seed run นี้ |

จุดขายด้านเวลาที่พิสูจน์ได้คือ **task-head construction** ไม่ใช่ broad end-to-end
speed. ใน dataset 12 มิติ linear models เป็นคู่แข่งที่เร็วและแข็งแรงมาก

## สิ่งที่โค้ดเพิ่มจากการทดลองนี้

- `TaskModel.score_codes(codes)` เป็น public API: `bb.transform(X_new)` ครั้งเดียว
  แล้ว reuse codes เดิมข้ามหลาย task โดย task จัดการ bit-prefix/table subset เอง
- validation ป้องกัน shape, table count, non-integer, negative และ out-of-range codes
- benchmark checkpoint raw CSV แบบ atomic และลบ stale raw ก่อน overwrite
- seed draws ไม่ขึ้นกับลำดับ target
- metadata รุ่นใหม่เก็บ argv, source checksum status, package versions, git revision,
  dirty flag, benchmark-script SHA-256 และ hash ของ `src/leadx` source tree
- paired-analysis script ป้องกัน ratio-of-medians error และ reject duplicate/incomplete draws
- default benchmark รวม Bagged-LogReg ให้ตรงกับ baseline envelope ของ summarizer

## Artifacts และ reproduce

Primary artifacts:

- `outputs/criteo_uplift_benchmark.py`
- `outputs/summarize_criteo_uplift.py`
- `outputs/criteo_uplift_test10_raw.csv` + metadata
- `outputs/criteo_uplift_visit_test10_raw.csv` + metadata
- `outputs/criteo_uplift_full20_3draw_raw.csv` + metadata
- `outputs/criteo_uplift_pilot_raw.csv` + metadata (joint-backbone engineering check)

Artifact status: ไฟล์เหล่านี้ยังเป็น working-tree artifacts ไม่ใช่ committed release
bundle. `visit_test10` มี argv, runtime versions, git revision/dirty flag และ script
SHA; source-tree SHA ถูกเพิ่มใน producer หลัง run นี้. `test10` และ full-scale ถูก
สร้างก่อน provenance patch จึงมี source/config metadata แต่ยังไม่มี fields เหล่านี้.
คำสั่งด้านล่าง regenerate ทุก artifact ด้วย schema ปัจจุบัน

```bash
cd /Users/ponpiboonsatangput/jarvis/sandbox/data_science/leadx

# conversion, 1M, test groups, 10 paired draws
uv run python outputs/criteo_uplift_benchmark.py \
  --targets conversion --eval-split test --draws 10 --seed-budgets 20,100 \
  --baseline-models logreg_pu,bagging_logreg_pu,histgb_pu,histgb_oracle \
  --output outputs/criteo_uplift_test10_raw.csv --overwrite

# frozen backbone transfer to visit
uv run python outputs/criteo_uplift_benchmark.py \
  --targets visit --eval-split test --draws 10 --seed-budgets 20,100 \
  --task-specific-targets '' \
  --baseline-models logreg_pu,bagging_logreg_pu,histgb_pu,histgb_oracle \
  --baseline-targets visit \
  --output outputs/criteo_uplift_visit_test10_raw.csv --overwrite

# full scale replication (exploratory; 3 draws only)
uv run python outputs/criteo_uplift_benchmark.py \
  --max-rows 0 --targets conversion --eval-split test \
  --draws 3 --seed-budgets 20 \
  --baseline-models logreg_pu,bagging_logreg_pu,histgb_pu,histgb_oracle \
  --output outputs/criteo_uplift_full20_3draw_raw.csv --overwrite

# paired deltas + bootstrap intervals
uv run python outputs/summarize_criteo_uplift.py \
  outputs/criteo_uplift_test10_raw.csv \
  outputs/criteo_uplift_visit_test10_raw.csv \
  outputs/criteo_uplift_full20_3draw_raw.csv
```

## Gate ถัดไปก่อน claim ต่อ community

1. ล็อก config/metric/gate ล่วงหน้าแล้วรัน dataset อิสระอีกอย่างน้อย 2 ชุด
2. เพิ่ม strong PU baselines: original Bagging-PU, Elkan–Noto และวิธีสมัยใหม่ที่เหมาะกับ tabular data
3. ใช้หลาย group splits; ถ้ามี timestamp ให้ทำ temporal holdout
4. ถ้าต้องการขาย A+B พร้อมกัน ต้องปรับ frozen geometry ให้รักษา contrast/localization
   โดยไม่ rebuild ต่อ task แล้วชนะ baseline ทั้ง accuracy และ total runtime

**สถานะสุดท้าย:** rare/few-seed hypothesis มี preliminary exploratory signal
แบบ regime-specific; reusable-backbone accuracy moat และ end-to-end speed moat
ยังไม่ผ่าน
