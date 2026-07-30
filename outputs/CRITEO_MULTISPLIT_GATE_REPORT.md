# Criteo multi-split strong-PU gate (2026-07-21)

## คำตัดสิน

**Gate ไม่ผ่านสำหรับ claim ว่า LeadX ชนะ strong PU baseline ที่ 20 confirmed
seeds.** เมื่อเพิ่ม Bagging Linear SVM และเปลี่ยนจาก fixed split เดียวเป็น 5
feature-group splits ใหม่ ความได้เปรียบ PR-AUC เดิมลดเหลือ **+1.8%** และ
hierarchical 95% interval คร่อมศูนย์กว้าง. Recall@0.5% เป็นบวกใน median ของทั้ง
5 splits แต่ interval ยังคร่อมศูนย์ จึงนับเป็นสัญญาณที่ควรตรวจต่อ ไม่ใช่ชัยชนะ.

| metric | paired median เทียบ best PU baseline | hierarchical 95% interval | pair wins | split-median wins |
|---|---:|---:|---:|---:|
| PR-AUC | **+1.8%** | **[−8.5%, +9.3%]** | 26/50 | 2/5 |
| Recall@0.5% | **+3.7%** | **[−1.0%, +11.9%]** | 31/50 | 5/5 |

ดังนั้น claim `+20.4% PR-AUC` จากรายงาน 2026-07-18 ยังเป็นคำอธิบายผลของ
fixed split และ linear-PU envelope เดิมได้ แต่ **ห้ามใช้เป็น estimate ทั่วไป**
หลังมีหลักฐาน multi-split และ SVM baseline ชุดนี้.

## Protocol ที่ล็อกก่อนรัน

- Criteo Uplift v2.1 uniform sample เดิม 1,000,000 rows; conversion observed
  rate 0.296% ใน sample
- split seeds ใหม่ `31001,31002,31003,31004,31005`; feature-hash 60/20/20;
  exact duplicate feature rows อยู่ split เดียวกัน
- test groups, 10 nested seed draws ต่อ split, 20 confirmed positives ต่อ draw
- LeadX config คงที่จากก่อนเห็น splits ใหม่: `b10 K80`, task-specific contrast
  RPForest, lift/log, hierarchical depths auto
- baseline envelope เลือกค่าที่ดีที่สุด **ภายใน pair และ metric เดียวกัน** จาก
  LogReg PU-naive, 30-model Bagged-LogReg และ Bagging Linear SVM
- Bagging Linear SVM ทำตามโครง inductive Bagging-PU ของ
  [Mordelet–Vert](https://arxiv.org/abs/1010.0772): ทุก learner ใช้ positives
  ทั้งหมด, สุ่มเฉพาะ unlabeled, `K=|P|=20`, 35 linear SVMs, fixed `C=1`
- interval ใช้ hierarchical bootstrap 100,000 ครั้ง: resample split ก่อน แล้ว
  resample draws ภายใน split; รายงานนี้ยัง descriptive เพราะ 5 partitions
  มาจาก sample เดียวกัน ไม่ใช่ 5 datasets อิสระ

Elkan–Noto probability correction แบบ `g(x)/c` ไม่ได้เพิ่มเป็นแถวซ้ำ เพราะเป็น
positive constant scaling และจึงไม่เปลี่ยนลำดับสำหรับ PR-AUC/top-k metrics.
Elkan–Noto weighted-retraining variant และ modern PU methods ยังไม่ได้ทดสอบ.

## Raw medians ข้าม 50 pairs

| model | PR-AUC | Recall@0.5% | fit/task median | score median |
|---|---:|---:|---:|---:|
| LeadX contrast RPForest | **.1511** | **.3520** | 27.48s | 2.91s |
| Bagging Linear SVM PU | .1497 | .3330 | **.013s** | .041s |
| Bagged-LogReg PU | .1088 | .2802 | .058s | .038s |
| LogReg PU-naive | .1075 | .2760 | .261s | **.001s** |

Ratio ของ raw medians ไม่ใช่ paired effect; ให้ใช้ตารางคำตัดสินด้านบน. สนามนี้มี
เพียง 12 dense features จึงยืนยันอีกครั้งว่า LeadX ไม่มี end-to-end speed moat
เมื่อเทียบ linear models.

## ความเสถียรแยกตาม split

| split seed | PR paired median / wins | Recall@0.5% paired median / wins |
|---:|---:|---:|
| 31001 | +3.3% / 7-of-10 | +4.8% / 6-of-10 |
| 31002 | −0.7% / 5-of-10 | +7.1% / 6-of-10 |
| 31003 | +12.2% / 7-of-10 | +7.6% / 7-of-10 |
| 31004 | −0.3% / 5-of-10 | +6.7% / 7-of-10 |
| 31005 | −9.0% / 2-of-10 | +0.2% / 5-of-10 |

PR sensitivity ต่อ data partition คือสาเหตุที่ fixed-split result สูงเกินหลักฐาน
รวม. Recall มีทิศทางสม่ำเสมอกว่า แต่ effect ยังเล็กและ uncertainty ยังคร่อมศูนย์.

## Artifacts และ reproduce

- `outputs/criteo_uplift_multisplit_gate_raw.csv`
- `outputs/criteo_uplift_multisplit_gate_metadata.json`
- `outputs/criteo_uplift_multisplit_gate_summary.csv`
- producer: `outputs/criteo_uplift_benchmark.py`
- analyzer: `outputs/summarize_criteo_uplift.py`

```bash
uv run python outputs/criteo_uplift_benchmark.py \
  --max-rows 1000000 --targets conversion --eval-split test \
  --draws 10 --seed-budgets 20 \
  --split-seeds 31001,31002,31003,31004,31005 \
  --frozen-engines '' --task-specific-engines rpforest \
  --task-specific-targets conversion \
  --baseline-models logreg_pu,bagging_logreg_pu,bagging_svm_pu \
  --baseline-targets conversion \
  --output outputs/criteo_uplift_multisplit_gate_raw.csv --overwrite

uv run python outputs/summarize_criteo_uplift.py \
  outputs/criteo_uplift_multisplit_gate_raw.csv \
  --baselines logreg_pu_naive,bagging_logreg_pu,bagging_linear_svm_pu
```

## การตัดสินใจถัดไป

ไม่ขยาย Criteo เป็น seed budgets 50/100 ในรอบนี้ เพราะ primary 20-seed PR gate
ไม่ผ่าน. งานที่ให้ข้อมูลใหม่กว่าคือ dataset อิสระ หรือปรับ algorithm เพื่อเพิ่ม
localization/recall แล้วประเมินด้วย protocol เดิมโดยไม่แตะ splits ชุดนี้อีก.
