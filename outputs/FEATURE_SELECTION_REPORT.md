# Feature selection before LeadX — measured, rejected (v0.4, 2026-08)

**คำถาม:** ตัด features ก่อนเข้า LeadX ช่วยที่ ultra-imbalance มั้ย? สมมติฐานคือ
SNR ต่อ projection ของ random-hyperplane/rpTree ≈ √(k/(D−k)) — ถ้า D ใหญ่และมี
feature ที่ไม่มีสัญญาณเยอะ (fraud D=153, home_credit D=118) การคัดทิ้งควรช่วย

**คำตอบ: ไม่ช่วย — ทำให้แย่ลงมาก และแย่ลงแม้ใช้ oracle**

## ผล — paired ต่อ (dataset, ratio, draw) เทียบ all-features, primary = PR-AUC

| variant | median Δ PR-AUC | wins | worst | best | gate |
|---|---|---|---|---|---|
| oracle_ig_16 | **−40.4%** | 1/9 | −78.5% | +24.6% | ❌ |
| oracle_ig_48 | **−29.0%** | 1/9 | −70.2% | +26.2% | ❌ |
| pu_ig_16 | **−43.8%** | 1/9 | −85.4% | +11.2% | ❌ |
| pu_ig_48 | **−26.2%** | 1/9 | −62.5% | +0.6% | ❌ |

Gate ที่ pre-register ไว้ (commit `74664bd` ก่อนรัน): median Δ > +5% และชนะ ≥7/9
— **ไม่มี variant ใดเข้าใกล้เลย** ทุกตัวติดลบสองหลัก ชนะ 1/9 เท่ากันหมด

Cells: fraud @ 0.3%/0.1%, home_credit @ 0.3% × 3 draws (seed `rng(100*d+7)` ตรงกับ
`imbalance_focus_rpf.py`). Baseline PR-AUC: fraud .0595/.0112, home_credit .0066

## หลักฐานชิ้นสำคัญ — oracle ไม่ดีกว่า PU-IG

| k | oracle median | pu_ig median | oracle ชนะ pu_ig |
|---|---|---|---|
| 16 | −40.4% | −43.8% | 6/9 |
| 48 | −29.0% | **−26.2%** | 4/9 |

`oracle_ig` คำนวณ IG จาก **label จริงทั้ง training set** — คือขีดบนของสิ่งที่การคัด
feature แบบ univariate ทำได้ ถ้าปัญหาคือ "IG แบบ PU ประมาณค่าห่วยเพราะ seeds น้อย
(20–61 ตัว)" oracle ต้องชนะขาด **แต่มันเสมอกับ PU-IG** (4–6 จาก 9 คือเหรียญโยน)

→ ความล้มเหลว**ไม่ได้**อยู่ที่การประมาณ IG แต่อยู่ที่ **เกณฑ์ univariate เองผิดกับ
density-ratio scorer**

## ทำไม

1. **IG เป็นเกณฑ์ marginal — LeadX กินสัญญาณ joint** bucket ที่ discriminative ไม่ได้
   มาจาก "ทุก feature ในนั้นทำนายเก่งเดี่ยวๆ" แต่มาจาก*การประกอบกัน*ที่หายากและ
   enriched IG ตัด feature ที่ marginal ไม่มีสัญญาณแต่ไปร่วมสร้าง conjunction ที่แยกได้ทิ้ง
   — ซึ่งสำหรับ density ratio นั่นคือสัญญาณ

2. **ตัด feature = ลด resolution ของ partition** D น้อยลง → configuration ของ bucket
   ที่แยกออกได้น้อยลง → lift หยาบขึ้นตรงหัวแถว สอดคล้องกับสองอย่างที่วัดได้:
   **k=48 แย่น้อยกว่า k=16 ทุกช่อง** (monotone กับ D) และ **ROC เสียหายน้อยกว่า PR-AUC
   มาก** (median −1.7 ถึง −9.2% เทียบ −26 ถึง −44%) — ความเสียหายกระจุกที่หัวแถว
   ซึ่งคือที่ที่ rare-event retrieval อยู่จริง

3. สมมติฐาน √(k/(D−k)) **ตั้งบนข้อสันนิษฐานที่ผิดในโจทย์นี้** — มันสมมติว่าสัญญาณอยู่ใน
   subspace k มิติของ feature ที่ informative เดี่ยวๆ ใน fraud/credit ไม่ใช่แบบนั้น

## ผลพลอยได้ต่อ positioning

การที่ LeadX ใช้ feature กว้างๆ ได้โดยไม่พัง **เป็นคุณสมบัติ ไม่ใช่จุดอ่อน** — ตรงข้ามกับ
ที่ผมเดาไว้ ผู้ใช้ไม่ต้องทำ feature selection ก่อน และไม่ควรทำ

Bonus: `all` ใช้เวลา fit นานกว่า (median 24s ที่ D เต็ม vs 20s ที่ k=16) — คือจ่าย
เวลาเพิ่ม ~20% เพื่อ PR-AUC ที่ดีกว่า 26–44%

## ขอบเขต (สิ่งที่ผลนี้ **ไม่ได้** ตอบ)

- ทดสอบบน feature จริงของสองชุดข้อมูล — **ไม่ได้**ทดสอบกรณีเติม pure random noise
  columns เข้าไป 500 คอลัมน์ ข้อสรุปคือ "อย่าคัดจาก feature จริง" ไม่ใช่ "LeadX ทน
  noise ได้ไม่จำกัด"
- 9 paired observations, 2 ชุดข้อมูล, k แค่ 2 ค่า (16, 48) — แต่ทิศทางแรงและสม่ำเสมอ
  พอที่จะปิดประเด็นโดยไม่ต้องกวาดละเอียด
- ไม่ได้ทดสอบ **การถ่วงน้ำหนัก** (ต่างจากการ*คัดทิ้ง*) — `feature_weighting="contrast"`
  ที่ ship อยู่ยังเปิดใช้ตลอดในทุก variant ข้างบน

## Reproduce
```bash
cd sandbox/data_science/leadx
PYTHONPATH=outputs uv run python outputs/feature_selection_experiment.py
#   → outputs/feature_selection_raw.csv (source of truth)
```
