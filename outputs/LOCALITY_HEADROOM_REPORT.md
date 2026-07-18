# Locality Headroom Report — Phase 1

**คำถาม:** partition geometry ของ leadx ยังมี headroom ไหม? (gate ก่อนลงทุน Phase 2A
local projection forest)

## Setup

- Dataset: home-credit PU proxy (spec เดิมอ้าง fraud+temporal split ซึ่งไม่มีใน workspace
  — ใช้ credit default + stratified split แทน, จดเป็นข้อจำกัด)
- Train 246,009 (pos 19,854) / test 20,000 คงที่ (subsample เพื่อให้ exact kNN ไหว)
- Seeds: 10% / 25% ของ positives × 10 draws (ไม่มี hyperparameter ใดจูนบน test;
  ยกเว้น kNN oracle รายงาน per-draw-best-k โดยตั้งใจ — ให้ oracle ได้เปรียบ เพื่อให้
  gate conservative ไปทาง "มี headroom")
- Scorer เดียวกันทุกวิธี: lift (α=2, geometric agg) — แยกผล geometry ออกจาก scorer

## ผล (median PR-AUC ข้าม 10 draws)

| method | 10% seeds | 25% seeds |
|---|---|---|
| **leadx b10 K=80** | **0.1548** | 0.1609 |
| leadx b8 K=80 | 0.1548 | 0.1594 |
| exact kNN oracle (best-k, cos) | 0.1422 | **0.1641** |
| leadx b10 K=40 | 0.1432 | 0.1519 |
| refined (supervised split) b12 K=15 | 0.1315 | 0.1456 |
| leadx b12 K=15 (baseline เดิม) | 0.1244 | 0.1353 |
| mode-aware (4 modes, max) | 0.1186 | 0.1265 |

variance ข้าม draws ต่ำทุกวิธี (std ≤ 0.005) | cosine ≥ euclidean ทุก k

## Decision gate

- 10%: kNN-oracle ชนะ 5/10 draws, −0.7% rel → **ไม่ผ่าน**
- 25%: kNN-oracle ชนะ 10/10 แต่ +8.3% rel (< เกณฑ์ +10%) → **ไม่ผ่าน**
- **ข้อสรุป: geometry headroom เล็ก → ไม่ไป Phase 2A; เส้นทางถัดไปคือ Phase 2B
  (representation/metric)** — สอดคล้องกับหลักฐานก่อนหน้า (supervised split direction
  ชนะ PCA/random; feature_weighting ช่วยจริง)

## Findings เชิงปฏิบัติ

1. **Tables scaling คือกำไรใหญ่ที่ยังไม่ได้เก็บ**: K=15→40→80 (bits ต่ำ) ให้ +24% rel
   PR-AUC เหนือ config เดิม และที่ 10% seeds **ชนะ exact-kNN oracle** — ensemble
   collision kernel smooth กว่า hard-k kNN เมื่อ seeds เบาบาง (ยืนยัน "more tables >
   more bits" จาก literature review) → **ship: เพิ่ม 40 เข้า DEFAULT_GRID + จด
   คำแนะนำ PU: b8–10, K=40–80**
2. **Supervised dense-bucket refinement (PU)**: ช่วย base ตัวเอง (+6–8% rel) แต่ถูก
   tables scaling ครอบ — ไม่คุ้ม implement สำหรับ PU seeds น้อย (เวอร์ชัน two-class
   +0.012 ยังเป็น candidate แต่ลดความสำคัญลง)
3. **Mode-aware scoring (Instagram-inspired): ปฏิเสธ** — แย่กว่า baseline ทุกกรณี
   (การแบ่ง seeds เป็น modes ทำให้หลักฐานต่อ mode บางลงเกินกำไรที่ได้)
4. Annoy/HNSW refs: ข้าม — exact kNN เป็น oracle ที่แข็งกว่าสำหรับคำถาม headroom
   (ANN มีไว้เทียบ latency ซึ่งไม่ใช่คอขวด: leadx hash 80 tables = ~10s ต่อ 246k)

## Provenance & repro (แก้ตาม audit P0)

- **source of truth = raw CSVs** (locality_headroom_raw.csv / phase2b_raw.csv) —
  ตัวเลขในตารางเป็น median ณ เวลารัน; gate verdict ตรวจซ้ำจาก raw ได้เสมอ
- b8k80 อยู่ใน CONFIGS ของ script แล้ว (เดิมอยู่เฉพาะ raw จาก ad-hoc probe)
- ผล combo whiten+contrast และ library-verification เป็น spot-check 3 draws
  (บันทึกใน DESIGN_NOTES/commit log) — ไม่มี raw CSV แยก จัดเป็น evidence ชั้นรอง

## Reproduce

    uv run python outputs/locality_headroom_benchmark.py
    # ผลดิบ: outputs/locality_headroom_raw.csv

## Files changed

- outputs/locality_headroom_benchmark.py (ใหม่ — harness)
- outputs/locality_headroom_raw.csv (ผลดิบทุก draw/method/config)
- src/leadx/tuning.py (DEFAULT_GRID: n_tables += [40])
- tests/test_tuning.py (assertion grid ใหม่)
- README/DESIGN_NOTES (PU config guidance + Phase-1 summary)


---

# Phase 2B — Learned Geometry (ผลและสิ่งที่ ship)

harness เดียวกับ Phase 1 (home-credit PU, 10 draws × 2 fracs), baseline = b10 K80:

| method | 10% seeds | 25% seeds | gate (vs baseline) |
|---|---|---|---|
| **pu_diag (contrast weights, task-aware)** | **0.2011** | **0.2026** | **+30%/+26% rel, 10/10 ← ผ่าน** |
| **pca_rr (whiten, unsupervised-reusable)** | **0.1789** | **0.1952** | **+16%/+21% rel, 10/10 ← ผ่าน** |
| pu_guided (1 bit seed-contrast/table) | 0.1648 | 0.1708 | +6% 10/10 ผ่านแต่ถูก pu_diag ครอบ |
| baseline b10 K80 | 0.1548 | 0.1609 | — |
| itq (PCA-10 + ITQ rotation) | 0.1337 | 0.1375 | −14% ปฏิเสธ (บีบมิติแรงเกิน) |
| agh (anchor embedding 200) | 0.1191 | 0.1211 | −24% ปฏิเสธ |

- combo whiten+contrast: ไม่ stack (0.2005 ≈ pu_diag เดี่ยว) — information ซ้อนกัน
- **คำเตือนสำคัญ: whiten เป็น data-dependent** — synthetic ที่ residual directions เป็น
  noise ล้วน whiten ทำร้าย (0.946→0.813) เพราะขยาย noise ให้เท่า signal → จึงเข้า
  **tuner grid** ("whiten": [False, True]) ให้ตัดสินต่อ dataset ไม่ตั้ง always-on

**Shipped เข้า library:**
- `whiten=True` (ProjectionBank/LeadX/backbone/Tuner grid) — PCA-whitening พับเข้า P
  (`P_eff = W @ Q`) → pipeline/save-load ไม่เปลี่ยน; unsupervised → **ใช้กับ backbone
  reuse ได้**
- `feature_weighting="contrast"` (pu + background) — weights = |mean(seeds) − mean(bg)|
  task-aware (ปิด reuse); library reproduce prototype: 0.2027 ≈ 0.2011

verify library vs prototype (3 draws): baseline 0.1506 / whiten 0.1675 / contrast 0.2027
