# Background sample — ต้องใหญ่แค่ไหน (v0.5, 2026-08)

**คำถาม:** โจทย์ 500M ต้อง fit บน background sample คำถามคือเล็กได้แค่ไหน

**คำตอบ: เล็กได้มาก — 5% (10k แถว) ให้ผลที่แยกจาก background เต็ม (203k) ไม่ออก
ด้วยซ้ำ และ fit เร็วขึ้น 5.9 เท่า** แต่กว่าจะรู้ต้องแก้การทดลองที่ออกแบบผิดก่อน

## ทำไมการ sample ถึงถูกต้องตั้งแต่แรก

สูตร lift ใช้ background เป็น**สัดส่วนมวล** ไม่ใช่ count ดิบ ([scoring.py:235](../src/leadx/scoring.py)):

```
lift(b) = (n_seed_b + α) / (n_seed_total · n_total_b / n_bg_total + α)
                                          ^^^^^^^^^^^^^^^^^^^^^^
```

uniform sample ประมาณ `n_total_b / n_bg_total` ได้แบบ **unbiased** — ที่เสียคือ
*ความแปรปรวน* ของ bucket ที่มีมวลน้อย และ geometry/encoder stats ที่ fit จาก
background ตาม parity contract

## รอบแรก: การทดลองที่ตอบคำถามไม่ได้ (บันทึกไว้เป็นบทเรียน)

`background_sample_experiment.py` กวาด frac ∈ {1, 2.5, 5, 10, 25, 50}% × 3 cells ×
3 draws เทียบกับ background เต็ม ผลคือ **ไม่มี fraction ใดผ่าน gate** — แต่ผลก็
**ไม่ monotone กับขนาด sample เลย** ซึ่งเป็นไปไม่ได้ถ้าขนาด sample เป็นตัวขับ:

| cell | 1% | 2.5% | 5% | 10% | 25% | 50% |
|---|---|---|---|---|---|---|
| fraud @ 0.1% | −4.9 | +1.9 | **+14.1** | −13.1 | **+20.2** | −1.2 |
| fraud @ 0.3% | −35.3 | −4.2 | −16.2 | −7.4 | −4.2 | −11.0 |
| home_credit @ 0.3% | — | −8.1 | −11.8 | −7.2 | −16.9 | −0.5 |

**Confound ที่ระบุได้:** RPForest fit **บน background** (parity contract) ดังนั้น
เปลี่ยน background sample = เปลี่ยน *geometry realization* ไปด้วย การทดลองนี้จึงวัด
sample-size effect กับ geometry-seed noise ปนกันโดยไม่มีทางแยก ซ้ำร้าย baseline ที่
fraud @ 0.1% แกว่งเอง **13 เท่า**ข้าม draw (pr_auc 0.0019 / 0.0112 / 0.0256)

**gate ที่ pre-register ไว้ (median > −5%, draw แย่กว่า −15% ≤ 1) จึงเป็นเกณฑ์ที่
สอบไม่ผ่านโดยโครงสร้าง** — noise floor กว้างกว่าเกณฑ์ ผมตั้ง gate ก่อนวัด noise floor
ซึ่งเป็นลำดับที่ผิด

## รอบสอง: วัด noise floor ตรงๆ

`background_sample_control.py` บน cell ที่ baseline เสถียรที่สุด (fraud @ 0.3%,
baseline 0.064/0.059/0.050), seeds ตรึงชุดเดียว, 6 reps ต่อ arm:

| arm | median pr_auc | min..max | **CV** | **spread** | fit |
|---|---|---|---|---|---|
| `full` — bg เต็ม 202,857 แถว, เปลี่ยนแค่ `random_state` 0..5 | .0638 | .0596–.0738 | **8.06%** | **22.2%** | 63.2s |
| `sampled` — bg 5% = 10,143 แถว, sample seed 6 แบบ, rs ตรึง | .0617 | .0507–.0633 | **8.18%** | **20.4%** | **10.7s** |

- **CV เท่ากันแทบเป๊ะ (8.06 vs 8.18%)** — การ sample เหลือ 5% ไม่เพิ่ม noise
  เกินกว่าที่การเปลี่ยน random seed เฉยๆ ทำอยู่แล้ว
- median ต่างกัน **−3.3%** ซึ่งเล็กกว่า CV ของแต่ละ arm เอง
- **fit เร็วขึ้น 5.9×** (63.2s → 10.7s)

→ ทุกตัวเลขในตารางรอบแรก**อยู่ในกรอบ noise ทั้งหมด** ยกเว้นค่าเดียว

## สิ่งที่สรุปได้จริง

**พอ:** 5% / ~10k แถว (ที่ `n_bits=10` → 1,024 buckets/table ≈ **10 แถวต่อ bucket**)
— แยกจาก background เต็มไม่ออก

**ไม่พอ:** 1% / ~2k แถว (≈2 แถวต่อ bucket) — median −26.6%, 4/6 draws แย่กว่า −15%
ค่านี้อยู่**นอก**กรอบ noise (±~11% รอบ median) จึงเป็นสัญญาณจริง

**ตัวแปรที่น่าจะกำกับคือ "แถวต่อ bucket" ไม่ใช่ "% ของประชากร"** — background ต้อง
มีแถวพอให้ประมาณมวลต่อ bucket ได้เสถียร คือ `n_bg` เทียบกับ `2^n_bits` **ยังไม่ได้
วัดการ scale ของกฎนี้** (วัดที่ `n_bits=10` จุดเดียว) ถือเป็นสมมติฐานเชิงกลไก
ไม่ใช่ผลวัด

## สำหรับ 500M

ไม่ต้องใช้ 500M เป็น background — sample หลักแสนแถวเกินพอสำหรับ `n_bits` ปกติ
(10–12 → 1k–4k buckets) และให้ margin เหลือเฟือเทียบ 10 แถว/bucket ที่วัดว่าพอแล้ว
คอขวดที่แท้จริงของ 500M คือ**ขั้น score** ซึ่งแก้ด้วย streaming
([STREAMING_SCALE_REPORT.md](STREAMING_SCALE_REPORT.md)) ไม่ใช่ขั้น fit

## ขอบเขต

- noise floor วัดที่ cell เดียว (fraud @ 0.3%), 6 reps ต่อ arm, `n_bits=10`,
  `engine="rpforest"` — ไม่ได้กวาดข้าม n_bits/engine
- CV 8% ที่วัดได้เป็นของ config นี้ ไม่ใช่ค่าคงที่ของ library
- ข้อสรุป "1% ไม่พอ" มาจาก 6 observations ของการทดลองรอบแรกที่ confounded —
  ทิศทางแรงพอจะเชื่อ แต่ยังไม่ได้ re-run แบบ controlled

## Reproduce
```bash
cd sandbox/data_science/leadx
PYTHONPATH=outputs uv run python outputs/background_sample_experiment.py   # รอบแรก (confounded)
PYTHONPATH=outputs uv run python outputs/background_sample_control.py      # noise floor
#   → outputs/background_sample_raw.csv, outputs/background_sample_control_raw.csv
```
