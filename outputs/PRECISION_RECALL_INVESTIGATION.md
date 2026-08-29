# จะพัฒนา precision/recall ของ LeadX ได้ยังไง — การสืบสวน (v0.5, 2026-08-29)

**สรุปหน้าเดียว:** วันนี้รัน 6 การทดลอง ตั้งสมมติฐานเชิงกลไก 4 ข้อ **ผิดทั้ง 4 ข้อ**
ผลสุทธิคือ **ยังไม่พบวิธีเพิ่ม precision/recall ที่ ultra-imbalance** — และการวัด
หยุดการเปลี่ยน default สองครั้งที่จะทำให้แย่ลง สิ่งที่ได้คือแผนที่ของ regime ที่
ชัดขึ้นมาก และตัวเลข noise floor ที่ทุกการทดลองต่อจากนี้ต้องอ้างอิง

## 1. คอขวดคือ variance ไม่ใช่ bias

`variance_decomposition.py` — fraud @ 0.3%, 6 reps ต่อ arm:

| แหล่งความแปรปรวน | CV (PR-AUC) | CV (ROC) | spread |
|---|---|---|---|
| **seed draw** (ได้ confirmed positives ชุดไหน) | **18.06%** | 2.68% | 50.4% |
| geometry (`random_state`) + contrast weights | 8.06% | 0.30% | 22.2% |
| geometry อย่างเดียว (ปิด weighting) | 4.94% | 0.75% | 13.2% |

**อ่านสามอย่าง:**
1. ตัวการใหญ่สุดคือ **ได้ seeds ชุดไหนมา** มากกว่า geometry 2.2 เท่า — ทีมสองทีมที่มี
   confirmed set ต่างกันได้ผลต่างกันได้ถึง 50% โดยโมเดลเหมือนกันเป๊ะ
2. `feature_weighting="contrast"` เป็น **shared noise** จริง (ปิดแล้ว CV 8.06 → 4.94)
   เพราะคำนวณครั้งเดียวจาก seeds 61 ตัวแล้วคูณ input ของทุก table — 80 tables
   เฉลี่ยทิ้งไม่ได้ **แต่**มันซื้อความแม่น +13.4% มาด้วย (median .0553 → .0638)
   → เป็น bias-variance trade ไม่ใช่บั๊ก
3. ROC นิ่งมาก (CV 0.3–2.7%) แต่ PR แกว่ง (5–18%) → **การจัดอันดับโดยรวมเสถียร
   แต่หัวแถวเปราะ** ซึ่งคือที่ที่ precision@k อยู่

## 2. variance ยุบตัวเมื่อ seeds เยอะ — และนี่คือ leverage ที่ใหญ่ที่สุด

`seed_count_curve.py` — fraud natural, 5 reps ต่อจุด, n_bits=10 (1,024 buckets):

| n_seeds | seeds/bucket | PR-AUC | **CV(PR)** | p@0.1% | CV(p@0.1%) |
|---|---|---|---|---|---|
| 50 | 0.05 | .1968 | **9.98%** | .656 | **26.1%** |
| 150 | 0.15 | .2296 | 10.84% | .689 | 12.4% |
| 500 | 0.49 | .2636 | 2.86% | .800 | 6.4% |
| 1,500 | 1.47 | .2899 | 2.67% | .844 | 2.2% |
| 4,000 | 3.91 | .2991 | **0.94%** | .844 | 4.0% |

**CV ยุบ 10 เท่า, precision@0.1% +29%** คำทำนาย "CV ≤3% ที่ ≥1 seed/bucket" ตรง

**นัยที่สำคัญที่สุดของทั้งวัน:** ปัญหา variance ที่เห็นใน benchmark ทั้งโปรเจกต์
(seeds 20–61 ตัว) เป็น **artifact ของ regime นั้น** โจทย์จริงที่มี 30k confirmed
positives อยู่คนละที่บนแกนนี้ 500 เท่า และจะไม่เจอปัญหานี้

→ **ทางที่ได้ผลชัดที่สุดไม่ได้อยู่ใน library แต่อยู่ที่จำนวน confirmed positives**

## 3. สมมติฐานที่ตั้งแล้วผิด (4 ข้อ)

| # | สมมติฐาน | หลักฐานที่หักล้าง |
|---|---|---|
| 1 | LCB / variance-aware lift จะช่วย precision@k เพราะหัวแถวถูกขับด้วย singleton bucket | ยังไม่ได้สร้าง — §2 แสดงว่ากลไกนี้หายไปเองที่ seeds เยอะ จึงแก้ปัญหาที่โจทย์จริงไม่มี |
| 2 | n_bits ที่ดีที่สุดเลื่อนตามจำนวน seeds (crossover) | `bits_vs_seeds`: อันดับ 12>10>8 **เหมือนกันเป๊ะ**ทั้ง 150 และ 4,000 seeds (5/5 ทั้งคู่) |
| 3 | จะมีจุดพลิกกลับของ n_bits ที่ 14–20 | `bits_turnover`: ไม่พลิกเลยถึง 20 — ที่ 20 bits = 1M buckets, **0.004 seeds/bucket, 99.6% ว่าง** แต่ยังดีสุด (+5.4% vs 12, 5/5) |
| 4 | hyperplane + bits สูง ควรแทน rpforest เป็น default | `engine_at_ultra`: ที่ ultra-imbalance hyperplane **แพ้** (0.3%: p@0.1% −5.6%/−4.6%; PR −8.9%/−14.6%) |

ข้อ 3 ที่ไม่พลิกน่าจะเพราะ **`hier_depths` + multi-probe + `lift` fallback=1 จัดการ
bucket บางไว้แล้ว** — hier ยืมกำลังจาก parent, multi-probe หา Hamming-1 เพื่อนบ้าน,
fallback=1 แปลว่า "table ไม่มีหลักฐานก็ไม่โหวต" สามอย่างนี้ปลดล็อกให้เพิ่ม resolution
ได้โดยไม่จ่ายค่าปรับ (ยังไม่ได้ทดสอบกลไกนี้ตรงๆ — เป็นคำอธิบาย ไม่ใช่ผลวัด)

## 4. แผนที่ regime ที่ได้ (ผลข้างเคียงที่มีค่าที่สุด)

**อันดับ engine พลิกตาม imbalance ไม่ใช่ตาม seeds** — ทดสอบ 2 seed level ในแต่ละฝั่ง:

| regime | ผู้ชนะ | หลักฐาน |
|---|---|---|
| natural (3.5%), 150 seeds | hyperplane@12–20 | p@0.1% +14.5%/+14.9% vs rpforest@12 (5/5) |
| natural (3.5%), 4,000 seeds | hyperplane@16–20 | p@0.1% +7.9%/+9.2% (4-5/5), PR +3.0% (5/5), fit เร็วกว่า 3.4× |
| **ultra (0.3%), 61 seeds** | **rpforest** | hyperplane@16/20: p@0.1% −5.6%/−4.6% (2/5) |
| **ultra (0.1%), 20 seeds** | **rpforest** (เสมอ) | hyperplane@20 "+25%" แต่ 2/5 = เหรียญโยน |
| **ultra + seeds เยอะ (โจทย์จริง)** | **ไม่ทราบ** | วัดไม่ได้ด้วยข้อมูลที่มี — ดู §5 |

→ **default `rare_pu` (rpforest, n_bits=10) ถูกแล้วสำหรับสิ่งที่มันเล็งไว้** ไม่เปลี่ยน

`n_bits` 10→12 ที่ได้ +2.7–3.9% (5/5) ที่ natural นั้น **ที่ ultra ให้แค่ +2.51% PR
(3/5), p@0.1% 0.0% (2/5)** = อยู่ในกรอบ noise ยังไม่พอจะเปลี่ยน default

## 5. ช่องว่างที่ยังวัดไม่ได้ — และมันคือช่องของ PromptPay พอดี

โจทย์จริงคือ **ultra-imbalance (1:16,700) *และ* 30,000 seeds พร้อมกัน** — มุมนี้
ไม่มีในตารางข้างบน และ **reproduce ในเครื่องไม่ได้**: fraud มี positives 7,348 ตัว
ถ้าจะให้ ratio = 0.1% ต้องมี ~7.3M แถว (ชุดข้อมูลมี 210k หลัง split)

สิ่งที่อนุมานได้ (**อนุมาน ไม่ใช่ผลวัด**): อันดับ engine ไม่เปลี่ยนตาม seeds ภายใน
แต่ละระดับ imbalance (ทดสอบ 150 vs 4,000 ที่ natural, 20 vs 61 ที่ ultra) จึงน่าจะ
คงเป็น rpforest ที่ ultra แม้มี seeds เยอะ — แต่ต้องยืนยันด้วยข้อมูลจริง

## 6. ข้อเสนอ

**ไม่เปลี่ยน default** — หลักฐานไม่พอ และการวัดหยุดสองการเปลี่ยนที่จะทำให้แย่ลง

สิ่งที่ทำได้จริงตอนนี้ เรียงตามผลกระทบ:

1. **ใช้ confirmed positives ให้ครบ** — จาก §2 นี่คือ leverage ที่ใหญ่ที่สุด
   (p@0.1% +29% จาก 50→4,000 seeds) และ PromptPay มี 30k อยู่แล้ว
2. **cascade** (+22.5% Recall@0.1%, 6/6 — `COMBINER_REPORT.md`) ยังเป็นตัวที่วัดแล้ว
   ให้ผลใหญ่สุดในบรรดาสิ่งที่ library ควบคุมได้
3. **วัด noise floor ก่อนเชื่อผลใดๆ** — CV ที่ ultra-imbalance คือ ~8–18% แปลว่า
   effect เล็กกว่านั้นต้องใช้ paired + wins ไม่ใช่ median เดี่ยวๆ
4. ถ้างานไม่ใช่ ultra-imbalance: `hyperplane` + `n_bits` 16–20 ดีกว่าและเร็วกว่า
   3.4× (มีหลักฐาน 4-5/5 สองระดับ seeds) — ควรเขียนเป็น guidance ไม่ใช่ default

## Reproduce
```bash
cd sandbox/data_science/leadx
PYTHONPATH=outputs uv run python outputs/variance_decomposition.py
PYTHONPATH=outputs uv run python outputs/seed_count_curve.py
PYTHONPATH=outputs uv run python outputs/bits_vs_seeds_experiment.py
PYTHONPATH=outputs uv run python outputs/bits_turnover_experiment.py
PYTHONPATH=outputs uv run python outputs/engine_at_ultra_experiment.py
#   → outputs/{variance_decomposition,seed_count_curve,bits_vs_seeds,
#              bits_turnover,engine_at_ultra}_raw.csv
```
