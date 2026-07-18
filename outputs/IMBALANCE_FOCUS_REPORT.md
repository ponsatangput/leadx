# Imbalance-focus benchmark — LeadX ดีกว่าวิธีอื่นตรงไหนกันแน่ (2026-07-18)

## คำถามและวิธีวัด

audit's extreme_imbalance_benchmark เทียบ LeadX PU (เห็นแค่ positives) กับ HistGB
label-เต็ม → วัด "ราคาของการไม่มี negatives" ปนกับ "ทน imbalance" รายงานนี้แยกคู่เทียบ:

- **ข้อมูลเท่ากัน** (seeds = 10% ของ positives, ไม่มี negative label):
  `leadx_pu_10pct` (config deploy: b10 K80 lift/log contrast, parity) vs
  `histgb_pu_naive` (unlabeled=negative + balanced — มาตรฐานที่คนใช้จริง) vs
  `logreg_pu_naive`
- **จุดอ้างอิง**: `histgb_oracle_full` (label เต็ม — เพดาน) และ `leadx_pu_allseeds`
  (positives 100% — setting ของ audit)

**ลักษณะการทดลอง:** rarity stress test แบบ semi-synthetic — features จริง แต่
ปรับ prevalence ด้วยการ downsample positives; **fixed split เดียว** (sample_ratio
seed 42/43), 3 seed draws เฉพาะโมเดลที่ขึ้นกับ draw, forest random_state คงที่

HistGB ทุกตัวใช้ hyperparameters ที่ audit จูน (lr .08, iters 150, **l2=1.0**) —
สำคัญมาก: default l2=0 ทำ oracle พังที่ 0.1% (ROC 0.53 ทุก random_state) และจะได้
ข้อสรุปกลับด้าน fraud = temporal split; 3 draws (median); oracle เช็ค variance
เพิ่ม 3 random_state ที่จุดสุดขั้ว (PR 0.055–0.067 ที่ 0.1% — เสถียร)

## ผล (median PR-AUC / ROC-AUC)

### fraud (IEEE, temporal)

| ratio | n_seeds | oracle เต็ม | histgb PU-naive | **leadx 10%** | leadx allseeds |
|---|---|---|---|---|---|
| 3% | 626 | .474 / .890 | **.378 / .844** | .266 / .793 | .277 / .805 |
| 1% | 204 | .318 / .873 | **.175 / .819** | .110 / .782 | .143 / .799 |
| 0.3% | 61 | .159 / .865 | .037 / .717 | .023 / **.763** | .070 / .778 |
| 0.1% | 20 | .055 / .840 | .0025 / .657 | **.0086 / .729** | .044 / .768 |

### home-credit

| ratio | n_seeds | oracle เต็ม | histgb PU-naive | **leadx 10%** | leadx allseeds |
|---|---|---|---|---|---|
| 3% | 298 | .092 / .741 | .074 / .687 | .071 / .701 | .077 / .716 |
| 1% | 97 | .032 / .724 | .018 / .617 | **.021 / .683** | .028 / .710 |
| 0.3% | 29 | .0074 / .676 | .0035 / .518 | **.0047 / .599** | .0069 / .672 |

(logreg_pu_naive แพ้ทุกจุดยกเว้น fraud 3%/1% ที่สูสี leadx — ดู raw)

## ข้อค้นพบ

1. **Crossover ที่ข้อมูลเท่ากันมีจริงบนข้อมูลจริง** — LeadX ชนะ PU-naive เมื่อ
   imbalance รุนแรง: fraud 0.1% ชนะ 3.5x บน PR (.0086 vs .0025), lift@0.1% 57x
   vs 11x ของ base rate; home-credit ชนะตั้งแต่ 1% ลงไป จุดตัดขึ้นกับ dataset:
   home-credit ~1%, fraud ~0.2% (fraud มี interaction แรงที่ trees ใช้ประโยชน์ได้
   ตราบใดที่ positives พอ)
2. **โซน imbalance ปานกลาง (≥1% + seeds หลักร้อย) HistGB PU-naive ยังชนะ**
   บน fraud (.378 vs .266 ที่ 3%) — LeadX ไม่ใช่คำตอบทุกโจทย์ imbalance
3. **ไม่ชนะ oracle ที่จูนแล้วที่ไหนเลย** *(ผลยุค bank config — superseded บางส่วน
   โดย update rpforest ด้านล่าง: ชนะที่ HC 0.3% เฉพาะ allseeds setting)* —
   ผลรอบแรก (oracle default l2=0 พัง)
   เป็นการชนะหุ่นฟาง แต่ gap แคบลงเมื่อ imbalance รุนแรงขึ้น: home-credit
   allseeds = 84→94% ของ oracle (3%→0.3%); fraud = 58→69% — "ได้เกือบเท่า
   label เต็มโดยไม่มี negative label สักตัว" คือ claim ที่ยืนได้
4. **Degrade นุ่มนวลที่สุดในกลุ่มข้อมูลเท่ากัน** (fraud ROC, 3%→0.1% = positives
   หาย 30x): leadx allseeds −0.037, oracle(label เต็ม) −0.050, PU-naive −0.187,
   logreg −0.227 — คุณสมบัติ imbalance-invariant ของ lift โผล่บนข้อมูลจริง
5. config ใหม่ (K80+contrast) ที่ allseeds บน fraud ≈ เท่า config เก่าของ audit
   (K40) — สอดคล้อง §8b ว่า contrast ช่วยที่ seeds น้อย ไม่ใช่ allseeds

## Caveats

- single split ต่อ ratio (sample_ratio seed 42/43); 3 draws เฉพาะโมเดลที่ขึ้นกับ
  seed draw; oracle/allseeds fit เดียว (เช็ค variance แยกแล้วที่จุดสุดขั้ว)
- PU-naive คือ baseline "ที่คนทำจริง" แต่ไม่ใช่ PU method ที่แข็งแรงสุดใน
  literature (เช่น Elkan-Noto calibration) — เทียบกับพวกนั้นยังไม่ได้ทำ
- PR-AUC สัมบูรณ์ที่ 0.1% ต่ำทุกโมเดล (base rate 0.001) — ตัวชี้วัดใช้งานจริงคือ
  lift@0.1% (คิว investigation): leadx 46–57x ด้วย 20 seeds
- config ที่วัด (contrast + rpforest fit ต่อโจทย์) เป็น **task-specific geometry**
  — ไม่ใช่ reusable backbone ของ scale benchmark; จุดขาย rare-event กับ
  build-once/reuse จึงพิสูจน์**แยกกัน** ยังไม่มีหลักฐานว่าเกิดพร้อมกัน

## Update: rerun ด้วย engine="rpforest" (หลัง §8f — config ธงใหม่)

เงื่อนไขเดิมทุกอย่าง (draw seeds เดียวกัน) เปลี่ยนแค่ leadx → rpforest+hier
(raw: imbalance_focus_rpf_raw.csv):

| cell | bank 10% | **rpf 10%** | PU-naive | ผลที่เปลี่ยน |
|---|---|---|---|---|
| fraud 0.3% | .0235 | **.0595** | .0366 | เดิมแพ้ PU-naive → **ชนะ 1.6x**; lift@0.1% 83x vs 61x |
| fraud 0.1% | .0086 | **.0112** | .0025 | ชนะเดิม → กว้างขึ้น (4.5x median) — แต่ชนะ 2/3 draws (d0 แพ้ .0019 vs .0020) และ variance สูงมาก (.0019–.0256) |
| fraud 3%/1% | .266/.110 | .267/.125 | .378/.175 | ยังแพ้ (positives พอให้ trees) |
| HC 3% | .0711 | **.0784** | .0737 | เดิมแพ้นิด → **ชนะ** |
| HC 1% | .0212 | **.0245** | .0183 | ชนะกว้างขึ้น |
| HC 0.3% | .0047 | **.0066** | .0035 | ชนะกว้างขึ้น |

**สรุปที่ข้อมูลเท่ากัน: ชนะ PU-naive 5/7 regimes** — home-credit ทุก ratio ที่วัด
(รวม 3%); fraud 0.3%/0.1% (จุดตัดขยับจาก ~0.1% เป็น 0.3%); **แพ้ fraud 3%/1%**
— claim ที่ปลอดภัย: "outperforms selected PU-naive baselines in some extreme
few-seed regimes" บน fixed split เดียว ยังไม่ generalize

**vs oracle label เต็ม — เฉพาะ allseeds setting, split เดียว**: HC = 91% / 95% /
**137%** ของ oracle (3% / 1% / 0.3%) — ที่ 0.3% allseeds ชนะ oracle ที่จูนแล้ว
(.0102 vs .0074; เกิน max ของ oracle ทุก random_state ที่วัดไว้ .0043-.0088)
แต่ที่ **10% seeds (29 ตัว) ยังแพ้ oracle** (.0066 vs .0074) — ทุกการอ้าง
"ชนะ oracle" ต้องติด qualifier "allseeds, one fixed split"; fraud allseeds 0.1% ต่ำกว่า bank
(.029 vs .044) — **ปิดแล้ว**: decomposition ({engine}×{hier}×3 seeds,
rpf_anomaly_raw.csv) ชี้ว่าเป็นผลของ hier ที่ dilute เมื่อ seeds หลักร้อย
(fraud) + seed variance ใหญ่ ไม่ใช่ข้อบกพร่องของ forest (rpf_flat .038 ≈
bank_flat .041); แนวใช้งาน: hier เปิดเมื่อ seeds น้อย ปิดเมื่อ seeds มาก

## Reproduce

```bash
cd sandbox/data_science/leadx
# ตารางหลัก (bank + baselines):
PYTHONPATH=outputs uv run python outputs/imbalance_focus_benchmark.py
#   → outputs/imbalance_focus_raw.csv
# update section (rpforest, draw seeds เดียวกัน):
PYTHONPATH=outputs uv run python outputs/imbalance_focus_rpf.py
#   → outputs/imbalance_focus_rpf_raw.csv
# paired rpf-vs-bank (gate ของ §8f): outputs/rpforest_experiment.py
#   → outputs/rpforest_raw.csv
# anomaly decomposition: outputs/rpf_cov_experiment.py → outputs/rpf_anomaly_raw.csv
# (raw CSV = source of truth ของทุกตาราง)
```
