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
| 3% | 655 | .092 / .741 | .074 / .687 | .071 / .701 | .077 / .716 |
| 1% | 208 | .032 / .724 | .018 / .617 | **.021 / .683** | .028 / .710 |
| 0.3% | 62 | .0074 / .676 | .0035 / .518 | **.0047 / .599** | .0069 / .672 |

(logreg_pu_naive แพ้ทุกจุดยกเว้น fraud 3%/1% ที่สูสี leadx — ดู raw)

## ข้อค้นพบ

1. **Crossover ที่ข้อมูลเท่ากันมีจริงบนข้อมูลจริง** — LeadX ชนะ PU-naive เมื่อ
   imbalance รุนแรง: fraud 0.1% ชนะ 3.5x บน PR (.0086 vs .0025), lift@0.1% 57x
   vs 11x ของ base rate; home-credit ชนะตั้งแต่ 1% ลงไป จุดตัดขึ้นกับ dataset:
   home-credit ~1%, fraud ~0.2% (fraud มี interaction แรงที่ trees ใช้ประโยชน์ได้
   ตราบใดที่ positives พอ)
2. **โซน imbalance ปานกลาง (≥1% + seeds หลักร้อย) HistGB PU-naive ยังชนะ**
   บน fraud (.378 vs .266 ที่ 3%) — LeadX ไม่ใช่คำตอบทุกโจทย์ imbalance
3. **ไม่ชนะ oracle ที่จูนแล้วที่ไหนเลย** — ผลรอบแรก (oracle default l2=0 พัง)
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

## Reproduce

```bash
cd sandbox/data_science/leadx
PYTHONPATH=outputs uv run python outputs/imbalance_focus_benchmark.py
# raw: outputs/imbalance_focus_raw.csv (source of truth)
```
