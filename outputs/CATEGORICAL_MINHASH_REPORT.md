# MinHash categorical handling (v0.2, workstream A) — 2026-07

## คำตัดสิน

**MinHash เป็น categorical handler ที่ดีกว่า one-hot/feature-hashing เดิมชัดเจน**
บน IEEE-CIS fraud (PU, 1:1,000, 5 paired draws) — ตัว algorithm รองรับ **feature
ทุกแบบพร้อมกัน**: numeric ผ่าน random-projection/rpforest, categorical ผ่าน
MinHash, ต่อ tables แบบ additive แล้วให้คะแนนด้วย scorer เดียว.

Ablation ที่ตรึง numeric geometry ให้เท่ากันทุก variant (เทียบเฉพาะการเพิ่ม
categorical), paired ต่อ draw (ไม่ใช่ ratio-of-medians):

| คำถาม | PR-AUC | Recall@0.5% | สรุป |
|---|---|---|---|
| **MinHash vs one-hot** | **+55.6% median, 5/5** | **+36.8%, 5/5** | MinHash ดีกว่าทุก draw |
| **one-hot vs numeric-only** | **−38.8%, 0/5** | −13.6%, 2/5 | one-hot *ทำร้าย* |
| **MinHash vs numeric-only** | −0.9%, 2/5 (เสมอ) | **+17.4%, 5/5** | เพิ่ม recall แม้ PR เสมอ |

**อ่านผล:**
1. MinHash ชนะ one-hot **5/5 ทุก draw** (+56% PR) — improvement ที่ consistent
   ที่สุดของโปรเจกต์ (paired ทุก draw, ไม่คร่อมศูนย์)
2. การจัดการ categorical แบบเดิม (one-hot/feature-hashing) **ทำร้ายจริง** —
   แพ้ numeric-only 0/5 บน PR; MinHash กู้ค่านี้กลับมา (+17% recall vs numeric)
3. within-table concat (ต่อบิตใน code เดียว, conjunctive) **แย่สุด** — bucket
   sparse เกินที่ imbalance สูง → additive (separate tables) คือคำตอบ

## กลไก

แต่ละแถว categorical = set ของ token `{column=value}`; MinHash ให้ bucket
collision ∝ **Jaccard** (สัดส่วนคอลัมน์ที่สองแถวตรงกัน). แต่ละ bit = high-bit ของ
`minhash_winner_token × c_kj` → bit-agreement = 0.5·(1+Jaccard), bits อิสระ
(prefix-mask ได้), y-independent (seeds เท่านั้น → composes กับ reusable backbone).
ID-like columns (card1-6) กลายเป็น token แทนถูก z-score เป็น ordinal.

## Baselines (paired, leadx_cat_minhash เทียบ)
- vs max(logreg-PU, xgboost-PU): PR **+31%, 3/5**; Recall@0.5% −15%, 2/5 (mixed)
- xgboost-PU-naive แย่สุดที่ extreme imbalance นี้ (PR 0.012) — struggle จริง

## API (shipped ใน library)

```python
LeadX(mode="pu", tune=False, engine="rpforest",
      cat_engine="minhash",              # เปิด MinHash categorical
      minhash_columns=["ProductCD", "card1", ...],  # None = auto (string cols)
      n_bits_cat=8)
```
numeric engine เห็นเฉพาะ columns ที่ไม่ใช่ minhash; categorical tables ต่อ
additive; hier ปิดอัตโนมัติ (tables bit-width ผสม → flat lift, validated).
`cat_engine="onehot"` (default) = พฤติกรรมเดิมทุกอย่าง. save/load ครบ.
shipped API reproduce ทิศทาง minhash > onehot บน fraud (0.0434 vs 0.0403, 1 draw).

## Caveats
- **dataset เดียว (fraud), 1:1,000** — fraud rows ไม่พอทำ 1:10,000; pure ultra-
  imbalance test อยู่บน Criteo (dense, ไม่มี categorical) แยกกัน
- baseline win แบบ mixed (PR ชนะ, recall แพ้) — ไม่ใช่ clean win เหนือ baseline
- `tune=True` + minhash ยังไม่รองรับ; backbone reuse integration = follow-up
- within-table concat มีในโครง benchmark แต่ปฏิเสธ (additive ดีกว่า)

## Reproduce
```bash
cd sandbox/data_science/leadx
uv run python outputs/minhash_cat.py                       # correctness self-test
uv run --with xgboost,pyarrow,pandas python outputs/fraud_categorical_benchmark.py
#   → outputs/fraud_categorical_raw.csv (source of truth)
```
