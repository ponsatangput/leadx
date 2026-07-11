# leadx — Fast LSH Ensemble Classifier

> **LEAD** = **L**SH **E**nsemble **A**pproximate **D**ensity

Fast classification บนข้อมูลขนาดใหญ่ (ระดับ 10M rows) ด้วย **Locality-Sensitive Hashing (LSH)** + **seed-based bucket scoring** — เหมาะกับงาน lead scoring / PU ranking ที่มี positive ยืนยันไม่กี่ตัวแต่ข้อมูลมหาศาล

แทนที่จะเรียนรู้ decision boundary แบบ model ทั่วไป `leadx` ใช้ LSH แบ่งข้อมูลเป็น bucket แล้วให้คะแนนแต่ละ bucket จากสัดส่วนของ "seeds" (ตัวอย่างที่มี label) ที่ตกลงไปในนั้น — ทำให้ **fit และ predict เป็น batch operation ล้วน** (matrix multiply + group-by) ไม่มี loop ต่อแถว จึง scale ได้ถึงหลักสิบล้านแถวในเวลาไม่กี่วินาที

```
fit 20,000 seeds × 50 features → 0.4s
predict 1,000,000 rows        → 1.4s
```

---

## Installation

Package ใช้ [uv](https://docs.astral.sh/uv/) เป็นตัวจัดการ environment:

```bash
cd sandbox/data_science/leadx
uv run python -c "from leadx import LeadX; print('ok')"
```

Dependencies: `numpy`, `polars`, `scikit-learn` (metrics เท่านั้น)

---

## Quickstart

### Two-class (มี label ทั้งสอง class)

```python
import polars as pl
from leadx import LeadX

# X เป็น Polars DataFrame (numeric และ/หรือ categorical columns)
# y เป็น array ของ 0/1
clf = LeadX(mode="two_class")   # auto-tune เปิดโดย default
clf.fit(X_seed, y)

proba = clf.predict_proba(X_large)      # (n, 2) — proba[:, 1] คือ P(positive)
```

### PU learning (มีแต่ positive ที่ยืนยันแล้ว)

```python
clf = LeadX(mode="pu")
clf.fit(X_positive)                     # ไม่ต้องส่ง y (ถือว่า positive ทั้งหมด)

scores = clf.score_samples(X_large)     # ranking score 0..1 (ไม่ใช่ probability)
```

ถ้ามี background population (unlabeled) ส่งเข้ามาด้วยจะได้ score ที่คุณภาพดีกว่า (seed rate ต่อ bucket แทน log-density):

```python
clf.fit(X_positive, background=X_unlabeled)
```

---

## แนวคิดของ algorithm

### 1. Encoding ด้วย random hyperplane (SimHash)

สุ่ม projection vectors หลายตัว เอาไป dot กับข้อมูลแต่ละแถว แล้วเก็บ **sign** ของผลลัพธ์เป็น bit:

```
bit_j(x) = 1 if (x · p_j) > 0 else 0
```

`B` bits ต่อกันเป็นรหัส bucket หนึ่งตัว (uint64) — จุดที่อยู่ใกล้กันใน feature space มีโอกาสสูงที่จะได้รหัสเดียวกัน (collision probability = `1 − θ/π` เมื่อ θ คือมุมระหว่างสองจุด)

### 2. Ensemble ของ hash tables

ทำซ้ำ `K` ชุด (K tables) ด้วย projection vectors คนละชุด — จุดที่ "ซวย" ถูกระนาบสุ่มผ่ากลางกลุ่มใน table หนึ่ง ยังมีโอกาสอยู่ bucket ที่ถูกใน table อื่น คะแนนสุดท้ายเฉลี่ยข้ามทุก table เพื่อลด variance

### 3. Seed-based scoring

Seeds (ข้อมูลมี label) ถูก hash ลง bucket → แต่ละ bucket เก็บสถิติ `(n_positive, n_total)` เท่านั้น (ไม่เก็บรายชื่อสมาชิก จึงเล็กมาก) คะแนนของ bucket ใช้ **m-estimate smoothing** เพื่อกัน bucket ที่มี seed น้อยได้คะแนนสุดโต่ง:

```
score(bucket) = (n_pos + m·π) / (n_total + m)
```

โดย π = สัดส่วน positive ทั้งชุด, m = smoothing strength — bucket ที่มี seeds มากคะแนนวิ่งเข้าหาสัดส่วนจริง, bucket ที่มี seeds น้อยถูกดึงเข้าหา prior

---

## จุดที่ library แก้จากวิธีดั้งเดิม

| ปัญหาเดิม | ทางแก้ |
|---|---|
| **Tune ยาก** — ต้องสุ่ม vectors หลายรอบกว่าจะได้ชุดที่กระจายตัวดี | `ProjectionBank` คัด vectors อัตโนมัติตามเกณฑ์วัดได้ 5 ข้อ + resample ให้เองจนพอ |
| **เลือกจำนวน bits/tables ไม่มีหลัก** | `Tuner` หา config ที่ดีที่สุดจาก out-of-fold validation — ประเมินทั้ง grid จากการ hash **ครั้งเดียว** (nested prefix masking) |
| **Categorical features ใช้ไม่ได้** | `FeatureEncoder` เลือก one-hot / feature hashing อัตโนมัติตาม cardinality |
| **Bucket ไม่มี seed → ไม่มีคะแนน** | Multi-probe (มอง bucket เพื่อนบ้าน Hamming-1) → prior fallback |
| **Bucket seed น้อย → คะแนนสุดโต่ง** | m-estimate smoothing |

---

## เกณฑ์ "ชุด vectors ที่ดี" (ProjectionBank)

`ProjectionBank` คัด projections โดยวัดจากข้อมูลจริง (บน sample ~100k แถว) — เปลี่ยนจาก "สุ่มแล้วลุ้น" เป็น "คัดอัตโนมัติรอบเดียว":

1. **Bit balance** — แต่ละ bit ต้องผ่าข้อมูลใกล้ครึ่งๆ (minority ≥ 40%) ไม่งั้น bit นั้นแทบไม่มี information
2. **Bit redundancy** — bits คู่ใดก็ตามต้องไม่ correlated เกิน (|ρ| < 0.3) วัดบนข้อมูลจริง ไม่ใช่แค่มุมระหว่าง vectors
3. **Bucket distribution** — effective bucket count (perplexity) สูง ไม่มี mega-bucket กลืนข้อมูลครึ่งชุด
4. **Table diversity** — partitions ของแต่ละ table ต้องต่างกันพอ (NMI ต่ำ)
5. **[supervised mode]** Information gain ต่อ bit เทียบ label — วัดแบบ **cross-fitted** เพื่อกัน overfit; ถ้า seeds < 50/class จะ refuse แล้ว fallback เป็น unsupervised

> ⚠️ **Contract:** `ProjectionBank` คาดหวัง input ที่ **centered/standardized แล้ว** — `LeadX` ทำให้ผ่าน `FeatureEncoder` โดยอัตโนมัติ ถ้าเรียก `ProjectionBank` ตรงๆ ด้วยข้อมูลที่ mean เบ้ เกณฑ์ balance จะ anti-select ทิศทางที่มี offset และคะแนนจะเสียหายรุนแรง (จะมี warning เตือน)

---

## Auto-tuning

```python
clf = LeadX(mode="two_class", tune=True)   # default
clf.fit(X_seed, y)

clf.config_            # config ที่ชนะ: {strategy, n_bits, n_tables, m, weights}
clf.tuning_results_    # Polars DataFrame — ทุก config เรียงตาม OOF score
```

**Trick ที่ทำให้ tuning เร็ว:** hash ด้วย `B_max` bits และ `K_max` tables เพียงครั้งเดียว แล้ว

- **จำนวน bits น้อยลง** = mask เอา prefix bits (`code >> shift`) — bits ถูกเรียงจากคุณภาพสูงสุดไว้แล้ว
- **จำนวน tables น้อยลง** = ใช้ subset แรก

ดังนั้นการลอง 100 configs มีต้นทุนแทบเท่าลอง 1 config

ปิด auto-tune แล้วกำหนดเองก็ได้:

```python
clf = LeadX(tune=False, n_bits=14, n_tables=10, m=10.0)
```

---

## Diagnostics

```python
clf.diagnostics_["projections"]   # balance, redundancy, n_eff, diversity ต่อ table
clf.diagnostics_["scoring"]       # miss_rate_join, miss_rate_final (หลัง predict)
```

ถ้า `miss_rate_final > 30%` จะมี warning — สัญญาณว่า `n_bits` มากไป (bucket ละเอียดจนว่าง) หรือ seeds น้อยไป

---

## Save / Load

```python
clf.save("model_dir")               # projections.npy + tables.parquet + meta.json
clf2 = LeadX.load("model_dir")
```

> Caveat: hashed categorical ใช้ polars hash — โหลดข้าม polars major version อาจได้ hash ต่างกัน (version บันทึกไว้ใน `meta.json`)

---

## Public API

| Method | คำอธิบาย |
|---|---|
| `LeadX(mode, tune, n_bits, n_tables, m, strategy, weights, grid, chunk_size, random_state)` | constructor |
| `.fit(df, y=None, background=None)` | train; PU mode ไม่ต้องส่ง y |
| `.predict_proba(df)` | `(n, 2)` — two_class เท่านั้น |
| `.score_samples(df)` | ranking score `(n,)` — ใช้ได้ทั้งสอง mode |
| `.save(path)` / `.load(path)` | serialize / deserialize |
| `.config_`, `.tuning_results_`, `.diagnostics_` | fitted attributes |

**Parameters สำคัญ:**

- `mode`: `"two_class"` (label สองฝั่ง) หรือ `"pu"` (positive-unlabeled)
- `strategy`: `"unsupervised"` (default) หรือ `"supervised"` (คัด bits ด้วย label; auto-tune ลองทั้งคู่)
- `weights`: `"mean"` (เฉลี่ยตรงๆ) หรือ `"coverage"` (table ที่ miss เยอะพูดเบาลง)
- `chunk_size`: ขนาด chunk ตอน predict (default 1M) — คุม peak memory

---

## Architecture

Modular core + thin facade — ทุก component ทดสอบแยกได้:

```
LeadX (facade)
    │
    ├── FeatureEncoder    numeric standardize + categorical (one-hot / hashing)
    ├── ProjectionBank    คัด projection vectors ตามเกณฑ์ 5 ข้อ
    ├── HashEnsemble      Z @ P → sign → bit-pack เป็น uint64 (K tables)
    ├── SeedScorer        bucket → score (Polars), multi-probe, PU/two-class
    └── Tuner             nested prefix-masked OOF grid search
```

---

## Limitations (v1)

- ยังไม่รองรับ multi-class (> 2 classes) — โครงสร้าง score ต่อ bucket ขยายได้แต่ยังไม่ทำ
- ยังไม่มี GPU / out-of-core (แต่ chunked predict รองรับข้อมูลใหญ่กว่า RAM ได้บางส่วน)
- Data-dependent projections ขั้นสูง (ITQ / PCA-rotate) เป็นทิศทาง v2

---

## Accuracy vs other algorithms

AUC (stratified 5-fold CV) เทียบ baselines — leadx เปิด auto-tune, kNN/LogReg ใส่ StandardScaler:

| dataset | leadx | HistGradBoost | RandomForest | kNN | LogReg |
|---|---|---|---|---|---|
| breast_cancer (real) | 0.991 | 0.994 | 0.992 | 0.991 | **0.995** |
| synthetic linear | 0.974 | **0.992** | 0.987 | 0.989 | 0.889 |
| two_moons (non-linear) | 0.913 | 0.984 | 0.980 | **0.986** | 0.943 |
| synthetic hard | 0.860 | **0.926** | 0.925 | 0.902 | 0.834 |

**สรุป:** leadx แข่งได้สูสีบน dataset ที่แยกได้ดี (breast_cancer) แต่ตามหลัง tree-based
2–6 AUC points บนข้อมูลที่ signal กระจายหรือ boundary โค้ง (two_moons) — เป็น trade-off
ที่คาดไว้: แลกความแม่นเล็กน้อยกับความเร็วและ scalability (predict 1M rows < 1s ที่คู่แข่ง
tree-based/kNN ทำไม่ทัน) จุดคุ้มของ leadx คือข้อมูล**ใหญ่มาก**ที่ latency สำคัญ ไม่ใช่
dataset เล็กที่ accuracy ทุกจุดมีค่า

## Examples

```bash
uv run python examples/worked_example.py   # end-to-end: categorical + missing → fit/tune/predict/diagnostics/save-load
uv run python examples/benchmark.py        # AUC + เวลา เทียบ kNN / LightGBM หลายขนาดข้อมูล
uv run python examples/auc_comparison.py   # AUC เทียบ 5 algorithms บน 4 dataset (5-fold CV)
```

`worked_example.py` สาธิต workflow เต็มบนข้อมูลลูกค้าสังเคราะห์ที่มีทั้ง numeric,
categorical (one-hot + hashing) และ missing values ทั้ง two-class และ PU

`benchmark.py` แสดง trade-off ของ leadx — บนข้อมูลสังเคราะห์ 200k→1M แถว leadx
predict เสร็จในหลักวินาที ส่วน exact kNN ทำไม่ไหว (predict เป็น O(n_train·n_test));
ความแม่นมักต่ำกว่า gradient boosting เล็กน้อย แลกกับความเร็วและ memory footprint

## Development

```bash
uv run pytest tests/ -m "not slow"   # unit + property + integration (เร็ว)
uv run pytest tests/ -m slow -s      # 1M-row scale test
```

เอกสารออกแบบ: [`docs/superpowers/specs/2026-07-10-leadx-design.md`](../../../docs/superpowers/specs/2026-07-10-leadx-design.md)
