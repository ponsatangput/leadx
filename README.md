# leadx — Fast LSH Ensemble Classifier

> **LEAD** = **L**SH **E**nsemble **A**pproximate **D**ensity

Fast classification บนข้อมูลขนาดใหญ่ (ระดับ 10M rows) ด้วย **Locality-Sensitive Hashing (LSH)** + **seed-based bucket scoring** — เหมาะกับงาน lead scoring / PU ranking ที่มี positive ยืนยันไม่กี่ตัวแต่ข้อมูลมหาศาล

แทนที่จะเรียนรู้ decision boundary แบบ model ทั่วไป `leadx` ใช้ LSH แบ่งข้อมูลเป็น bucket แล้วให้คะแนนแต่ละ bucket จากสัดส่วนของ "seeds" (ตัวอย่างที่มี label) ที่ตกลงไปในนั้น — ทำให้ **fit และ predict เป็น batch operation ล้วน** (matrix multiply + group-by) ไม่มี loop ต่อแถว จึง scale ได้ถึงหลักสิบล้านแถวในเวลาไม่กี่วินาที เชิงทฤษฎีมันคือ **approximate soft-kNN**: bucket-density ในละแวกรหัสเดียวกัน ≈ soft kNN ที่แลกความแม่นของ neighbor จริงกับความเร็วของ hash lookup

```
fit 20,000 seeds × 50 features → 0.4s
predict 1,000,000 rows        → 1.4s
```

## สารบัญ

- [When to use leadx](#when-to-use-leadx) — เลือกให้ถูกงาน (สำคัญ อ่านก่อน)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Modes & options](#modes--options)
- [Auto-tuning](#auto-tuning)
- [Backbone reuse](#backbone-reuse--hash-ครั้งเดียว-หลายโจทย์)
- [Diagnostics](#diagnostics)
- [Save / Load](#save--load)
- [API reference](#api-reference)
- [Benchmarks](#benchmarks)
- [Architecture](#architecture)
- [Known limitations & failure modes](#known-limitations--failure-modes)
- [Examples](#examples)
- [Development](#development)

---

## When to use leadx

leadx ไม่ใช่เครื่องมือทั่วไปที่ชนะทุกงาน — มันแลกความแม่นเล็กน้อยกับความเร็ว, ความเรียบง่ายของโครงสร้าง (model = hash tables), และความสามารถ reuse ข้ามโจทย์ ตัดสินใจตามตารางนี้:

| สถานการณ์ | ใช้ leadx? |
|---|---|
| **PU learning** — มีแต่ positive ยืนยันแล้ว + unlabeled ไม่มี labeled negative | ✅ **ใช่** — boosting/LogReg รันไม่ได้เลย นี่คือ niche จริง |
| **หลายโจทย์บน feature space เดียว** (multi-label, ทดลองหลายนิยาม target, re-score รายวัน) | ✅ **ใช่** — `LeadXBackbone` hash ครั้งเดียว แต่ละโจทย์ ~30ms |
| ต้อง **deploy เป็น hash table ธรรมดา** ไม่มี model server | ✅ **ใช่** — model คือ dict lookup |
| ข้อมูล **ใหญ่มาก** ต้องการ predict latency ต่ำ + accuracy พอใช้ | ⚠️ **พิจารณา** — เร็วกว่า kNN มาก แต่ HistGradientBoosting ก็เร็วและแม่นกว่า |
| **two-class ปกติ มี label ครบ** ที่ accuracy ทุกจุดมีค่า | ❌ **ไม่** — ใช้ gradient boosting (แม่นกว่า 2–6 AUC points) |
| **imbalanced สุดโต่ง** (positive < ~0.3%) | ✅ **ใช่** — `score_type="lift"` แบนราบตาม ratio ขณะที่ boosting ทรุด (ตัดกันที่ ~0.3%) |
| **imbalanced ปานกลาง** (1–10%) ที่มี labeled negatives | ❌ **ไม่** — boosting ยังชนะ (PR-AUC สูงกว่า) |

รายละเอียดตัวเลขดู [Benchmarks](#benchmarks) และเหตุผลเชิงลึกดู [Known limitations](#known-limitations--failure-modes)

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

# X เป็น Polars DataFrame (numeric และ/หรือ categorical columns), y เป็น array ของ 0/1
clf = LeadX(mode="two_class")           # auto-tune เปิดโดย default
clf.fit(X_seed, y)
proba = clf.predict_proba(X_large)      # (n, 2) — proba[:, 1] คือ P(positive)
```

### PU learning (มีแต่ positive ที่ยืนยันแล้ว)

```python
clf = LeadX(mode="pu")
clf.fit(X_positive)                     # ไม่ต้องส่ง y (ถือว่า positive ทั้งหมด)
scores = clf.score_samples(X_large)     # ranking score 0..1 (ไม่ใช่ probability)

# มี background population (unlabeled) → score ดีกว่า (seed rate ต่อ bucket แทน log-density)
clf.fit(X_positive, background=X_unlabeled)
```

### Noisy data (feature ที่ไม่เกี่ยวเยอะ) → เปิด feature weighting

```python
clf = LeadX(mode="two_class", feature_weighting="corr")   # ถ่วง feature ตาม |corr กับ y|
clf.fit(X_seed, y)
```

### หลายโจทย์บน rows เดียว → backbone reuse

```python
from leadx import LeadXBackbone

bb = LeadXBackbone(n_bits=16, n_tables=15).fit(X)   # hash ครั้งเดียว
churn = bb.fit_task(y_churn)                         # ~30ms ต่อโจทย์
fraud = bb.fit_task(y_fraud)
churn.predict_proba(X_new)
```

---

## How it works

### 1. Encoding ด้วย random hyperplane (SimHash)

สุ่ม projection vectors หลายตัว dot กับข้อมูลแต่ละแถว เก็บ **sign** เป็น bit:

```
bit_j(x) = 1 if (x · p_j) > 0 else 0
```

`B` bits ต่อกันเป็นรหัส bucket (uint64) — คู่จุดที่ทำมุม θ กันชนกัน (ได้ bit เดียวกัน) ด้วยความน่าจะเป็น `1 − θ/π` จุดใกล้กันจึงมีโอกาสสูงที่จะได้รหัสเดียวกัน

### 2. Ensemble ของ K hash tables

table เดียวหยาบมาก (แต่ละ bucket คือกรวยกว้างผ่าน feature space) — **K tables ต่างหากที่กู้ resolution กลับมา**: สัดส่วน table ที่คู่จุดชนกัน = ตัวประมาณแบบ smooth ของ `(1 − θ/π)^B` → ensemble แปลง partition หยาบหลายอันให้เป็น **similarity kernel ต่อเนื่อง** (นี่คือ soft-kNN) คะแนนสุดท้ายเฉลี่ยข้ามทุก table

### 3. Seed-based scoring

Seeds (ข้อมูลมี label) hash ลง bucket → แต่ละ bucket เก็บสถิติ `(n_positive, n_total)` เท่านั้น (ไม่เก็บสมาชิก จึงเล็กมาก — 10M seeds → ตารางไม่กี่แสนแถว) คะแนน bucket ใช้ **m-estimate smoothing**:

```
score(bucket) = (n_pos + m·π) / (n_total + m)
```

π = สัดส่วน positive ทั้งชุด, m = smoothing strength — bucket ที่มี seeds มากคะแนนวิ่งเข้าหาสัดส่วนจริง, bucket ที่มี seeds น้อยถูกดึงเข้าหา prior (กันคะแนนสุดโต่งที่เป็นแหล่ง variance หลัก)

**แถวที่ตก bucket ไม่มี seed** → multi-probe (มอง bucket เพื่อนบ้าน Hamming-1) → prior fallback

---

## Modes & options

| option | ค่า | ความหมาย |
|---|---|---|
| `mode` | `"two_class"` / `"pu"` | label สองฝั่ง / positive-unlabeled |
| `strategy` | `"unsupervised"` (default) / `"supervised"` | คัด projection vectors แบบไม่ใช้ / ใช้ label (auto-tune ลองทั้งคู่) |
| `weights` | `"mean"` (default) / `"coverage"` | รวม score ข้าม tables แบบเฉลี่ยตรง / ถ่วงให้ table ที่ miss เยอะพูดเบาลง |
| `feature_weighting` | `"none"` (default) / `"corr"` | ถ่วง feature ตาม \|corr กับ y\| ก่อน hash (แก้ noise dilution; two_class เท่านั้น; ปิด reuse) |
| `score_type` | `"prop"` (default) / `"lift"` | สัดส่วน smoothed / density ratio (lift) — lift เป็น **imbalance-invariant**: prior ไม่อยู่ในสูตร, fallback = 1 (เป็นกลาง), สเกลตีความได้ ("เข้มข้นกว่าค่าเฉลี่ย 3 เท่า") |
| `alpha` | float (default 2) | shrink-toward-1 strength ของ lift — กล่องหลักฐานน้อยถูกดึงเข้า 1 (กัน outlier กล่องจิ๋ว) |
| `agg` | `"mean"` (default) / `"log"` | รวม lift ข้าม tables แบบ arithmetic / geometric (NB-style) |
| `m` | float (default 10) | smoothing strength (สำหรับ `"prop"`) |
| `chunk_size` | int (default 1M) | ขนาด chunk ตอน predict — คุม peak memory |

---

## Auto-tuning

```python
clf = LeadX(mode="two_class", tune=True)   # default
clf.fit(X_seed, y)
clf.config_            # config ที่ชนะ: {strategy, n_bits, n_tables, m, weights}
clf.tuning_results_    # Polars DataFrame — ทุก config เรียงตาม OOF score
```

**Trick ที่ทำให้ tuning เร็ว:** hash ด้วย `B_max` bits และ `K_max` tables เพียงครั้งเดียว แล้ว

- **n_bits น้อยลง** = mask เอา prefix bits (`code >> shift`) — bits เรียงจากคุณภาพสูงสุดไว้แล้ว
- **n_tables น้อยลง** = ใช้ subset แรก

การลอง 100 configs จึงมีต้นทุนแทบเท่าลอง 1 config ปิด auto-tune แล้วกำหนดเองได้:

```python
clf = LeadX(tune=False, n_bits=14, n_tables=10, m=10.0)
```

---

## Backbone reuse — hash ครั้งเดียว, หลายโจทย์

จุดแข็งเชิงสถาปัตยกรรมที่ boosting เลียนแบบไม่ได้: projections เป็น **y-independent** → encode+project+hash ครั้งเดียว แล้วแต่ละ labeling ใหม่เหลือแค่ group-by บน codes ที่ cache ไว้

```python
from leadx import LeadXBackbone

bb = LeadXBackbone(n_bits=16, n_tables=15, random_state=0).fit(X)   # แพงครั้งเดียว
churn = bb.fit_task(y_churn)                    # ~30ms — คืน TaskModel
fraud = bb.fit_task(y_fraud)
leads = bb.fit_task(mode="pu", seeds=X_pos, background=X)
churn.predict_proba(X_new)                      # re-hash rows ใหม่ผ่าน projections ร่วม
bb.save(path); LeadXBackbone.load(path)         # แชร์ projections ข้ามทีม/เวลา
```

per-task ปรับ granularity ได้ถูกๆ: `fit_task(y, n_bits=b, n_tables=k)` โดย `b ≤` backbone (prefix mask) และ `k ≤` backbone (subset) — ไม่ต้อง re-hash

**ข้อแลก:** backbone ใช้ unsupervised projections เท่านั้น → ใช้ `feature_weighting` ไม่ได้ (weights ใช้ y → codes task-specific → ปิด reuse) เลือกอย่างใดอย่างหนึ่งตาม use case

---

## Diagnostics

```python
clf.diagnostics_["projections"]   # balance, redundancy, n_eff, diversity ต่อ table
clf.diagnostics_["scoring"]       # miss_rate_join, miss_rate_final (หลัง predict)
```

`miss_rate_final > 30%` → warning (สัญญาณว่า `n_bits` มากไปจน bucket ว่าง หรือ seeds น้อยไป)

---

## Save / Load

```python
clf.save("model_dir")               # projections.npy + tables.parquet + meta.json
clf2 = LeadX.load("model_dir")

bb.save("backbone_dir")             # projections.npy + seed_codes.npy + meta.json
bb2 = LeadXBackbone.load("backbone_dir")
```

> Caveat: hashed categorical ใช้ polars hash — โหลดข้าม polars major version อาจได้ hash ต่างกัน (version บันทึกใน `meta.json`)

---

## API reference

### `LeadX` — facade โจทย์เดียวจบ

```python
LeadX(mode="two_class", tune=True, n_bits=16, n_tables=10, m=10.0,
      strategy="unsupervised", weights="mean", grid=None,
      feature_weighting="none", chunk_size=1_000_000, random_state=0)
```

| method / attr | คำอธิบาย |
|---|---|
| `.fit(df, y=None, background=None)` | train; PU ไม่ต้องส่ง y; background สำหรับ PU |
| `.predict_proba(df) → (n, 2)` | class probability — two_class เท่านั้น |
| `.score_samples(df) → (n,)` | ranking score — ทั้งสอง mode |
| `.save(path)` / `LeadX.load(path)` | serialize / deserialize |
| `.config_` | config ที่ใช้จริงหลัง fit |
| `.tuning_results_` | Polars DataFrame ทุก config (None ถ้า tune=False) |
| `.diagnostics_` | projection quality + scoring miss rate |
| `.feature_weights_` | น้ำหนักต่อ feature (None ถ้า feature_weighting="none") |

### `LeadXBackbone` — hash ครั้งเดียว reuse หลายโจทย์

```python
LeadXBackbone(n_bits=16, n_tables=15, random_state=0, chunk_size=1_000_000)
```

| method / attr | คำอธิบาย |
|---|---|
| `.fit(df)` | encode + unsupervised projections + cache seed codes |
| `.fit_task(y=None, mode="two_class", m=10.0, weights="mean", n_bits=None, n_tables=None, seeds=None, background=None) → TaskModel` | สร้างโมเดลต่อโจทย์จาก cached codes (ถูกมาก) |
| `.transform(df) → (n, n_tables) uint64` | hash rows ใหม่ผ่าน projections เดิม |
| `.save(path)` / `LeadXBackbone.load(path)` | serialize / deserialize |
| `.diagnostics_` | projection quality |

### `TaskModel` — โมเดลต่อโจทย์ (คืนจาก `fit_task`)

| method / attr | คำอธิบาย |
|---|---|
| `.predict_proba(df) → (n, 2)` | two_class เท่านั้น |
| `.score_samples(df) → (n,)` | ทั้งสอง mode |
| `.backbone` | reference ไป backbone ที่แชร์ projections |

---

## Benchmarks

### Accuracy vs other algorithms (5-fold CV AUC)

leadx เปิด auto-tune, kNN/LogReg ใส่ StandardScaler:

| dataset | leadx | HistGradBoost | RandomForest | kNN | LogReg |
|---|---|---|---|---|---|
| breast_cancer (real) | 0.991 | 0.994 | 0.992 | 0.991 | **0.995** |
| synthetic linear | 0.974 | **0.992** | 0.987 | 0.989 | 0.889 |
| two_moons (non-linear) | 0.913 | 0.984 | 0.980 | **0.986** | 0.943 |
| synthetic hard | 0.860 | **0.926** | 0.925 | 0.902 | 0.834 |

สูสีบน dataset ที่แยกได้ดี ตามหลัง tree-based 2–6 points บนข้อมูล signal กระจาย/boundary โค้ง

### Large + imbalanced (PR-AUC — เหมาะ imbalanced กว่า ROC-AUC)

| train / test | metric | **leadx** | HistGradBoost | LogReg |
|---|---|---|---|---|
| 100k / 500k (2% pos) | PR-AUC | 0.21 | **0.63** | 0.13 |
| 500k / 1M (1% pos) | PR-AUC | 0.23 | **0.50** | 0.20 |
| 500k / 1M | fit time | 18s | **2s** | 0.2s |

> ⚠️ บน imbalanced ที่มี labeled negatives — **gradient boosting ชนะขาด** leadx niche จริงคือ **PU** (ไม่มี labeled negative ให้ boosting ใช้): จัดอันดับ rare positives ได้ PR-AUC 0.23 **เท่ากับตอนมี label ครบ** เพราะ density approach เป็น PU โดยธรรมชาติ

### Imbalance invariance ของ `score_type="lift"`

ตรึง positive สัมบูรณ์ 2,000 ตัว แล้วไล่ imbalance 100 เท่า (test balanced, ROC-AUC) —
[examples/imbalance_invariance.py](examples/imbalance_invariance.py):

| pos rate | leadx lift(log) | HistGradBoost |
|---|---|---|
| 10% | 0.880 | **0.909** |
| 1% | 0.889 | 0.901 |
| 0.3% | **0.889** | 0.879 |
| 0.1% | **0.883** | 0.846 |

**leadx แบนราบตลอด** (prior ไม่อยู่ในสูตร lift) ขณะ boosting ทรุดตาม imbalance —
**จุดตัด ~0.3%**: ต่ำกว่านั้น leadx ชนะแม้ใน two-class ที่มี label ครบ นี่คือ niche
ที่สองของ leadx (คู่กับ PU): **rare-positive สุดโต่ง**

### Noise features & `feature_weighting`

signal 8 features คงที่ เติม noise จน D โต:

| D | leadx ดิบ | `feature_weighting="corr"` | HistGB |
|---|---|---|---|
| 50 | 0.927 | 0.974 | 0.990 |
| 100 | 0.889 | 0.976 | 0.990 |
| 200 | 0.791 | **0.973** | 0.989 |

feature weighting ปิด gap ~95% ด้วยต้นทุนแทบศูนย์ (แต่ปิด reuse — ดู [failure modes](#known-limitations--failure-modes))

### Backbone reuse (200k × 40 features, 6 โจทย์บน rows เดียว)

| | เวลา | AUC |
|---|---|---|
| `backbone.fit` (ครั้งเดียว) | 2.5s | — |
| `fit_task` (ต่อโจทย์) | **36ms** | 0.94–0.97 |
| HistGradBoost (fit เต็มต่อโจทย์) | 1.3s | 0.996 |

boosting เทรนใหม่ทั้งต้นทุกครั้งที่ y เปลี่ยน — leadx marginal cost ต่อโจทย์แทบเป็นศูนย์

---

## Architecture

Modular core + thin facade — ทุก component ทดสอบแยกได้:

```
LeadX (facade โจทย์เดียว)         LeadXBackbone (reuse หลายโจทย์)
    │                                 │  ├── fit() → cache codes
    ├── FeatureEncoder    numeric standardize + categorical (one-hot / hashing)
    ├── ProjectionBank    คัด projection vectors ตามเกณฑ์ (unsupervised/supervised)
    ├── HashEnsemble      Z @ P → sign → bit-pack เป็น uint64 (K tables); prefix-mask
    ├── SeedScorer        bucket → score (Polars), multi-probe, PU/two-class
    └── Tuner             nested prefix-masked OOF grid search
```

---

## Known limitations & failure modes

จุดอ่อนที่ **วัดจริงแล้ว** (รายละเอียด + ตัวเลขเต็มใน [DESIGN_NOTES.md](DESIGN_NOTES.md)):

1. **Noise-feature dilution (เพดาน accuracy จริง)** — random projection คลุกทุก feature เท่ากัน; noise เยอะ → AUC ทรุด (0.99→0.79 เมื่อ D=8→200) แก้ด้วย `feature_weighting="corr"` (แลก reuse) หรือใช้ boosting
2. **Coverage collapse ในมิติสูง** — B มากไป → bucket ว่าง → miss rate พุ่ง → score หล่นเป็น prior (จัดการได้ด้วย tuner ที่เลือก B ในโซนปลอดภัย)
3. **Seed ต้องเป็นตัวแทน distribution จริง** — score = density ratio; selection bias / covariate shift ทำให้เพี้ยน (miss rate ช่วยตรวจ shift)
4. **Calibration** — score เป็น ranking ที่ monotone-ish ไม่ใช่ P(y|x) จริง (ดีสำหรับ AUC/AP; ถ้าต้อง probability ให้ต่อ isotonic บน OOF)
5. **ยังไม่รองรับ** multi-class (>2), GPU/out-of-core, data-dependent projections (ITQ/PCA) — เป็นทิศทาง v2

---

## Examples

```bash
uv run python examples/worked_example.py   # end-to-end: categorical + missing → fit/tune/predict/diagnostics/save-load
uv run python examples/benchmark.py        # AUC + เวลา เทียบ kNN หลายขนาดข้อมูล
uv run python examples/auc_comparison.py   # AUC เทียบ 5 algorithms บน 4 dataset (5-fold CV)
uv run python examples/large_imbalanced.py # large + imbalanced: PR-AUC + PU niche demo
uv run python examples/backbone_reuse.py   # hash ครั้งเดียว, reuse หลายโจทย์ (~30ms/โจทย์)
```

---

## Development

```bash
uv run pytest tests/ -m "not slow"   # unit + property + integration (เร็ว)
uv run pytest tests/ -m slow -s      # 1M-row scale test
```

เอกสารเพิ่มเติม:
- [DESIGN_NOTES.md](DESIGN_NOTES.md) — failure modes, design tensions, สิ่งที่วัดจริง
- [design spec](../../../docs/superpowers/specs/2026-07-10-leadx-design.md) — สเปกออกแบบเดิม
