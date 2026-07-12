# leadx — Design Notes & Failure Modes

เอกสารนี้เก็บบทวิเคราะห์เชิงลึกที่ได้จากการออกแบบ + วัดผลจริง — เหตุผลเบื้องหลังการตัดสินใจ, จุดอ่อนที่พิสูจน์แล้ว, และ trade-off ที่หลีกไม่ได้ อ่านคู่กับ [README.md](README.md)

## 1. algorithm นี้คืออะไร (แก่นที่ต้องเข้าใจให้ตรง)

leadx = **approximate soft-kNN** ผ่าน LSH collision kernel:

- คู่จุดที่ทำมุม θ ชนกันใน 1 bit ด้วยความน่าจะเป็น `1 − θ/π` (SimHash / Goemans–Williamson) → ชนครบ B bits ใน 1 table = `(1 − θ/π)^B`
- **1 table หยาบมาก** — แต่ละ bucket คือกรวย (cone) กว้างผ่าน feature space
- **K tables กู้ resolution กลับมา**: สัดส่วน table ที่คู่จุดชนกัน = ตัวประมาณแบบ smooth ของ `(1 − θ/π)^B` (variance ~ `1/K`) → ensemble แปลง hard partition หลายอันให้เป็น **similarity kernel ต่อเนื่อง**
- density ของ same-label seeds ในละแวกรหัส = score

**สัจพจน์:** สมาชิกใน bucket เดียวกันต้องคล้ายกัน (locality) — นี่คือรากฐาน ไม่ใช่สิ่งที่ต้องพิสูจน์ หน้าที่เดียวของ LSH คือทำให้ near→ชนกัน, far→ไม่ชน

## 2. "ชุด vectors ที่ดี" วัดจากอะไร — และบันไดของ proxy

สิ่งที่อยากได้จริงคือ "partition ที่ทำให้ seed-density ทำนาย y ได้ดี" แต่มันขึ้นกับ y (reuse ไม่ได้) เราจึงไต่ลงมาเป็น proxy ทีละชั้น:

```
task AUC (y-dependent, จริงที่สุด)          ← เป้าหมายจริง แต่ reuse ไม่ได้
  ↑ อนุมาน: geometric locality ⇒ label locality   ← สมมุติฐานหลักของ LSH (ตรวจไม่ได้ถ้าไม่มี y)
locality fidelity (near ชนกัน, far ไม่ชน)    ← ชั้นที่ "ดี" จริงและ y-agnostic
  ↑ อนุมาน: occupancy สม่ำเสมอ ⇒ locality ดี      ← ผิด (ดู §3)
n_eff / coverage (bucket distribution)       ← เห็นแค่ granularity ไม่เห็นว่าใครอยู่กับใคร
  ↑ อนุมาน: bits balanced + independent ⇒ occupancy ดี  ← ผิดถ้ามี higher-order dependence
bit balance + pairwise redundancy            ← เกณฑ์ที่ ProjectionBank ใช้จริง (ชั้นล่างสุด)
```

**บทเรียนสำคัญ:** ยิ่งไต่ลงชั้นล่าง ยิ่งวัดถูก/ถูกลง แต่ยิ่งเป็น proxy ที่ห่างจากเป้าหมายจริง เกณฑ์ bit-level ที่ ProjectionBank ใช้มีค่าหลักๆ ตรงที่มัน **บังคับให้อยู่ในตระกูล hyperplane** ซึ่งโครงสร้างของมันการันตี locality ให้ฟรี ไม่ใช่ที่ตัวเลข balance เอง

### n_eff คืออะไร

Effective number of buckets = `exp(H)` โดย `H = −Σ q_b ln q_b` (q_b = สัดส่วนข้อมูลใน bucket b)

| การกระจาย (B=2, 4 buckets) | N_eff |
|---|---|
| 25/25/25/25 | 4.0 (ใช้ครบ) |
| 40/30/20/10 | 3.4 |
| 97/1/1/1 | 1.2 (เหมือนมี bucket เดียว) |

## 3. หลุมพราง: n_eff สูง ≠ partition ดี (พิสูจน์แล้ว)

| partition | n_eff (เฉลี่ย/table) | test AUC |
|---|---|---|
| hyperplane LSH | 1,210 | **0.981** |
| random assignment (ทำลาย locality) | 3,888 (เกือบ max 4,096) | **0.497** |

random partition มี n_eff **สูงกว่า 3 เท่า** แต่ AUC = 0.5 — เพราะ **n_eff มองไม่เห็น locality** มันวัดแค่ "ขนาด bucket เบ้ไหม" ไม่รู้ว่าใครอยู่ bucket ไหน → **ห้ามใช้ n_eff เป็น objective** มันเป็นได้แค่ guardrail กัน mega-bucket (score หยาบ) และ bucket ว่าง (ไม่มี seed)

## 4. Failure modes ที่วัดจริง

### #1 Noise-feature dilution — เพดาน accuracy จริง

random projection `w·x` คลุมทุก feature เท่ากัน ถ้า signal อยู่ใน 8 จาก 200 features อีก 192 ตัวปนเข้า projection → collision สะท้อน "คล้ายเรื่อง noise":

| D (8 signal + noise) | leadx | kNN | HistGradBoost |
|---|---|---|---|
| 8 | 0.991 | 0.993 | 0.992 |
| 50 | 0.927 | 0.967 | 0.990 |
| 200 | **0.791** | 0.908 | 0.989 |

leadx ทรุด 0.20 (miss rate 0% ตลอด → ไม่ใช่ coverage) และ**หนักกว่า kNN** — boosting นิ่งเพราะเลือก feature เอง นี่คือเหตุผลเชิงโครงสร้างที่ leadx แพ้ boosting บน tabular ที่มี noise

**ทางแก้ที่วัดแล้วได้ผล:** ถ่วง feature ตาม |corr กับ y| → D=200 กลับเป็น **0.973** (ปิด gap ~95%)

> ⚠️ **การค้นพบจากการ implement:** `FeatureEncoder` standardize ต่อ feature → **หักล้างการถ่วงน้ำหนักที่ทำก่อนหน้า** (`(c·x−mean)/std = (x−mean)/std`) การถ่วงต้องทำ **หลัง** standardize เท่านั้น (ฝังใน `feature_weighting="corr"`) การ preprocess ภายนอกไม่มีผล

### #2 Coverage collapse ในมิติสูง

มิติสูง → คู่จุดสุ่มทำมุมเข้าใกล้ 90° หมด → collision `(1/2)^B` เล็กมาก → query ตก bucket ว่าง → prior:

| B (bits/table, D=50) | test AUC | miss rate |
|---|---|---|
| 8–12 | ~0.87 | 0% |
| 16 | 0.848 | 1.4% |
| 20 | 0.820 | **42%** |
| 24 | 0.720 | **86%** |

หน้าผาชัดที่ B≈16–20 — จัดการได้เพราะ tuner เลือก B ในโซนปลอดภัย (จึงเป็น knob ไม่ใช่ข้อบกพร่องเชิงโครงสร้าง) ทางแก้อีกฝั่งคือ recursive refinement ของ dense bucket (v2)

### #3 Seed representativeness / covariate shift

score = density ratio เชื่อว่า "ความหนาแน่น seed ใน bucket = ความหนาแน่นจริง" → selection bias (positive เก็บคนละทางกับ negative) หรือ data drift ทำให้เพี้ยน `miss_rate_final` ช่วยตรวจ shift แต่ไม่มีกลไก correct bias

### #4 Calibration

score เฉลี่ยข้าม table + m-estimate เป็น ranking monotone-ish ไม่ใช่ P(y|x) — ดีสำหรับ AUC/AP, ถ้าต้อง threshold ตาม expected value ให้ต่อ isotonic บน OOF

## 5. Design tensions ที่หลีกไม่ได้

### Accuracy ↔ Reuse

- **feature weighting** (แก้ #1) ใช้ y → codes กลาย task-specific → **ปิด backbone reuse**
- **backbone reuse** ต้องการ y-independent projections → ใช้ feature weighting ไม่ได้

ไม่มีทางได้ทั้งสองเต็มๆ พร้อมกัน → ทำเป็นสองโหมด: `feature_weighting="corr"` (accuracy, โจทย์เดียว) กับ `LeadXBackbone` (reuse, หลายโจทย์) ผู้ใช้เลือกตาม use case

### Resolution ↔ Coverage

B มาก = bucket ละเอียด (แม่นขึ้น) แต่ coverage ต่ำ (miss เยอะ) — สองตัวดึงกันคนละทาง optimal B ขึ้นกับ #seeds + signal → เป็น bias-variance ที่ต้องจูน per-task (ทำได้ถูกด้วย prefix-mask บน cached codes)

## 5.1 Methodology caveats (จาก code review — จดไว้ ยังไม่แก้)

1. **`feature_weighting` leak เข้า auto-tune** — weights คำนวณจาก y ทั้งชุดก่อนส่ง Z เข้า
   Tuner → validation folds เห็น y ของตัวเองผ่าน weights → `tuning_results_` optimistic
   เล็กน้อย (เกรดเดียวกับ supervised-bank ที่จดใน comment ของ tuning.py) — แก้จริงต้อง
   cross-fit weights ต่อ fold (v2)
2. **PU tuner: background ใช้ทั้ง fit และ eval** — `cbg` สร้างตาราง (denominator) และเป็น
   negative proxy ตอนวัด AP พร้อมกัน → bias เล็ก มีผลเมื่อ background เล็ก
3. **coverage weights นับ `hit` จาก direct join ก่อน multi-probe** — table ที่ probe
   ช่วยกู้ได้เยอะถูกถ่วงเท่า table ที่ miss จริง — เป็น design choice (ค่า conservative)

## 5.2 Lift scoring — design ที่ converge แล้ว (implemented)

จากการหารือเรื่อง "คะแนนดิบ vs สัดส่วน": คะแนนดิบ (log-density) ปน "กล่องใหญ่" กับ
"กล่องเข้มข้น" — สิ่งที่ถูกคือ **density ratio (lift)** เทียบ reference population:

```
lift(b) = (n_pos(b) + α) / (E[n_pos(b) | เป็นกลาง] + α),  E = N_pos · n_ref(b)/N_ref
```

- **imbalance-invariant**: class prior π ไม่อยู่ในสูตร → ranking ไม่สะเทือนตาม ratio
  (ยืนยันเชิงประจักษ์: ตรึง 2,000 positives, ratio เปลี่ยน 100 เท่า → AUC แบนราบ
  0.879–0.889 ขณะ HistGB ทรุด 0.909→0.846; จุดตัด ~0.3% pos rate)
- **shrink-toward-1**: กล่องหลักฐานน้อยถูกดึงเข้า 1 (กัน outlier กล่องจิ๋ว lift พุ่ง —
  ความแรง α ให้ Tuner เลือก; validate กับข้อมูลจริงอีกครั้ง)
- **fallback = 1** (เป็นกลาง) — ภาษาเดียวทั้งระบบ: 1=ไม่มีหลักฐาน >1=บวก <1=ลบ
  (แก้ปัญหา prior fallback คนละสเกลที่เคยต้องอาศัย normalize)
- **reference ตรึงตอน fit** (batch-invariant): two_class = seeds ทั้งหมด, pu =
  background (จำเป็น — lift ใน pu ไม่มี background จะ raise)
- **aggregation**: arithmetic ("mean") vs geometric ("log", NB-style คูณ likelihood
  ratio ข้าม tables) — Tuner ตัดสิน; วัดแล้ว log ดีกว่าเล็กน้อยและเสถียร
- **predict_proba ภายใต้ lift**: P(y=1|x) ≈ π·lift (SCAR) clip [0,1] — probability
  หยาบ ranking คงเดิม

## 6. ทิศทางที่พักไว้ (พิจารณา v2)

- **median-threshold hashing** — offset hashing (threshold = median ของ `w·x` แทนตัดที่ 0) → balanced bit โดยอัตโนมัติ, ลบ bit-test machinery, ไม่บิดข้อมูล (LSH ไม่จำเป็นต้องผ่าน origin)
- **PCA/ZCA whiten / ITQ** — แก้ anisotropy ให้เส้นสุ่มตัดตรงที่ข้อมูลอยู่จริง (พักเพราะเปลี่ยน data distribution)
- **recursive refinement** ของ dense bucket — แก้ coverage imbalance (รอข้อมูลจริงก่อน)
- **cross-fitted / shared feature weights** — ลด accuracy↔reuse tension
- **multi-class**, **isotonic calibration**, **GPU/out-of-core**
