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

### #1b Categorical blindness — แก้แล้ว (ดู §5.3 type-aware bits)

hyperplane bit ที่คลุมทุกมิติแทบมองไม่เห็น one-hot: `P(bit เปลี่ยน | เปลี่ยน
category) = 3.6%` → signal ใน categorical ให้ AUC ≈ สุ่ม (0.526) — ข้อสังเกตจาก
ผู้ใช้ แก้ด้วย `cat_bit_frac` (bits แยกชนิด): 0.526 → 0.929

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

## 5.1 Methodology caveats (จาก code review)

1. ~~**`feature_weighting` leak เข้า auto-tune**~~ — **✅ แก้แล้ว (audit P0)**:
   Tuner cross-fit weights + supervised geometry ต่อ fold — validation fold ไม่เห็น
   y ตัวเองผ่าน geometry; weights เต็มคำนวณหลัง tune เฉพาะสำหรับ deploy
   (regression: noise-data best score ~0.5, spy ยืนยัน bank fit ต่อ fold)
2. ~~**PU tuner: background ใช้ทั้ง fit และ eval**~~ — **✅ แก้แล้ว (audit P0)**:
   แบ่ง background เป็น fit/eval คนละครึ่ง; refit สุดท้ายใช้เต็มตามเดิม
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

## 5.3 v2 — implemented

### Median-threshold hashing (default ใหม่: `hash_mode="median"`)

`bit = (w·x > t)` โดย t = **tie-aware threshold** (~median สำหรับ continuous; สำหรับ
discrete atoms เลือกจุดตัดระหว่าง unique values ที่ mass split ใกล้ 50/50 สุด — median
ดิบตกบน atom ของ majority แล้วให้ bit ตายได้ เช่น one-hot 90/10 → 21/80 bits ตาย
ก่อน fix, 0/80 หลัง fix):
- **ตัด machinery ทั้งชุด** (balance test, resampling, redundancy greedy) ใน unsupervised
  path → fit เร็วขึ้น ~15x และไม่มีสมมุติฐานให้ผิดอีก
- **ฆ่า failure mode "input ไม่ centered"** — threshold เลื่อนตามข้อมูลเอง (aniso+offset:
  0.818 → 0.875); origin mode ยังอยู่เพื่อ backward compat
- **accuracy: เสมอกันทางสถิติ** — paired 25 คู่ (5 families × 5 seeds): diff
  +0.001 ± 0.015, median ชนะ 12/25. รายครอบครัว: normal +0.011, noisy +0.009,
  skewed +0.002, imbalanced −0.006, heavy-tail −0.011 (เคสเดียวที่ origin มีแต้มต่อ
  — ใช้ hash_mode="origin" ได้ถ้าจำเป็น). จุดขายคือ speed/robustness/simplicity
  ไม่ใช่ความแม่น
- supervised ยังใช้ oversample + cross-fitted IG ranking บน median bits

### Shared feature weights (`backbone.fit(df, y_ref=...)`)

ลด tension accuracy↔reuse แบบจ่ายบางส่วน: เรียน |corr| weights จาก**โจทย์อ้างอิงเดียว**
แล้วแชร์กับทุกโจทย์บน backbone (วัด OOS, d=100, 8 signal features ร่วม):

| | plain backbone | + shared weights |
|---|---|---|
| โจทย์ที่แชร์ signal features (3 โจทย์) | 0.66–0.69 | **0.92–0.96** |
| โจทย์ signal คนละชุด (disjoint) | 0.646 | **0.540** ← แย่ลง |

**เงื่อนไขการใช้:** y_ref ต้อง represent กลุ่มโจทย์ที่จะ reuse จริง — ถ้าโจทย์ในอนาคต
signal อยู่คนละ features weights จะกดมันทิ้ง (semi-task-agnostic ไม่ใช่ fully)

### Tuner ↔ final model consistency (code review)

- `hash_mode` ถูกส่งเข้า Tuner แล้ว (เดิม tune ด้วย median เสมอแม้ผู้ใช้ขอ origin)
- โมเดลสุดท้ายหลัง tune = **prefix slice ของ winning bank** ที่ถูกประเมินจริง
  (เดิม refit bank ใหม่ — GridSearchCV-style ไม่ผิดแต่มี variance ระหว่าง
  evaluated↔deployed; ตอนนี้เป็นศูนย์ + ประหยัด refit)
- serialization เก็บ `hash_mode`/`cat_bit_frac` ครบทั้ง LeadX/backbone
- **legacy hash_mode inference** (`infer_hash_mode` ใน hashing.py): artifact เก่า
  ที่ไม่มี key — ไม่มี thresholds.npy (era-1) หรือ thresholds ศูนย์ล้วน (era-2 origin)
  → "origin"; ไม่เป็นศูนย์ → "median" — file existence อย่างเดียวไม่พอเพราะ era-2
  origin ก็ save thresholds (เป็นศูนย์); predictions ไม่เปลี่ยนทุกกรณี
- **diagnostics หลัง tune สะท้อน deployed prefix** (`projection_diagnostics` ใน
  projections.py): เดิมรายงานจาก bank เต็ม B_max/K_max — ตอนนี้คำนวณใหม่จาก
  P/thresholds ที่ slice แล้วบน sample (~20k) โครงสร้างการันตีตรงกับ hasher เสมอ

### Type-aware bits — แก้ categorical blindness (`cat_bit_frac`)

ข้อสังเกตจากผู้ใช้ที่วัดแล้วจริงและรุนแรง: hyperplane bit ที่คลุมทุกมิติแทบมองไม่เห็น
categorical — contribution ของ one-hot (`w_j/√k`) จิ๋วเทียบความแกว่งของมิติ numeric รวม
→ วัดได้ `P(bit เปลี่ยน | เปลี่ยน category) = 3.6%` (ควร ~50%) → ถ้า signal อยู่ใน
categorical วิธีเดิมพังเกือบสุ่ม

**ทางแก้:** จัดสรร bits แยกชนิด — cat bits project เฉพาะ categorical blocks (+median
threshold = random bipartition ของ categories ที่ balanced ตามมวล) interleave กับ
numeric bits เพื่อให้ prefix masking เหลือทั้งคู่. ผลวัดผ่าน facade (3 seeds):

| signal | frac=0 (เดิม) | type-aware (auto) | tuner เลือก |
|---|---|---|---|
| categorical ล้วน | 0.526 | **0.929** | auto |
| ผสม num+cat | 0.773 | **0.888** | auto |
| numeric ล้วน | **0.896** | 0.811 | **0.0** ← เลี่ยง regression |

`cat_bit_frac` อยู่ใน tuner grid (ต้อง re-hash ต่อ frac — จ่ายได้เพราะ median-mode fit
เร็ว) supervised strategy จัดสรรชนิด bits เองผ่าน IG ranking (candidate pool ผสมสองชนิด)
เชิงทฤษฎี: การยอมรับว่า hyperplane เป็น hash family ของ continuous space — categorical
ต้องใช้ family ของมันเอง (bipartition) แล้วต่อรหัสเข้า bucket เดียวกัน

## 5.5 Real-data validation (telco / adult / home-credit)

ผลรอบแรกกับข้อมูลจริง (examples/real_data_benchmark.py):

**Two-class:** HistGB นำทุกชุดตามคาด — telco แค่ −0.006 (categorical หนัก, cat_bit_frac
ทำงาน), adult −0.029, home-credit −0.030 (weak signal 122 features = ระบอบ noise
dilution); feature_weighting ช่วยสม่ำเสมอ (+0.006…+0.019)

**PU (niche หลัก) — ยืนยัน:** home-credit เห็น positive 30% ไม่มี negative → AUC 0.643,
recall@top10% 21.3% (สุ่ม 10%) ใน setting ที่ supervised ตรงๆ ใช้ไม่ได้ (ไม่มี labeled negatives — เทียบผ่าน PU-naive conversion ภายหลังใน 8c)

**Invariance — ยืนยัน / crossover — ต้อง qualify:** ตรึง 789 positives แปร ratio
7.3→0.35% → leadx แบนราบ (0.604/0.602/0.608) ตามทฤษฎี แต่ HistGB บน weak signal
ทรุดช้ากว่า synthetic มาก (−0.012 vs −0.063) และยังนำทุก ratio → **crossover ที่เห็น
บน synthetic เกิดเมื่อ signal ชัด** — invariance เป็นคุณสมบัติของ leadx จริง แต่ไม่
การันตีชนะเมื่อระดับสัมบูรณ์ต่ำกว่า

**บทเรียน methodology:** rarity sweep รอบแรกของเราสับสนตัวแปร (ลดทั้ง ratio และ
จำนวน positive สัมบูรณ์พร้อมกัน) — invariance ต้องทดสอบแบบตรึง N_pos แล้วแปร negative
เท่านั้น

## 5.6 Phase-1 Locality headroom (จาก literature-review proposal)

Exact-kNN oracle gate บน home-credit PU (รายละเอียด outputs/LOCALITY_HEADROOM_REPORT.md):
**gate ไม่ผ่าน → geometry headroom เล็ก → Phase 2A (local projection forest) ไม่คุ้ม**
เส้นทางถัดไปคือ representation (Phase 2B: ITQ/PCA cross-fit, task-aware transform)

**กำไรที่เจอระหว่างทาง:** tables scaling — b10,K=80 ชนะ kNN-oracle ที่ 10% seeds
(0.1548 vs 0.1422) และ +24% rel เหนือ config เดิม → DEFAULT_GRID เพิ่ม K=40;
คำแนะนำ PU: b8–10, K=40–80 ("more tables > more bits" ยืนยันแล้ว)
ปฏิเสธเพิ่ม: mode-aware scoring (แย่ลงทุกกรณี), PU-refinement (ถูก tables scaling ครอบ)

**Phase 2B (learned geometry) — ผ่าน gate 2 ตัว, shipped:**
- `feature_weighting="contrast"` (pu): +26-30% rel PR-AUC, 10/10 draws — task-aware
- `whiten=True`: +16-21% rel, 10/10 — **unsupervised/reusable** แต่ **data-dependent**
  (ขยาย noise บนข้อมูลที่ residual directions ไม่มี signal — วัดแล้วแย่ลงได้ถึง −0.13
  บน synthetic บางแบบ) → เข้า tuner grid ให้ตัดสินต่อ dataset
- ปฏิเสธ: ITQ แบบ PCA-10 (บีบมิติแรงเกิน −14%), Anchor Graph Hashing (−24%),
  whiten+contrast combo (ไม่ stack — information ซ้อนกัน)
- รายละเอียด: outputs/LOCALITY_HEADROOM_REPORT.md (Phase 2B section)

## 5.4 ทดลองแล้วปฏิเสธ (กันเดินซ้ำรอย)

**kNN re-rank ของ shortlist (top 10%)** — fraud @ 0.3%, 61 seeds, 6 reps paired, primary p@1%:
`knn_seed` (-d ถึง seed ที่ 5) **-85.9%** (0/6), `knn_ratio` (log d_bg - log d_seed,
kNN density-ratio) **-34.2%** (1/6, PR -50%), `blend` (rank-avg กับ lift) +1.0% (3/6)
แต่ PR -8.1% -> ไม่มี variant ผ่าน gate (+3%, >= 5/6, PR >= -2%)
**ข้อสรุป:** ระยะ Euclidean ใน encoded+weighted space (~170 มิติ) *แย่กว่า* bucket
density ของ LeadX มาก — แถวที่ใกล้ seed ที่สุดแทบไม่ใช่ fraud (p@0.1% = 0) สอดคล้องกับ
ผล feature selection: สัญญาณอยู่ใน *การประกอบกัน* ของ feature ขณะที่ระยะรวมทุกมิติถูก
มิติที่มีสัญญาณอ่อนจำนวนมากกลบ; การโหวตข้าม random projections ทนต่อสิ่งนี้ได้ดีกว่า
(ไม่ได้ทดสอบ metric อื่น เช่น cosine/Mahalanobis — ปิดเฉพาะ Euclidean kNN)
(ดู `outputs/knn_rerank_experiment.py`, `outputs/knn_rerank_raw.csv`)

**n_tables 80 -> 160/320 ที่ ultra-imbalance** — fraud @ 0.3%, 61 seeds, 6 reps paired:
320 vs 80 median **+4.59%** PR-AUC แต่ชนะ **4/6** (gate ต้องการ +3% และ >= 5/6) -> ไม่ผ่าน;
สอง rep ที่ "แพ้" คือ -0.4% ทั้งคู่ (เสมอ) อีกสี่ rep +4.1..+6.6% -> ทิศทางบวกแต่ยังไม่พอ
ที่ fit ช้าลง 3.9x (58s -> 229s). precision@0.1% +8.0% (4/6) เป็น secondary
**สิ่งที่หักล้างชัด:** CV ไม่ลดเลย (14.5 -> 15.5%) และ `miss_rate_join` คงที่ 0.947
ทุก arm -> ความแปรปรวนที่เหลือคือ **seed draw** (สอดคล้อง variance_decomposition
ที่ seed draw CV 18% > geometry 8%) ซึ่งเพิ่ม tables แก้ไม่ได้ และสัดส่วน table ที่มี
หลักฐานบวกต่อแถวไม่ขึ้นกับจำนวน tables (เป็นสมบัติของ seeds/bucket ไม่ใช่ของ ensemble)
(ดู `outputs/tables_capacity_ultra.py`, `outputs/tables_capacity_ultra_raw.csv`)

**Feature selection ก่อน fit (IG top-k)** — ultra-imbalance, 9 paired draws:
ทุก variant **แพ้ขาด** — oracle_ig_16 −40.4%, oracle_ig_48 −29.0%, pu_ig_16 −43.8%,
pu_ig_48 −26.2% (PR-AUC, ชนะ 1/9 เท่ากันหมด; gate ต้องการ +5% และ ≥7/9)
**หลักฐานชี้กลไก:** `oracle_ig` (IG จาก label จริงทั้ง train set = ขีดบนของการคัด
univariate) **เสมอกับ** `pu_ig` (4–6 จาก 9) → ความล้มเหลวไม่ได้อยู่ที่ประมาณ IG ผิด
แต่**เกณฑ์ univariate เองผิดกับ density-ratio scorer**: bucket ที่ discriminative มาจาก
*การประกอบกัน*ของ feature ที่หายากและ enriched ไม่ใช่จากการที่แต่ละตัวทำนายเก่งเดี่ยวๆ
IG จึงตัด feature ที่ร่วมสร้าง conjunction ทิ้ง ยืนยันด้วยสองสัญญาณ: k=48 แย่น้อยกว่า
k=16 ทุกช่อง (monotone กับ D) และ ROC เสียหายน้อยกว่า PR-AUC 5–10 เท่า (ความเสียหาย
กระจุกที่หัวแถว = จุดที่ resolution ของ partition สำคัญ) สมมติฐานเดิม SNR≈√(k/(D−k))
สมมติว่าสัญญาณอยู่ใน subspace ของ feature ที่ informative เดี่ยวๆ — ไม่จริงในโจทย์นี้
→ **LeadX ใช้ feature กว้างๆ ได้เป็นคุณสมบัติ ผู้ใช้ไม่ต้อง (และไม่ควร) คัดก่อน**
(ดู `outputs/FEATURE_SELECTION_REPORT.md`; ยังไม่ได้ทดสอบกรณีเติม pure noise columns)

**Two-step PU (reliable negatives + self-training)** — home-credit PU, 5 draws:
onestep 0.1985 vs twostep_same 0.1918 / twostep_reweight 0.1902 (แพ้ 0/5 ทั้งคู่)
**เหตุผลเชิงโครงสร้าง:** two-step ออกแบบมาสำหรับ discriminative classifiers ที่*ต้องมี*
labeled negatives — แต่ lift ของเราเป็น density ratio ที่ใช้ background **ทั้งหมด**เป็น
ตัวหารอยู่แล้ว (ข้อมูลมากกว่า) การแทน background ด้วย "easy negatives" (bottom 40%)
คือการ*ตัดมวลใกล้ boundary ออกจากตัวหาร*ตรงจุดที่ resolution สำคัญที่สุด + inherit
error ของรอบแรก → density-ratio scorer ไม่มีอะไรให้ two-step ปลดล็อค

**Rank-based contrast weights** (per-feature |AUC−0.5| แทน |mean diff|):
0.1993 vs 0.1985 — parity (4/5 แต่ +0.4% rel ต่ำกว่าทุก gate) — mean-diff เพียงพอแล้ว
บนข้อมูลที่ standardize

**Adaptive-depth scoring** (แถวเลือกความลึก prefix เองตามหลักฐาน ≥ tau):
ไม่ช่วย (B=20: 0.918 vs fixed 0.940) — กลไก lift fallback=1 + multi-probe ทำหน้าที่
"ถอยเมื่อหลักฐานบาง" อยู่แล้วแบบ implicit และดีกว่า (ไม่ทิ้งหลักฐาน 1-2 seeds)
**ผลพลอยได้สำคัญ:** การวัดนี้เผยว่า**หน้าผา coverage (#2) ถูก lift redesign แก้ไป
เงียบๆ แล้ว** — B=20 direct-miss 73% แต่ AUC ดีสุดในตาราง เพราะ fallback=1 ใน
geometric mean = "table ไม่มีหลักฐานก็ไม่โหวต" ไม่ใช่โหวตค่าผิดแบบ prop/prior เดิม
→ ตาราง failure mode #2 สะท้อนพฤติกรรมของ prop scoring; ภายใต้ lift หน้าผาตื้นกว่ามาก

**Pair-difference directions** (ทิศจากผลต่างคู่จุด, rpTree-style):
ไม่ช่วยทุก dataset (แย่ลง 0.005–0.014) — encoder standardize ทำ variance เท่ากันทุกแกน
แล้ว ทิศ gaussian จึงเหมาะสมอยู่แล้ว (บนข้อมูลดิบคงช่วย แต่ pipeline ไม่เจอข้อมูลดิบ)

**Margin-aware multi-probe** (fallback flip เฉพาะ T bits ที่ |z−threshold| แคบสุด
แทน flip ทุก bit — ข้อเสนอ audit P1-4; 3 draws, K=40):

| regime | miss rate | none | all (ปัจจุบัน) | m2 | m4 | m4+pair |
|---|---|---|---|---|---|---|
| home-credit PU b10 (deploy) | 0.0% | 0.2005 | 0.2005 | 0.2005 | 0.2005 | 0.2005 |
| home-credit PU b20 | 17% | 0.1894 | 0.1893 | 0.1896 | 0.1896 | 0.1896 |
| adult 2c 30seeds b14 | 82% | 0.721 | **0.757** | 0.741 | 0.747 | 0.751 |
| adult 2c 30seeds b20 | 88% | 0.661 | **0.727** | 0.694 | 0.710 | 0.715 |

**ผล: ปฏิเสธ** — สมมติฐานผิดทั้งสอง regime: (1) config ที่ deploy จริง (bits ต่ำจาก
tables scaling) miss≈0 → probe ไม่เคยทำงาน (2) โซน miss สูง ตัวจำกัดคือ *recall*
(หา bucket ที่มี seeds ให้เจอสักอัน — neighbors ส่วนใหญ่ก็ว่าง) ไม่ใช่ *precision*
(เลือก neighbor คล้ายสุด) → probe 20 ทิศชนะ probe 4 ทิศฉลาดๆ monotone ตาม T
และ argument ประหยัด cost ไม่มีน้ำหนัก (fallback join เป็นเสี้ยวเล็กของ runtime)
**บทเรียน:** Lv et al. multi-probe ออกแบบมาเพื่อ ANN index หนาแน่น — คนละ regime
กับ seed table บาง (raw: outputs/margin_multiprobe_raw.csv)

**Covariance-sampled node directions** (rpforest variant: ทิศต่อ node ~
N(0, Σ_node) ผ่าน (Z−μ)ᵀr — PCA-flavored แต่คง diversity; 3 draws vs rpf
random draw เดียวกัน): ชนะ 1/4 cells (fraud 0.1% +32%) แพ้ HC 0.3% −16%
เสมอที่เหลือ natural ไม่ขยับ + fit แพง 1.5-1.7x → **ปฏิเสธ** — สอดคล้อง
บทเรียนเดิม: หลัง standardize ทิศ isotropic เพียงพอ (pair-difference, whiten
dataset-specific) และ split แบบ ~balanced จัดการ density แล้ว การปรับ*ทิศ*ตาม
variance ไม่เพิ่มอะไร (raw: rpf_cov_raw.csv)

**MI feature weighting** (แทน `contrast` ที่เป็น first moment ด้วย mutual
information ระหว่าง feature กับ is-seed — จับความต่างของ distribution ทุกแบบ,
PU-native; + variant ที่มี soft mRMR redundancy penalty). IEEE fraud temporal
1:1,000, 3 folds × 2 draws, มี/ไม่มี +60 noise features:

| paired vs contrast | PR | R@0.5% | R@0.1% |
|---|---|---|---|
| no noise: mi | +14.3% (5/6) | −2.9% (1/6) | +11.7% (4/6) |
| +60 noise: mi | −4.8% (2/6) | 0.0% (2/6) | +17.1% (4/6) |

**ผล: ปฏิเสธ** — gate ที่ล็อกไว้ (mi ชนะ contrast บน R@0.5% ในสนาม +noise,
median>0 และ ≥4/6) ได้ median 0.0%, 2/6 = ตก; mi_mrmr ≈ mi ไม่เพิ่มอะไร
(รอบแรก mrmr พังเพราะ NaN จาก corrcoef บนคอลัมน์คงที่ใน subsample — บั๊ก ไม่ใช่ผล)
**lead ที่เหลือไว้**: R@0.1% mi ชนะทั้งสองสนาม (4/6 ทั้งคู่) — สอดคล้องทฤษฎีว่า
ช่วยที่ shortlist แคบสุด แต่หลักฐานอ่อนและไม่ใช่ primary metric

**ผลพลอยได้ที่มีค่ากว่า — weighting คุ้มเฉพาะเมื่อ feature ไม่สะอาด:**
`none` vs `contrast` → ไม่มี noise: none **ชนะ** PR +22.8% (5/6); มี noise:
none **แพ้ยับ** PR −33.5%, R@0.1% −20% (0/6) → feature selection กับ feature
weighting ทดแทนกันบางส่วน "คัด top-N แล้วเปิด contrast ทับ" ไม่ใช่คำตอบอัตโนมัติ
ต้องวัดบน feature set จริง (raw: outputs/mi_weighting_raw.csv,
MI_WEIGHTING_REPORT.md)

**บทสรุปเชิงทิศทาง:** หลัก "localize + กระจายดี" ถูกดูดซับเกือบหมดแล้วผ่าน tie-aware
median (กระจาย/bit) + K tables + lift/log (locality) + fallback=1/multi-probe (adaptive
resolution โดยพฤตินัย) — คานงัดที่เหลือ: rpTree hierarchical partition (balance โดยประมาณบน fit sample
โดยโครงสร้าง — งาน v3 ใหญ่, แลก single-matmul) และ validation กับข้อมูลจริง

**Feature bootstrap / random subspace ต่อ table** (สุ่ม subset ของ features ให้แต่ละ
hash table) — หวังแก้ noise dilution แบบ unsupervised (รักษา backbone reuse):

| case | full | sqrt(D) | D/4 | D/2 |
|---|---|---|---|---|
| 8 signal + 192 noise (D=200) | 0.796 | 0.798 | 0.756 | 0.793 |
| dense signal (D=40) | **0.955** | 0.913 | 0.936 | 0.948 |

**ผล: ปฏิเสธ** — ไม่ช่วยโจทย์เป้าหมาย (SNR เฉลี่ยต่อ projection ไม่เปลี่ยน แค่สับตำแหน่ง)
และทำร้าย dense-signal จริง (ตัด features = หั่นทิศ signal ร่วม) **บทเรียน:** สู้ noise
dilution ต้องมี "ความรู้เรื่อง relevance" (feature_weighting ใช้ y) ไม่ใช่แค่ "ความหลากหลาย"
— ไม่มีทางลัด unsupervised ง่ายๆ, tension accuracy↔reuse ยังอยู่ครบ

## 7. Evolution summary — ปัญหา → การแก้ → ผล (บันทึกเส้นทางทั้งหมด)

| ปัญหาที่พบ/หารือ | สิ่งที่ปรับ | ผลวัด |
|---|---|---|
| tune ยาก สุ่มหลายรอบแล้วลุ้น (pain point ตั้งต้น) | Tuner: nested prefix-masking, hash ครั้งเดียวทั้ง grid; final = prefix ของ bank ที่ประเมินจริง | prefix reuse ลดต้นทุนภายใน geometry เดียวกัน (100+ scoring configs ≈ hash ชุดเดียว) — แต่ engine/strategy/whiten/label-dependent folds ยังต้องสร้าง geometry แยกของใครของมัน |
| เส้นตัดผ่าน origin + machinery สมมุติฐานเยอะ | `hash_mode="median"` + tie-aware threshold — ลบ balance-test/resample/redundancy ทั้งชุด | fit เร็ว 15x, robust, AUC เสมอเดิม (+0.001±0.015, 25 คู่) |
| คะแนนดิบ vs สัดส่วน (คำถาม scoring) | `score_type="lift"` — density ratio, shrink-toward-1, fallback=1, agg mean/log | imbalance-invariant: แบนราบทุก ratio, ตัด HistGB ที่ ~0.3% |
| noise features กลบ signal | `feature_weighting="corr"` (หลัง standardize เท่านั้น) | D=200: 0.79 → 0.97 |
| one-hot ทั้ง 0/1 ตกฝั่งเดียวของ plane | `cat_bit_frac` type-aware bits + tuner เลือกสัดส่วน | cat-signal: 0.53 → 0.93; numeric ไม่ regress |
| จูนครั้งเดียวใช้หลายโจทย์ | `LeadXBackbone` + `y_ref` shared weights | 30ms/โจทย์; related tasks +0.25 AUC |
| bugs (batch-dependent PU, NaN, dead bits, config identity, tuner consistency) | fixes + regression tests (ปัจจุบัน 138 tests) | ทุกข้อมี test คุม |
| audit P0: OOF leakage / benchmark↔production parity | cross-fit ต่อ fold, PU bg split, encoder fit จาก background | ตัวเลขรายงานเชื่อได้ (noise-leak bound test) |
| variance ที่ seeds น้อยมาก (โซนธง imbalance) | `hier_depths` — EB shrink เข้าหา bucket แม่ข้าม prefix depths | +41% fraud 0.3%, +15% fraud 0.1% (§8d) |
| scalability ที่ 10M แถว (โจทย์ตั้งต้น) | encoder preallocate f32 + `BackgroundStats` cache + `seed_rows`/`score_cached` | backbone 351s→51s; โจทย์ 2+ = 0.04s (§8e) |
| mega-bucket ในย่านหนาแน่น (สมมติฐานเจ้าของ ต้นโปรเจกต์) | `engine="rpforest"` — rpTree: ทิศ+median ท้องถิ่นต่อ node → balanced โดยประมาณบน fit sample | fraud 0.3% +39%, HC 0.3% +42%, natural +6% (ratio-of-medians, prototype harness — paired/provenance ดู §8f); occupancy 6.3%→0.098% |
| rpforest ใช้กับ auto-tune ไม่ได้ | "engine" เป็นมิติใน tuner grid + `RPForest.prefix` + fit vectorized (−40%) + facade `engine="auto"` | tuner เลือก engine ต่อ dataset จาก OOF; deploy = prefix ที่วัดจริง (§8f) |

**เส้นทางตัวเลขธง (home-credit PU 10% seeds, PR-AUC, parity ทุกจุด):**
0.194 (baseline หลังปิด leakage) → 0.196 (+hier) → **0.208 (+rpforest; paired
+5.4%, 3/3 draws — +7% รวม; prototype harness — ดู provenance §8f)**
— extreme cells: fraud 0.3% 0.0235→0.0595 (2.5x), HC 0.3% 0.0047→0.0066 (+42%)

**Top-3 การปรับที่มีค่าที่สุด:** (1) lift scoring — เปลี่ยน positioning ด้วยคณิตศาสตร์
(2) type-aware bits — อุดรูรั่ว categorical (3) median/tie-aware hashing — ลบโค้ด
มากกว่าเพิ่ม ได้ simplicity+robustness+speed พร้อมกัน — ทุกอันมาจากการหารือ ไม่ใช่
textbook และผ่านด่านเดียวกัน: prototype → วัด → เข้า library เมื่อตัวเลขยืนยัน

## 6. ทิศทางที่พักไว้ (พิจารณา v3)

- ~~**PCA/ZCA whiten**~~ — ✅ ship แล้ว (Phase 2B: `whiten` ใน tuner grid; ITQ วัดแล้ว reject — §5.4/§8)
- ~~**recursive refinement** ของ dense bucket~~ — ✅ ทำแล้วในรูป `engine="rpforest"` (§8f — mega-bucket ยืนยัน+แก้)
- ~~**cross-fitted weights ใน Tuner**~~ — ✅ ทำแล้ว (audit P0 — ดู §5.1 ข้อ 1)
- **multi-class**, **isotonic calibration**, **GPU/out-of-core**

## 8b. ผลทดสอบข้อเสนอ P1 ของ audit (2026-07-18, b10K80 baseline, PU 10% seeds)

| ข้อเสนอ | fraud (temporal) | home-credit | คำตัดสิน |
|---|---|---|---|
| numeric-only whitening | 0.196 vs base 0.250 (0/5) | 0.138 vs 0.155 (0/5) | **ปฏิเสธ** — แพ้ทั้งคู่ แย่กว่า full-whiten บน HC ด้วย |
| robust contrast (cross-fit+shrink) | 0.268 (แพ้ plain 0.274) | 0.198 (ชนะ plain นิด) | parity — ไม่คุ้มความซับซ้อน |
| table selection (160→greedy 80) | — | greedy 0.1555 < first80 0.1565 < **all160 0.1581** | **ปฏิเสธ** — selection overfit seed-split; "ใช้ทั้งหมด" ชนะอีกครั้ง; 80→160 = +1% (plateau) |
| margin-aware multi-probe | — | ทุกเงื่อนไขเท่ากัน (miss≈0 ที่ b10) | **ปฏิเสธ** — โซน miss สูง (adult 30 seeds, miss 82-89%) all-bit ชนะทุก T: 0.727 vs m4p 0.715 vs m2 0.694 (b20) |
| tuner dedupe fix (P0-1) | — | — | **✅ แก้แล้ว + regression test** (supervised หายจริงตามที่ audit ชี้) |

**การแก้ claim ของ audit:** contrast บน fraud ไม่ได้แพ้ — ด้วย baseline b10K80
(config production) contrast **ชนะ fraud 5/5 (+10%: 0.274 vs 0.250)** — ผล −3%
ของ audit วัดภายใต้เงื่อนไข/baseline ที่ต่างจากของเรา (สาเหตุของส่วนต่างยังไม่ถูก
isolate) → contrast general กว่าที่ audit สรุป
(whiten ยังคง dataset-specific จริงตาม audit: แพ้ fraud ทุกแบบ)

## 8c. Imbalance-focus benchmark (2026-07-18) — นิยาม "ดีกว่าวิธีอื่น" ให้ชัด

คำถามจากผู้ใช้: จุดเด่นคือ imbalance — ตอนนี้ดีกว่าวิธีอื่นมั้ย? แยกคู่เทียบเป็น
"ข้อมูลเท่ากัน" (PU 10% seeds: leadx b10K80+contrast vs HistGB/logreg PU-naive)
กับเพดานอ้างอิง (HistGB label เต็ม, จูนตาม audit — **l2=1.0 จำเป็น**: default
l2=0 พังที่ 0.1% ROC 0.53 ทุก random_state → เกือบสรุปกลับด้าน)
รายละเอียด/ตารางเต็ม: outputs/IMBALANCE_FOCUS_REPORT.md (raw CSV เป็น source of truth)

- **ที่ข้อมูลเท่ากัน crossover มีจริงบนข้อมูลจริง**: leadx ชนะ PU-naive เมื่อรุนแรง —
  fraud 0.1% (20 seeds): PR 3.5x (.0086 vs .0025), lift@0.1% 57x vs 11x;
  home-credit ชนะตั้งแต่ 1% ลงไป; จุดตัด dataset-dependent (HC ~1%, fraud ~0.2%)
- **≥1-3% + seeds หลักร้อย**: HistGB PU-naive ยังชนะบน fraud — ไม่ใช่คำตอบทุกโจทย์
- **ไม่ชนะ oracle จูนแล้วที่ไหนเลย** แต่ gap แคบลงตาม imbalance: HC allseeds
  84→94% ของ oracle (monotonic), fraud ไม่ monotonic: 58→44-45→80% — claim
  label-efficiency จึงยืนได้เต็มที่เฉพาะ home-credit
- **degrade นุ่มนวลสุด** (fraud ROC, positives หาย 30x): leadx −0.04 ≈ oracle −0.05
  << PU-naive −0.19, logreg −0.23 — invariance ของ lift โผล่บนข้อมูลจริง
- **update หลัง rpforest (§8f)**: rerun ด้วย config ธงใหม่ — ชนะ PU-naive
  **5/7 regimes** (HC ทุก ratio; fraud 0.3% และ 0.1% — 0.1% ชนะ 2/3 draws
  variance สูง; แพ้ fraud 3%/1%); HC allseeds = 91-137% ของ oracle จูนแล้ว
  (ชนะที่ 0.3% **เฉพาะ allseeds, split เดียว** — ที่ 10% seeds ยังแพ้) —
  ทั้งหมดเป็น rarity stress test บน fixed split เดียว, ยังไม่เทียบ strong PU
  methods — ดู IMBALANCE_FOCUS_REPORT update section

## 8d. Hierarchical shrinkage — shipped (2026-07-18)

งานแรกของ roadmap "แข็งแรงขึ้นในโซนที่ชนะ": ที่ seeds 20-60 ตัว bucket ลึกมี
หลักฐาน 0-1 ตัว → variance คือตัวจำกัด แก้โดยเปลี่ยน shrink จาก "เข้าหา 1"
เป็น "เข้าหา lift ของ bucket แม่" แบบ EB ลูกโซ่จากหยาบไปลึก:
`L_d = (n_pos_d + α·L_parent) / (E_d + α)` — prefix structure (MSB-ordered)
ให้แม่ฟรีด้วย bit-shift ต่างจาก adaptive-depth ที่ reject (อันนั้น*เลือก* depth
เดียวต่อแถว อันนี้*รวม*ทุก depth) และแทน multi-probe ใน path นี้ไปเลย

| cell | base | hier (α=2) | Δ rel |
|---|---|---|---|
| fraud 0.3% | .0235 | **.0331** | +41% |
| fraud 0.1% | .0086 | **.0099** | +15% |
| home-credit 1% | .0212 | **.0226** | +6.5% |
| home-credit 0.3% | .0047 | .0047 | เสมอ |
| home-credit 8% (regression check) | .1946 | .1964 | +0.9% |

ชนะ/เสมอ 14/15 draws — gate ตั้งไว้ก่อนเห็นผล (ชนะ >=3/4 extreme + ไม่ regress
ที่ natural) ผ่าน → ship เป็น `hier_depths` (opt-in, lift เท่านั้น, "auto" =
(4,6,...,b)) ใน SeedScorer/LeadX/fit_task + save/load; ยังไม่เข้า DEFAULT_GRID
ของ tuner (รอหลักฐาน two_class + grid budget) — prototype:
outputs/hier_shrinkage_experiment.py (raw CSV)

## 8e. Scale validation 1M-10M + สองบั๊ก scalability (2026-07-18)

งานที่สองของ roadmap: พิสูจน์จุดเด่น "train ครั้งเดียวใช้หลายโจทย์" ที่ scale
ตั้งต้นของโจทย์ (10M แถว × 100 features) — ตัวเลขเต็ม: outputs/SCALE_VALIDATION_REPORT.md

**ผล (10M แถว, 24GB laptop):** backbone 51s ครั้งเดียว → โจทย์ PU ถัดไป **0.04s**
(vs HistGB ~115s/โจทย์ + ต้อง label เต็ม — เทียบ **fit-vs-fit เท่านั้น**: 20 โจทย์
~30x; end-to-end: LeadX ≈700s รวม scoring แต่ HistGB predict 10M **ยังไม่ได้วัด**
— ยังสรุปตัวคูณ end-to-end ไม่ได้);
score ประชากร 10M = 31s; peak RSS 12.2GB; ROC ไม่เสื่อมตาม scale (ดีขึ้น:
0.873→0.924 — background ใหญ่ = denominator นิ่ง)

**การวัดรอบแรกเปิดโปงบั๊ก 2 ตัว (แก้ + equivalence tests):**
1. encoder.transform สะสม blocks f64 + hstack → peak ~35GB ที่ 10M → memory
   compression ทำ 318s (super-linear) → preallocate f32: **25s**
2. fit_task จ่าย group-by background (task-independent!) ซ้ำทุกโจทย์ 21s →
   `BackgroundStats` memoize บน backbone ต่อ (b,k,depths): โจทย์ 2+ = **0.04s**
   + API ใหม่: `fit_task(seed_rows=..., background="cached")`, `TaskModel.score_cached()`

**บทเรียน:** benchmark ที่ 300k มองไม่เห็นทั้งสองตัว — super-linear โผล่เมื่อชน
memory ceiling และ "จ่ายซ้ำ" เจ็บจริงเมื่อของที่จ่ายซ้ำโต 30 เท่า

## 8f. RPForest — Balanced Local Projection Forest shipped (2026-07-18)

P2 ตัวแรกที่ลงมือ (ตามการตัดสินใจของเจ้าของ) — และ **สมมติฐาน mega-bucket จาก
ต้นโปรเจกต์ได้รับการยืนยันบนข้อมูลจริง**: bank เดิมบน fraud มี bucket เดียวกิน
6.3% ของประชากร (p99 = 2,509 แถว) ขณะที่ rpTree แคปที่ 0.098% (p99 = 199)
โดยโครงสร้าง — ทิศสุ่มใหม่ + median ท้องถิ่นต่อ node = partition ที่ *balanced โดยประมาณบน
fit sample* (≤200k; ties อาจทำ split เบี้ยว และไม่การันตี balance บน unseen
data/drift) — ต่างจาก SimHash-LSH: เป็น data-dependent recursive tree

ผลวัด (PU b10 K80 + contrast + hier, median 3 draws, gate ตั้งก่อนเห็นผล;
Δ ในตาราง = ratio-of-medians ซึ่ง median สองฝั่งอาจมาจากคนละ draw — ดู
paired per-draw ที่ระมัดระวังกว่าใต้ตาราง):

| cell | bank+hier | rpf+hier | Δ rel |
|---|---|---|---|
| fraud 0.3% | .0331 | **.0461** | **+39%** |
| home-credit 0.3% | .0047 | **.0066** | **+42%** |
| home-credit 1% | .0226 | **.0245** | +8% |
| fraud 0.1% | .0099 | .0097 | −1.6% (เสมอ — 20 seeds, signal-limited) |
| **natural 8%** | .1964 | **.2081** | **+6% — new best home-credit PU** |

**Provenance:** ตัวเลข §8f ทั้งตาราง (รวม 0.208) มาจาก **prototype class ใน
rpforest_experiment.py** — shipped class ภายหลังถูก vectorize (RNG stream ต่าง)
และ re-validate ผ่าน public API เฉพาะ natural cell: **0.206** (3 draws, ad-hoc
ไม่มี raw artifact) → สถานะ = **preliminary** จนกว่าจะ rerun ครบทุก cell ผ่าน
public API พร้อม raw ใหม่ (ส่วน imbalance_focus_rpf_raw ใช้ shipped facade แล้ว)

**Gate:** ผ่านตามเกณฑ์เดิม (ratio-of-medians: 3/4 extreme + natural +6%) —
แต่ **paired per-draw analysis** (คู่ draw เดียวกัน — วิธีที่ถูกกว่าเพราะ median
สองฝั่งอาจมาจากคนละ draw) ให้ภาพระมัดระวังกว่า: fraud 0.3% **+10.7%**,
HC 0.3% **+38.4%**, natural **+5.4%** (3/3 draws) ผ่าน >2%; แต่ HC 1% +1.9%
(ก้ำกึ่ง) และ fraud 0.1% **−1.7%** ไม่ผ่าน → หลักฐานสนับสนุน rpforest เป็น
**engine ทางเลือกแบบ regime-dependent ไม่ใช่ชนะสม่ำเสมอ** — สอดคล้องกับการ
ship เป็น opt-in + ให้ tuner เลือกต่อ dataset; หมายเหตุ: gate run ใช้ prototype
class ใน script (ก่อน vectorized fit ที่เปลี่ยน RNG stream — re-validate แล้ว
เฉพาะ natural cell 0.206) — ควร rerun paired ผ่าน public API ก่อนสรุปถาวร
→ ship เป็น `engine="rpforest"` (opt-in) บน LeadX + Backbone: leaf code มี MSB prefix =
tree path → SeedScorer/hier/mask_codes/BackgroundStats ใช้ร่วมได้ทั้งชุดไม่แก้อะไร

Trade-offs ที่จดไว้: fit+transform ~2-4.5x ของ single-matmul; model size
K×(2^b−1)×D f32 (~51MB ที่ K80 b10 D155 vs ~500KB ของ bank);
~~ยังไม่รองรับ tune=True~~ → **tuner รองรับแล้ว**: "engine" เป็นมิติใน
DEFAULT_GRID (forest = geometry combo เดียว จึงถูก; depth cap 12;
`RPForest.prefix` deploy สิ่งที่วัดจริง; facade `engine="auto"` ให้ tuner
เลือกต่อ dataset — sanity บน HC bg 40k: สอง engines ห่างกัน <2% และ tuner
เลือก hyperplane ซึ่งสอดคล้องข้อเท็จจริงที่ forest ได้เปรียบเฉพาะบางโซน);
cat_bit_frac/whiten ไม่มีผลใน engine นี้; hier บน forest: ปิดประเด็นแล้วด้วย decomposition ที่ fraud 0.1% allseeds
({engine}×{hier}×3 seeds — rpf_anomaly_raw.csv): **hier ช่วยเมื่อ seeds
หลักสิบ, dilute เมื่อ seeds หลักร้อยบน fraud** (−10% bank, −19% rpf ที่ 203
seeds) — anomaly ในตาราง positioning คือ confound นี้ + seed variance ไม่ใช่
ข้อบกพร่องของ forest; แนวใช้งาน: hier_depths="auto" สำหรับ seeds น้อย,
ปิดเมื่อ seeds มาก — prototype/raw: outputs/rpforest_experiment.py

## 8g. Streaming scoring + background sampling — 500M path (2026-08-29)

**คอขวดจริงของ 500M คือขั้น score ไม่ใช่ขั้น fit**

`score_samples` เดิม preallocate `(n_rows x n_encoded_dims)` float32 ทีเดียว —
`chunk_size` ที่มีอยู่มีผลแค่ตอน hash ไม่ใช่ตอน encode เพิ่ม `score_batches` /
`top_k` / `score_samples(chunk_rows=)` (+ `review_queue(chunk_rows=)` ใน sklearn API)

วัด 10M x 150 จาก parquet: **peak 11.32 -> 1.87 GB (-83%)** คะแนนตรงกัน และ
**เร็วกว่า 2.5x** (111k -> 277k rows/s) เพราะ whole-table path เสียเวลากับ memory
pressure ไม่ใช่การคำนวณ (อาการเดียวกับบั๊ก encoder ใน 8e) peak ขั้น fit = 1.25 GB
เท่ากันทั้งสาม mode -> ส่วนต่างเป็นขั้น score ล้วน extrapolate 500M: ~566 GB vs
คงที่ ~1.9 GB (ดู `outputs/STREAMING_SCALE_REPORT.md`)

สัญญาความถูกต้อง: **bucket codes เท่ากันเป๊ะ bit-for-bit** ข้ามทุกการแบ่ง chunk;
score ต่างได้ <=2 ULP จาก BLAS ใน `exp(log(per_table) @ w)` เท่านั้น; ranking ตรงกัน

**background sample เล็กได้กว่าที่คิดมาก** — 5% (10k แถว, `n_bits=10` -> ~10 แถว/
bucket) แยกจาก background เต็ม 203k ไม่ออก และ fit เร็วขึ้น 5.9x; 1% (~2 แถว/bucket)
แย่จริง (-26.6%) ตัวแปรที่น่าจะกำกับคือ **แถวต่อ bucket** (`n_bg` เทียบ `2^n_bits`)
ไม่ใช่ % ของประชากร — ยังไม่ได้วัดการ scale ของกฎนี้ ship เป็น
`LeadXClassifier(background_sample=N)`

**บทเรียนเรื่องวินัยการทดลอง (สำคัญกว่าตัวเลข):** การทดลอง background-sample รอบแรก
pre-register gate ไว้ (median > -5%, draw แย่กว่า -15% <= 1) แล้ว *ทุก* fraction ตก
gate — แต่ผลไม่ monotone กับขนาด sample เลย (fraud@0.1%: -4.9,+1.9,+14.1,-13.1,
+20.2,-1.2) พอวัด noise floor ตรงๆ (bg เต็ม เปลี่ยนแค่ random_state) พบว่า **spread
22.2% ของ median, CV 8.1%** ซึ่ง*กว้างกว่า gate เอง* -> gate นั้นสอบไม่ผ่านโดย
โครงสร้าง แม้เอา background เต็มมาเทียบกับตัวเองก็ตก
**บทเรียน: วัด noise floor ก่อนตั้ง gate ไม่ใช่หลัง** และ confound ที่ทำให้ผลรวน
คือ RPForest fit บน background ตาม parity contract -> เปลี่ยน background sample =
เปลี่ยน geometry realization ไปด้วย แยกสองอย่างไม่ออกถ้าไม่มี control arm

## 8h. Precision/recall investigation — 4 สมมติฐาน ผิดทั้ง 4 (2026-08-29)

**คอขวดคือ variance ไม่ใช่ bias** `variance_decomposition`: ตัวการใหญ่สุดคือ
**seed draw** (CV 18.06%, spread 50.4%) มากกว่า geometry (8.06%) 2.2 เท่า;
`feature_weighting="contrast"` เป็น shared noise จริง (ปิดแล้ว CV 8.06 -> 4.94)
แต่ซื้อความแม่น +13.4% = bias-variance trade ไม่ใช่บั๊ก; ROC นิ่ง (CV 0.3-2.7%)
แต่ PR แกว่ง (5-18%) -> อันดับรวมเสถียร หัวแถวเปราะ

**variance ยุบตัวตามจำนวน seeds** `seed_count_curve` (fraud natural, n_bits=10):
50 -> 4,000 seeds ทำให้ CV(PR) ยุบ 10 เท่า (9.98% -> 0.94%) และ precision@0.1%
+29% (.656 -> .844) -> **ปัญหา variance ที่เห็นทั้งโปรเจกต์เป็น artifact ของ
benchmark regime ที่ใช้ seeds 20-61** โจทย์จริง 30k seeds อยู่คนละที่ 500 เท่า

**สมมติฐานที่ผิด 4 ข้อ (กันเดินซ้ำรอย):**
1. LCB/variance-aware lift จะช่วย precision@k -> กลไกหายเองที่ seeds เยอะ
2. n_bits ที่ดีสุดเลื่อนตาม seeds (crossover) -> อันดับ 12>10>8 เหมือนกันทั้ง
   150 และ 4,000 seeds (5/5 ทั้งคู่)
3. n_bits จะพลิกกลับที่ 14-20 -> ไม่พลิกถึง 20 (1M buckets, 0.004 seeds/bucket,
   99.6% ว่าง แต่ยังดีสุด +5.4% vs 12, 5/5). น่าจะเพราะ hier + multi-probe +
   lift fallback=1 จัดการ bucket บางไว้แล้ว (คำอธิบาย ยังไม่ได้ทดสอบตรงๆ)
4. hyperplane+bits สูง ควรแทน rpforest เป็น default -> ที่ ultra-imbalance
   hyperplane **แพ้** (0.3%: p@0.1% -5.6%/-4.6%, PR -8.9%/-14.6%)

**แผนที่ regime — อันดับ engine พลิกตาม imbalance ไม่ใช่ตาม seeds:**
natural (ทั้ง 150 และ 4,000 seeds) -> hyperplane@16-20 ชนะ (p@0.1% +7.9..+14.9%,
4-5/5, fit เร็วกว่า 3.4x); ultra (ทั้ง 20 และ 61 seeds) -> rpforest ชนะ
-> **default `rare_pu` (rpforest, n_bits=10) ถูกแล้ว ไม่เปลี่ยน**
(n_bits 10->12 ที่ได้ +2.7-3.9% 5/5 ที่ natural นั้น ที่ ultra เหลือ +2.51% PR 3/5,
p@0.1% 0.0% 2/5 = ในกรอบ noise ยังไม่พอเปลี่ยน default)

**ช่องที่วัดไม่ได้:** ultra-imbalance *และ* seeds เยอะพร้อมกัน = โจทย์ PromptPay พอดี
ต้องมี ~7.3M แถวจึงจะได้ ratio 0.1% กับ positives 7,348 (มี 210k) — อนุมานว่า
rpforest ยังชนะเพราะอันดับ engine ไม่เปลี่ยนตาม seeds ภายในแต่ละ imbalance
แต่เป็น **อนุมาน ไม่ใช่ผลวัด**

ดู `outputs/PRECISION_RECALL_INVESTIGATION.md`

## 8. Audit 2026-07-18 — สิ่งที่ยังไม่ได้ทำและ roadmap ถัดไป

### คำตัดสินจากผลล่าสุด

Phase 2B เป็น **improvement เฉพาะบาง dataset ยังไม่ใช่ general improvement**:

| production path, 10% seeds, median 3 draws | Home Credit PR-AUC | Fraud temporal PR-AUC |
|---|---:|---:|
| baseline b10,K80 | 0.1506 | **0.1736** |
| `whiten=True` | 0.1675 (+11%) | 0.1111 (−36%) |
| `feature_weighting="contrast"` | **0.2027 (+35%)** | 0.1677 (−3%) |

Fraud ที่ 25% seeds: contrast 0.1755 เทียบ baseline 0.1872 (−6%). ดังนั้นห้ามเขียนว่า
whiten/contrast ชนะทั่วไป; whiten ต้องเป็น opt-in/tuned และ contrast ต้อง validate ต่อ task.

### P0 — แก้ correctness / evaluation ก่อนทดลอง algorithm เพิ่ม

- [x] ✅ **แก้ Tuner ที่ตัด supervised candidates ทิ้ง:** เงื่อนไข dedupe ของ
  `cat_bit_frac` ตอนนี้บังคับ numeric/whiten ให้ใช้ค่าแรก แต่ supervised ให้ใช้ค่าสุดท้าย
  ทำให้ default grid ไม่เคยประเมิน supervised หรือ supervised+whiten. เพิ่ม regression test
  ที่ assert coverage ของ `(strategy, whiten)` ทุกชุดที่ขอ.
- [x] ✅ **ปิด OOF leakage:** fit supervised projections และ label-derived feature weights
  ภายใน train fold แล้ว transform validation fold; แยก background-fit กับ background-eval
  ใน PU tuner.
- [x] ✅ **ทำ benchmark ↔ production parity:** (PU: encoder+geometry fit จาก background) benchmark fit encoder/projections จาก population
  แต่ `LeadX.fit(X_positive, background=...)` fit encoder จาก positive seeds. ตัดสิน contract
  ให้ชัด (แนะนำ fit y-independent encoder/geometry จาก background population) แล้วให้ harness
  เรียก public API เดียวกับที่ deploy.
- [x] ✅ **แก้ documentation/API example:** contrast ต้องมี `background`; `n_tables=80` ถูก ignore
  เมื่อ `tune=True` เพราะ default grid สูงสุด K=40. ตัวอย่างต้องใช้ `tune=False` หรือเพิ่ม K=80
  ใน grid อย่างตั้งใจ.
- [x] ✅ (บางส่วน) **ซ่อม reproducibility artifacts:** (b8k80 เข้า CONFIGS, provenance note ใน report; combo/verify raw ยังเป็น spot-check ชั้นรอง) ตัวเลข exact-kNN gate ใน report ไม่ตรง raw/current
  script, `b8,K80` มีใน raw แต่ไม่มีใน `CONFIGS`, และผล combo/library verification ไม่มี raw
  artifact. ทุก claim ต้องมี script + raw rows + frozen config.
- [x] ✅ **method identity:** (`whiten_applied_`) บันทึก `whiten_applied` แยกจากค่าที่ผู้ใช้ request; origin mode ที่
  ขอ whiten แต่ถูกข้ามห้าม serialize/รายงานเสมือนว่า whitening ทำงานจริง.

### P1 — improvement ราคาถูกที่ควรลองก่อนเปลี่ยน architecture

1. **Numeric-only PCA/whitening**
   - รักษา type-aware categorical bits ไว้ ไม่ผสม one-hot เข้า PCA ทั้งหมด.
   - ablation `PCA64 no-whiten` vs `PCA64 whiten` เพื่อแยกผล dimensionality reduction
     ออกจาก whitening; ตัด near-zero eigenvalues และ tune shrinkage ของ inverse variance.

2. **Robust PU contrast weights**
   - ใช้ standardized effect size + shrinkage เข้าหา weight=1 แทน normalize ด้วย max อย่างเดียว.
   - tune ความแรง `lambda` แบบ cross-fit; ทดลอง clipped/rank weights กัน seed noise และ
     contaminated background; แยก numeric/categorical relevance.

3. ~~**Overcomplete table selection**~~ — ✅ ทดลองแล้ว **ปฏิเสธ** (§8b: greedy แพ้ "ใช้ทั้งหมด")
   - สร้าง 120–160 candidate tables แล้วเลือก 40–80 จาก held-out marginal PU gain.
   - objective ต้องมี locality quality floor + positive-mode coverage − residual redundancy;
     occupancy/`n_eff` เป็น guardrail ไม่ใช่เป้าหมาย.
   - เทียบ first-40, best-single-table top-40 และ greedy quality+diversity บน codes ชุดเดียว.

4. ~~**Margin-aware multi-probe**~~ — **✅ วัดแล้ว ปฏิเสธ** (ดู §5.4 / §8b;
   outputs/margin_multiprobe_experiment.py)

### P2 — งานวิจัย architecture (ทำเมื่อ P0/P1 ผ่านเท่านั้น)

- [x] ✅ **Balanced Local Projection Forest:** วัดแล้ว — ratio-of-medians gate
  เดิมผ่าน แต่ **paired analysis ผ่านชัดเจน 2/4 extreme cells** → shipped เป็น
  **regime-dependent opt-in** `engine="rpforest"` + engine dimension ใน tuner
  (ดู §8f; mega-bucket hypothesis ยืนยันบนข้อมูลจริง)
- [ ] **Faithful ITQ / Anchor Graph Hashing:** prototype ปัจจุบันไม่ใช่ implementation ตาม
  paper เต็ม (AGH ไม่มี graph eigensystem; ITQ ถูกบีบ PCA-10 + median). ผลลบปัจจุบันปฏิเสธ
  ได้เฉพาะ prototype ไม่ใช่ algorithm family.
- [ ] **Task-aware low-rank metric:** เรียน seed-vs-background geometry แบบ cross-fit แล้ว
  hash ใน metric ใหม่; แยก method marker จาก reusable backbone.
- [ ] **ANN speed frontier:** Exact-KNN ตอบ quality headroom แต่ไม่ตอบ production trade-off;
  benchmark HNSW/Annoy/FAISS ด้วย candidate recall, latency และ memory ก่อนอ้างข้อได้เปรียบ.

### Validation gate ที่ต้องใช้รอบถัดไป

- freeze config จาก validation แล้วเปิด untouched outer test เพียงครั้งเดียว;
- อย่างน้อย Home Credit + Fraud temporal + synthetic categorical/noise stress tests;
- หลาย temporal windows/outer splits ไม่ใช่เปลี่ยน seed subset บน test เดิมอย่างเดียว;
- วัด PR-AUC, P@0.1%, P@1%, recall@review-budget, stability, fit/predict time และ memory;
- ship เมื่อ median PR-AUC +5% relative หรือ P@1% +3pp, ชนะอย่างน้อย 8/10 independent
  outer runs และไม่ทำ secondary dataset ลดเกิน 3%.

### ห้ามเดินซ้ำโดยไม่มีสมมติฐานใหม่

- maximize `n_eff`/entropy ตรงๆ;
- เพิ่ม bits เกินประมาณ 10 ขณะที่ seeds บาง;
- adaptive-prefix scoring อย่างเดียว;
- global pair-difference directions;
- random feature bootstrap/subspace;
- naive seed-mode clustering เมื่อหลักฐานต่อ mode บาง;
- always-on whitening;
- อ้างว่า ITQ/AGH ทั้ง family แพ้จาก proxy experiments ปัจจุบัน.
