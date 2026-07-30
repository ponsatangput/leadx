# Criteo ultra-extreme supervised imbalance arena (2026-07-21)

## คำตัดสิน

ผลรองรับว่าอาจมี **crossover เฉพาะระดับประมาณ 1:10,000** แต่ยังไม่ใช่ชัยชนะที่
ยืนยันแล้ว:

- ที่ **1:1,000 (0.1%)** LeadX RPForest แพ้ strongest supervised baseline
  ทุก metric และทุก split
- ที่ **1:10,000 (0.01%)** LeadX ชนะ PR-AUC 2/3 splits, paired median
  **+3.3%**, และ Recall@0.5% ชนะ 2/3 splits **+0.5 percentage point**
- แต่ Recall@0.01% ซึ่งเป็น shortlist ที่แคบที่สุดยังแพ้ median 1.0 percentage
  point; Recall@0.1% ยังแพ้ 0.5 point

ดังนั้นคำอ้างที่ปลอดภัยคือ:

> LeadX ไม่ได้เด่นจาก imbalance ทั่วไปหรือแม้แต่ 1:1,000 แต่มี preliminary
> evidence ว่า PR ranking และ broad-shortlist recall อาจเริ่ม crossover เมื่อ
> imbalance รุนแรงถึงประมาณ 1:10,000.

## Paired results เทียบ strongest supervised baseline ต่อ split/metric

| regime | metric | paired median effect | 3-split bootstrap range | wins |
|---|---|---:|---:|---:|
| 1:1,000 | PR-AUC | **−15.0% relative** | [−18.0%, −9.7%] | 0/3 |
| 1:1,000 | Recall@0.01% | −0.70 pp | [−0.75, −0.60] | 0/3 |
| 1:1,000 | Recall@0.1% | −2.35 pp | [−3.41, −0.70] | 0/3 |
| 1:1,000 | Recall@0.5% | −2.10 pp | [−2.40, −1.40] | 0/3 |
| **1:10,000** | **PR-AUC** | **+3.3% relative** | **[−7.1%, +6.2%]** | **2/3** |
| 1:10,000 | Recall@0.01% | −1.00 pp | [−1.01, 0.00] | 0/3 (+1 tie) |
| 1:10,000 | Recall@0.1% | −0.50 pp | [−0.50, +1.50] | 1/3 |
| **1:10,000** | **Recall@0.5%** | **+0.50 pp** | **[−0.50, +1.01]** | **2/3** |

ช่วง bootstrap ที่ 1:10,000 คร่อมศูนย์และมีเพียง 3 splits จึงเป็น descriptive
signal ไม่ใช่ confirmatory significance.

## Raw medians

| regime | model | PR-AUC | Recall@0.01% | Recall@0.1% | Recall@0.5% |
|---|---|---:|---:|---:|---:|
| 1:1,000 | HistGB balanced | **.09638** | **.0281** | **.1844** | **.4394** |
| 1:1,000 | LeadX RPForest | .08193 | .0220 | .1678 | .4208 |
| 1:10,000 | HistGB plain | **.01017** | **.0352** | **.1859** | .4070 |
| 1:10,000 | **LeadX RPForest** | .00944 | .0251 | .1809 | **.4221** |

Raw medians อาจมาจากคนละ split จึงห้ามนำมาหารกันเพื่ออ้าง effect; ใช้ paired
table ด้านบน. PR-AUC `.00944` ที่ prevalence `.0001` เท่ากับประมาณ 94 เท่าของ
random baseline แต่ยังต่ำในเชิงสัมบูรณ์.

## Protocol

- Criteo Uplift v2.1 uniform sample 10,000,000 rows, features `f0..f11`
- feature-group splits ใหม่ `51001,51002,51003`, 60/20/20; exact feature
  duplicates ไม่ข้าม split
- หลัง split เก็บ negatives ทุกแถวและ downsample เฉพาะ positives แบบ nested:
  - 1:1,000: train positives 5,988–5,991; test 1,996–1,997
  - 1:10,000: train positives 598–599; test 199–200
- ทุกโมเดลเห็น labels positive/negative ครบ; **ไม่ใช่ PU benchmark**
- LeadX: two-class RPForest, `b10 K80`, correlation feature weighting,
  proportion scoring; config ล็อกก่อนรัน
- baselines: balanced Logistic Regression, HistGradientBoosting plain และ
  balanced; เลือก strongest baseline ภายใน split และ metric เดียวกัน
- PR รายงาน relative effect; recall รายงาน percentage-point difference เพื่อให้
  นิยามยังถูกเมื่อ recall ของ baseline เป็นศูนย์

## เวลา

| model | median fit | median score ~2M test rows |
|---|---:|---:|
| LeadX RPForest | 81.8s | 21.3s |
| HistGB plain | 13.4–13.6s | 1.0–1.1s |
| HistGB balanced | 15.6–16.5s | 1.2–1.5s |
| LogReg balanced | 2.9–3.0s | .012s |

LeadX ไม่มี speed advantage บน Criteo 12 dense features แม้ใน regime ที่ accuracy
เริ่ม crossover.

## ข้อจำกัดและคำตัดสินถัดไป

- dataset เดียว, 3 partitions ของ sample เดียว ไม่ใช่ 3 datasets อิสระ
- 1:10,000 result มี test positives เพียงประมาณ 200 ต่อ split; metric ที่แคบมาก
  จึงมีความละเอียดจำกัด
- ผลที่สมควรตรวจต่อคือ PR-AUC และ Recall@0.5%; top 0.01% ยังไม่มีสัญญาณชนะ
- splits ชุดนี้ต้อง freeze เป็น evidence; ห้าม tune algorithm แล้วกลับมาประกาศ
  confirmation บนชุดเดิม

ขั้นถัดไปที่ให้ข้อมูลใหม่จริงคือ rerun protocol 1:10,000 บน dataset อิสระที่มี
features กว้างกว่า หรือ pre-register split ใหม่หลัง algorithm ถูก freeze.

## Artifacts

- `outputs/criteo_ultra_imbalance_benchmark.py`
- `outputs/criteo_ultra_imbalance_raw.csv`
- `outputs/criteo_ultra_imbalance_summary.csv`
- `outputs/criteo_ultra_imbalance_metadata.json`

```bash
uv run python outputs/criteo_ultra_imbalance_benchmark.py \
  --max-rows 10000000 --target-ratios 0.001,0.0001 \
  --split-seeds 51001,51002,51003 \
  --output outputs/criteo_ultra_imbalance_raw.csv --overwrite
```
