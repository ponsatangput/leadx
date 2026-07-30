# Criteo fully-supervised imbalance arena (2026-07-21)

> **Newer ultra-extreme stress test:** the conclusion below holds at the natural
> ~1:338 ratio and remains negative at 1:1,000. At 1:10,000, LeadX shows a
> preliminary PR/Recall@0.5% crossover; see
> [`CRITEO_ULTRA_IMBALANCE_REPORT.md`](CRITEO_ULTRA_IMBALANCE_REPORT.md).

## คำตัดสิน

**LeadX ไม่ชนะเมื่อโจทย์คือ extreme imbalance อย่างเดียวและทุกโมเดลได้ labels
ครบเท่ากัน.** บน Criteo conversion ที่ observed rate ประมาณ 0.3%, LeadX
two-class RPForest แพ้ strongest supervised baseline ทุก split ทั้ง PR-AUC และ
Recall@0.5%.

| LeadX arm | metric | paired median vs best supervised | 5-split bootstrap interval | wins |
|---|---|---:|---:|---:|
| RPForest + corr | PR-AUC | **−4.2%** | [−11.1%, −1.7%] | 0/5 |
| RPForest + corr | Recall@0.1% | −5.1% | [−8.4%, +6.5%] | 1/5 |
| RPForest + corr | Recall@0.5% | **−3.9%** | [−8.8%, −0.5%] | 0/5 |
| Hyperplane + corr | PR-AUC | −11.9% | [−19.4%, −5.9%] | 0/5 |
| Hyperplane + corr | Recall@0.5% | −6.1% | [−10.9%, −2.3%] | 0/5 |

ผลนี้แยกตัวแปรสำคัญได้ชัด: **class prevalence ต่ำอย่างเดียวไม่ใช่ moat ของ
LeadX.** สัญญาณก่อนหน้าที่เกิดเมื่อมี confirmed positives หลักสิบจึงมาจาก
label/seed scarcity และ retrieval regime ไม่ใช่จาก imbalance ratio เพียงอย่างเดียว.

## Protocol

- Criteo Uplift v2.1 uniform sample 1,000,000 rows; target = conversion
- observed positive rate ประมาณ 0.296%; train positives ประมาณ 1,750 ต่อ split
- feature-group splits ใหม่ `41001..41005`, 60/20/20; exact duplicate feature
  rows ไม่ข้าม split
- **ทุกโมเดลเห็น labels 0/1 ครบทุก train row**; ไม่มี PU arm และไม่มีการจำลอง
  confirmed-seed budget
- LeadX สอง arms: shipped fixed `b10 K80`, two-class proportion scoring,
  correlation feature weights, hyperplane หรือ RPForest
- supervised baselines บน encoder เดียวกัน: balanced Logistic Regression,
  HistGradientBoosting plain และ class-balanced
- comparator เลือก strongest baseline ภายใน split และ metric เดียวกัน
- interval bootstrap 100,000 ครั้งบน 5 splits; เป็น descriptive เพราะทั้งหมด
  เป็น partitions ของ sample เดียว ไม่ใช่ datasets อิสระ

## Raw medians ข้าม 5 splits

| model | PR-AUC | Recall@0.1% | Recall@0.5% | fit | score |
|---|---:|---:|---:|---:|---:|
| HistGB balanced | **.1996** | **.1382** | **.3876** | 2.13s | .139s |
| HistGB plain | .1939 | .1370 | .3855 | **2.08s** | .122s |
| LeadX RPForest + corr | .1884 | .1376 | .3826 | 25.33s | 2.14s |
| LogReg balanced | .1818 | .1317 | .3758 | .267s | **.001s** |
| LeadX hyperplane + corr | .1759 | .1157 | .3713 | 11.05s | .613s |

Ratio ของ raw medians ไม่ใช่ paired effect; ใช้ตารางคำตัดสินด้านบน. บนข้อมูล
12 dense features นี้ LeadX แพ้ทั้ง accuracy และเวลา end-to-end.

## ขอบเขตของข้อสรุป

- นี่เป็นหลักฐานที่ค่อนข้างตรงสำหรับ “imbalance-only” แต่ยังเป็น dataset เดียว
- LeadX ใช้ config คงที่ที่มีอยู่ ไม่ได้ tune ใหม่บน splits เหล่านี้; การ tune
  หลังเห็นผลจะทำให้ชุดนี้กลายเป็น development set และห้ามเรียก confirmation
- ผลไม่ได้หักล้าง use case ที่มี confirmed events เพียงหลักสิบ เพราะสนามนี้มี
  positive labels หลักพันและ reliable negative labels ครบ

ดังนั้น positioning ที่หลักฐานรองรับกว่า คือ **few-confirmed-event shortlist
retrieval** ไม่ใช่ general extreme-imbalance classification. ในทางผลิตภัณฑ์ไม่
จำเป็นต้องขายคำว่า PU แต่ benchmark ยังต้องเก็บ label-budget-matched PU/one-class
methods เป็น controls เพื่อความแฟร์.

## Artifacts และ reproduce

- `outputs/criteo_supervised_imbalance_benchmark.py`
- `outputs/criteo_supervised_imbalance_raw.csv`
- `outputs/criteo_supervised_imbalance_summary.csv`
- `outputs/criteo_supervised_imbalance_metadata.json`

```bash
uv run python outputs/criteo_supervised_imbalance_benchmark.py \
  --max-rows 1000000 \
  --split-seeds 41001,41002,41003,41004,41005 \
  --output outputs/criteo_supervised_imbalance_raw.csv --overwrite
```
