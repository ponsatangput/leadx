# Scale validation 1M–10M แถว — จุดเด่น "train ครั้งเดียว ใช้หลายโจทย์" ที่ scale จริง (2026-07-18)

Benchmark ก่อนหน้าทั้งหมดวัดที่ 200–300k แถว แต่โจทย์ตั้งต้นคือ **10M แถว × 100
features** — รายงานนี้วัดที่ scale นั้นจริง (synthetic: 95 numeric + 5 categorical,
5 PU tasks ปลูก signal คนละกลุ่ม features, positives ~1%, seeds 10% ของ positives)
เครื่องอ้างอิง: Apple Silicon 24GB RAM · script: scale_validation.py (raw CSV = source of truth)

## ผลหลัง fix (สองบั๊ก scalability ที่เจอจากการวัดรอบแรก — ดูท้ายรายงาน)

| n_rows | backbone.fit (ครั้งเดียว) | โจทย์แรก (สร้าง bg stats) | **โจทย์ถัดไป** | score ประชากรทั้งก้อน | HistGB ต่อโจทย์ (label เต็ม) | ROC median | peak RSS |
|---|---|---|---|---|---|---|---|
| 1M | 9.6s | 2.2s | **0.02s** | 3.0s | 10.7s | 0.873 | 4.6 GB |
| 3M | 17.4s | 6.5s | **0.03s** | 9.6s | 33.9s | 0.903 | 10.9 GB |
| 10M | 50.8s | 21.1s | **0.04s** | 31.2s | ~115s (extrapolate linear) | 0.924 | 12.2 GB |

- config: backbone b16 K40; tasks ที่ b10 prefix + lift/log + hier_depths="auto" +
  `seed_rows` + `background="cached"` — ทุกอย่างคือ API จริงที่ผู้ใช้เรียกได้
- HistGB ที่ 10M ข้ามการวัดตรง (กัน swap บนเครื่อง 24GB) — extrapolate เชิงเส้นจาก
  10.7s/33.9s ซึ่ง conservative

## อ่านผล

1. **จุดเด่นข้อ 1 ยืนที่ 10M**: จ่าย backbone 51s + โจทย์แรก 21s ครั้งเดียว
   → ทุกโจทย์ถัดไป **~40ms** (group-by ของ seeds เท่านั้น) เทียบ HistGB ~115s
   ต่อโจทย์ที่ต้อง label เต็ม: 20 โจทย์ = 73s vs ~2,300s (**~30x**); ยิ่งหลายโจทย์
   ยิ่งห่าง (asymptote = 0.04s vs 115s ต่อโจทย์เพิ่ม ≈ 3,000x marginal)
2. **Linear scaling ทุก stage** (หลัง fix): backbone 9.6→17.4→50.8s,
   score 3.0→9.6→31.2s ต่อ 1M→3M→10M; peak RSS 12.2GB ที่ 10M — เครื่อง
   laptop 24GB เอาอยู่ ไม่ต้อง out-of-core ที่ scale นี้
3. **ความแม่นไม่เสื่อมตาม scale** — ROC ดีขึ้นด้วยซ้ำ (0.873→0.924) เพราะ
   background ใหญ่ขึ้น = ตาราง denominator นิ่งขึ้น ที่ positives คงสัดส่วน 1%
4. **throughput จัดอันดับ**: 10M แถว / 31s ≈ 320k แถว/วินาที (K=40, join path)

## สองบั๊ก scalability ที่การวัดรอบแรกเปิดโปง (แก้แล้ว — นี่คือคุณค่าของการวัดที่ scale จริง)

| อาการรอบแรก | root cause | fix | ผล |
|---|---|---|---|
| backbone 10M = **351s** (super-linear 15x) | `FeatureEncoder.transform` สะสม blocks f64 + `np.hstack` → peak ~35GB → macOS memory compression | preallocate output f32 เขียนต่อ column-slice (ค่าเท่าเดิมเป๊ะ) | 318s → **25s**; backbone รวม 51s |
| fit_task = **21s ทุกโจทย์** | group-by background 40 tables × 10M + hier uniques เป็น task-independent แต่คำนวณซ้ำทุกโจทย์ | `BackgroundStats` memoize บน backbone ต่อ (b, k, depths) เมื่อ `background="cached"` | โจทย์ 2+ = **0.04s** (~500x) |

ทั้งคู่มี equivalence test คุม (ผล identical กับ path เดิม) — ROC ทุกค่าตรงกับรอบแรกเป๊ะ

## Caveats

- synthetic data (gaussian + planted linear signal) — เพียงพอสำหรับคำถาม
  เวลา/หน่วยความจำ/เสถียรภาพ แต่ตัวเลข ROC ไม่ใช่ claim ความแม่นบนข้อมูลจริง
  (อันนั้นดู REAL_DATA / IMBALANCE_FOCUS reports)
- HistGB 10M เป็น extrapolation; single draw ต่อ scale (timing variance ต่ำ
  แต่ไม่มี error bar)
- `background="cached"` + `score_type="prop"` ไม่ผ่าน stats cache (ต้องใช้ raw
  codes คำนวณ normalize range) — path deploy จริงคือ lift อยู่แล้ว

## Reproduce

```bash
cd sandbox/data_science/leadx
PYTHONPATH=outputs uv run python outputs/scale_validation.py
```
