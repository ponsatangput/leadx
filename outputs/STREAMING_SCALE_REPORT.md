# Streaming scoring — วัดจริงบน parquet บนดิสก์ (v0.5, 2026-08)

**โจทย์:** PromptPay 500M txn/สัปดาห์ `score_samples` เดิม preallocate matrix
`(n_rows × n_encoded_dims)` float32 ทีเดียว — ที่ 500M × ~150 features คือหลายร้อย GB
(`chunk_size` ที่มีอยู่เดิมมีผลแค่ตอน hash ไม่ใช่ตอน encode)

## ผลวัด — 10M แถว × 150 features, parquet 20 ไฟล์ 5.55 GB

| mode | เวลา | throughput | peak RSS | peak หลัง fit |
|---|---|---|---|---|
| `whole` — `read_parquet` ทั้งก้อน → `score_samples(df)` | 89.8s | 111k rows/s | **11.32 GB** | 1.25 GB |
| `streamed` — `scan_parquet` → `score_batches(chunk_rows=250k)` | **36.1s** | **277k rows/s** | **1.87 GB** | 1.25 GB |
| `topk` — `scan_parquet` → `top_k(k=10k, chunk_rows=250k)` | 35.3s | 283k rows/s | 1.92 GB | 1.25 GB |

- **peak memory 11.32 → 1.87 GB (−83%)** และคะแนนตรงกัน (checksum เท่ากัน)
- peak ของขั้น fit = **1.25 GB เท่ากันทั้งสามแบบ** (background sample) — ส่วนที่เหลือ
  คือขั้น score ล้วน: `whole` +10.1 GB ตามจำนวนแถว, `streamed` +0.62 GB ตาม chunk
- **streaming เร็วกว่า 2.5×** ด้วย ไม่ใช่แค่ประหยัด memory — `whole` เสียเวลาไปกับ
  allocator/memory pressure (อาการเดียวกับที่เคยเจอตอนแก้ encoder ใน §8 DESIGN_NOTES)

**Extrapolate ไป 500M × 150:** `whole` ต้องการ ~566 GB (ไม่ลง RAM เครื่องไหน)
เทียบ `streamed` ที่ **คงที่ ~1.9 GB** ไม่ขึ้นกับจำนวนแถว ที่ 277k rows/s
500M แถวใช้เวลา ~30 นาที single-process

## สิ่งที่เพิ่มเข้า API

```python
model.score_batches(source, chunk_rows)    # generator, memory คงที่
model.top_k(source, k, chunk_rows)         # shortlist, memory O(k + chunk)
model.score_samples(df, chunk_rows=None)   # backward compatible

clf.decision_function(X, chunk_rows=...)   # sklearn API
clf.review_queue(X, k=..., chunk_rows=...) # cascade shortlist ที่ scale ได้
```

`source` รับ `pl.DataFrame`, `pl.LazyFrame` (slice ก่อน collect), หรือ iterable
ของสองอย่างนี้ — `pl.scan_parquet("week=*/*.parquet")` จึงอ่านทีละ chunk

## สัญญาความถูกต้องที่เทสต์บังคับ (tests/test_streaming.py, 24 เทสต์)

1. **bucket codes เท่ากันเป๊ะ bit-for-bit** ข้ามทุกการแบ่ง chunk — ทั้ง
   encode → weight → hash เป็น per-row ล้วน (encoder ใช้สถิติที่ตรึงตอน fit,
   `normalize_pu` ใช้ค่าคงที่จาก fit)
2. **ranking ตรงกันเป๊ะ** และคะแนนต่างไม่เกิน ~ULP — วัดได้ **≤2 ULP** ซึ่งมาจาก
   `exp(log(per_table) @ w)` ขั้นสุดท้ายเท่านั้น (BLAS เปลี่ยนลำดับการบวกสะสม
   ตามจำนวนแถวใน array) ไม่ใช่จาก state รั่วข้ามแถว
3. `top_k` คืน index ของ **ทั้ง population** ไม่ใช่ภายใน chunk, tie แตกด้วย index
   น้อยก่อน — ตรงกับ `np.argsort(-s, kind="stable")[:k]`

## สิ่งที่พบระหว่างวัด (แก้แล้ว)

การวัดรอบแรกได้แค่ −52% เพราะ **ตัว benchmark เองมีข้อบกพร่อง**: ใช้
`scan_parquet(...).gather_every(k)` สุ่ม background ซึ่ง polars ต้อง materialize
ทุกแถวก่อน → peak ของขั้น fit พุ่งเป็น 8.4 GB เท่ากับอ่านทั้งไฟล์ แก้เป็น
**อ่านเฉพาะบาง partition** (`read_parquet(files[:n//20])`) → peak ขั้น fit ลงเป็น
1.25 GB นี่เป็นสูตรที่ใช้ได้จริงที่ 500M ด้วย (อย่าใช้ `gather_every` บน scan)

## ขอบเขต

- ข้อมูล synthetic (gaussian) 150 numeric columns — ไม่มี categorical/MinHash;
  throughput บนข้อมูลจริงจะต่างไปตาม cardinality และ dtype
- วัดที่ 10M แล้ว extrapolate ไป 500M เชิงเส้น — ยังไม่ได้รัน 500M จริง
- `background_policy="refresh"` ใช้ streaming ไม่ได้ (transductive — ต้องเห็นทุกแถว
  พร้อมกัน) API จะ raise พร้อมคำอธิบาย
- single-process — ยังไม่ได้วัดแบบขนานหลาย worker

## Reproduce
```bash
cd sandbox/data_science/leadx
uv run python outputs/streaming_scale_benchmark.py 10000000 150
#   → outputs/streaming_scale_raw.csv (source of truth)
```
