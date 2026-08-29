"""streaming scoring — วัด peak RSS และ throughput บน path จริง (parquet บนดิสก์)

โจทย์จริง (PromptPay 500M txn/สัปดาห์) ข้อมูลอยู่บน **ดิสก์** ไม่ใช่ใน RAM
benchmark จึงเขียน parquet ก่อน แล้ววัดสามแบบ แต่ละแบบใน **subprocess แยก**
(ru_maxrss เป็น high-water mark ตลอดอายุ process — ถ้ารันรวมกันจะปนเปื้อน):

  whole      read_parquet ทั้งก้อน -> score_samples(df)          — เส้นฐานเดิม
  streamed   scan_parquet -> score_batches(lf, chunk_rows=C)     — ไม่เคยถือทั้งชุด
  topk       scan_parquet -> top_k(lf, k, chunk_rows=C)          — shortlist ของ cascade

รัน: uv run python outputs/streaming_scale_benchmark.py [n_rows] [n_features]
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

CHUNK = 250_000
TOPK = 10_000
DATA = Path("/tmp/leadx_scale_bench")


def build_parquet(n: int, d: int, rows_per_file: int = 500_000) -> None:
    """เขียนเป็นหลายไฟล์เหมือน partition จริง — ไม่ถือทั้งชุดใน RAM ตอนสร้าง"""
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(d)]
    for fi, off in enumerate(range(0, n, rows_per_file)):
        m = min(rows_per_file, n - off)
        X = rng.standard_normal((m, d), dtype=np.float32)
        if fi == 0:
            X[: max(200, n // 1000)] += 2.0        # positives อยู่ต้นไฟล์แรก
        pl.DataFrame(X, schema=cols).write_parquet(DATA / f"part-{fi:04d}.parquet")
        del X


def peak_gb() -> float:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6   # macOS=bytes, linux=KB


def worker(mode: str, n: int, d: int) -> None:
    from leadx import LeadX
    glob = str(DATA / "*.parquet")
    lf = pl.scan_parquet(glob)
    n_seeds = max(200, n // 1000)
    seeds = lf.slice(0, n_seeds).collect()
    # geometry + สถิติ background จาก sample (lift ใช้ n_total/n_bg_total = สัดส่วนมวล
    # จึงประมาณจาก uniform sample ได้ — ดู BACKGROUND_SAMPLE_REPORT.md)
    bg = lf.gather_every(max(1, n // 200_000)).collect()
    model = LeadX(mode="pu", tune=False, n_bits=10, n_tables=40,
                  score_type="lift", agg="log", random_state=0).fit(seeds, background=bg)
    del bg, seeds
    base = peak_gb()
    t0 = time.perf_counter()
    if mode == "whole":
        s = model.score_samples(pl.read_parquet(glob))
        out = float(s[:5].sum())
    elif mode == "streamed":
        first = None
        for chunk in model.score_batches(pl.scan_parquet(glob), chunk_rows=CHUNK):
            if first is None:
                first = float(chunk[:5].sum())     # ไม่เก็บ score ทั้งชุด
        out = first
    else:
        _, sc = model.top_k(pl.scan_parquet(glob), TOPK, chunk_rows=CHUNK)
        out = float(sc[:5].sum())
    dt = time.perf_counter() - t0
    print("RESULT" + json.dumps({
        "mode": mode, "n_rows": n, "n_features": d, "seconds": round(dt, 1),
        "rows_per_s": int(n / dt), "peak_gb": round(peak_gb(), 2),
        "peak_after_fit_gb": round(base, 2), "checksum": round(out, 6)}))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i + 1], int(sys.argv[i + 2]), int(sys.argv[i + 3]))
        raise SystemExit
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    print(f"n_rows={n:,} n_features={d} chunk_rows={CHUNK:,} top_k={TOPK:,}")
    t0 = time.time()
    build_parquet(n, d)
    size = sum(f.stat().st_size for f in DATA.glob("*.parquet")) / 1e9
    print(f"parquet: {len(list(DATA.glob('*.parquet')))} ไฟล์ {size:.2f} GB "
          f"({time.time()-t0:.0f}s)\n")
    rows = []
    for mode in ("whole", "streamed", "topk"):
        p = subprocess.run([sys.executable, __file__, "--worker", mode, str(n), str(d)],
                           capture_output=True, text=True)
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT")), None)
        if line is None:
            print(f"{mode}: FAILED\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
            continue
        r = json.loads(line[6:])
        rows.append(r)
        print(f"  {r['mode']:<8} {r['seconds']:>6.1f}s  {r['rows_per_s']:>9,} rows/s  "
              f"peak {r['peak_gb']:>5.2f} GB  (after fit {r['peak_after_fit_gb']:.2f})")
    if rows:
        pl.DataFrame(rows).write_csv("outputs/streaming_scale_raw.csv")
        w = next((r for r in rows if r["mode"] == "whole"), None)
        c = next((r for r in rows if r["mode"] == "streamed"), None)
        if w and c:
            same = abs(w["checksum"] - c["checksum"]) < 1e-6
            print(f"\n  peak memory {w['peak_gb']:.2f} -> {c['peak_gb']:.2f} GB "
                  f"({100*(1-c['peak_gb']/w['peak_gb']):.0f}% ลดลง) | scores ตรงกัน: {same}")
            print(f"  extrapolate 500M x {d} feats: whole ~"
                  f"{w['peak_gb']*500e6/n:.0f} GB (ไม่ลง RAM) vs streamed "
                  f"~{c['peak_gb']:.1f} GB (คงที่)")
        print("\nwrote outputs/streaming_scale_raw.csv")
