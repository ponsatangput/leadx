"""streaming scoring — score ประชากรที่ใหญ่กว่า RAM ทีละ chunk

สัญญาที่เทสต์บังคับ (แถวเดียวกันต้องได้ผลเดียวกันไม่ว่าอยู่ chunk ไหน):

1. **bucket codes เท่ากันเป๊ะ bit-for-bit** — ทั้ง encode → weight → hash เป็น
   per-row ล้วน ถ้าข้อนี้พังแปลว่ามี state รั่วข้ามแถวจริง
2. **score เท่ากันภายใน ~ULP และ ranking ตรงกัน** — ไม่ใช่ bit-identical เพราะ
   ขั้นสุดท้าย `exp(log(per_table) @ w)` เป็น BLAS matmul ที่ลำดับการบวกสะสม
   ขึ้นกับจำนวนแถวใน array (วัดได้ <= 2 ULP) นอกเหนือการควบคุมของเรา
"""
import numpy as np
import polars as pl
import pytest

from leadx import LeadX

ULP = dict(rtol=1e-12, atol=0.0)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n, d = 4000, 8
    X = rng.normal(size=(n, d))
    X[:200] += 2.5                       # positives กระจุกอยู่มุมหนึ่ง
    df = pl.DataFrame(X, schema=[f"f{i}" for i in range(d)])
    df = df.with_columns(pl.Series("cat", rng.choice(["a", "b", "c"], n)))
    model = LeadX(mode="pu", tune=False, n_bits=6, n_tables=12,
                  score_type="lift", agg="log", random_state=0)
    model.fit(df[:200], background=df)
    return model, df


def test_bucket_codes_are_bit_identical_across_chunking(fitted):
    """สัญญาข้อ 1 — การแบ่ง chunk ต้องไม่เปลี่ยน bucket ที่แถวตกลงไปแม้แต่บิตเดียว"""
    model, df = fitted
    chunked = np.concatenate(
        [model.transform_codes(df.slice(o, 333)) for o in range(0, df.height, 333)])
    assert np.array_equal(chunked, model.transform_codes(df))


def test_score_batches_matches_single_shot(fitted):
    model, df = fitted
    once = model.score_samples(df)
    streamed = np.concatenate(list(model.score_batches(df, chunk_rows=333)))
    assert streamed.shape == once.shape
    np.testing.assert_allclose(streamed, once, **ULP)


def test_chunking_preserves_ranking_exactly(fitted):
    model, df = fitted
    a = np.argsort(-model.score_samples(df), kind="stable")
    b = np.argsort(-model.score_samples(df, chunk_rows=137), kind="stable")
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("chunk", [1, 7, 999, 10_000])
def test_chunk_size_does_not_change_scores(fitted, chunk):
    model, df = fitted
    small = df[:150]
    np.testing.assert_allclose(
        model.score_samples(small, chunk_rows=chunk), model.score_samples(small), **ULP)


def test_accepts_lazyframe_without_collecting_all(fitted):
    model, df = fitted
    out = np.concatenate(list(model.score_batches(df.lazy(), chunk_rows=500)))
    np.testing.assert_allclose(out, model.score_samples(df), **ULP)


def test_accepts_iterable_of_frames(fitted):
    model, df = fitted
    parts = [df[:1000], df[1000:2500], df[2500:]]
    out = np.concatenate(list(model.score_batches(parts)))
    np.testing.assert_allclose(out, model.score_samples(df), **ULP)


def test_batches_respect_chunk_rows(fitted):
    model, df = fitted
    sizes = [len(b) for b in model.score_batches(df, chunk_rows=1500)]
    assert sizes == [1500, 1500, 1000]


def test_diagnostics_aggregate_across_chunks(fitted):
    model, df = fitted
    model.score_samples(df)
    once = dict(model.diagnostics_["scoring"])
    list(model.score_batches(df, chunk_rows=250))
    streamed = model.diagnostics_["scoring"]
    assert streamed["miss_rate_final"] == pytest.approx(once["miss_rate_final"], abs=1e-12)
    assert streamed["miss_rate_join"] == pytest.approx(once["miss_rate_join"], abs=1e-12)


def test_top_k_matches_full_argsort(fitted):
    model, df = fitted
    s = model.score_samples(df)
    idx, sc = model.top_k(df, 25, chunk_rows=400)
    expect = np.argsort(-s, kind="stable")[:25]
    np.testing.assert_array_equal(idx, expect)
    np.testing.assert_allclose(sc, s[expect], **ULP)


def test_top_k_returns_descending(fitted):
    model, df = fitted
    _, sc = model.top_k(df, 40, chunk_rows=97)
    assert np.all(np.diff(sc) <= 0)


def test_top_k_larger_than_population(fitted):
    model, df = fitted
    idx, sc = model.top_k(df[:50], 500)
    assert len(idx) == 50 and len(sc) == 50


def test_top_k_indices_are_global_not_per_chunk(fitted):
    model, df = fitted
    # positives อยู่ 200 แถวแรก — top_k ต้องคืน index ของ *ทั้งตาราง*
    idx, _ = model.top_k(df, 100, chunk_rows=137)
    assert idx.max() < len(df)
    assert (idx < 200).mean() > 0.5          # ส่วนใหญ่ควรเป็น positive จริง


def test_invalid_chunk_rows_raises(fitted):
    model, df = fitted
    with pytest.raises(ValueError, match="chunk_rows"):
        list(model.score_batches(df, chunk_rows=0))
    with pytest.raises(ValueError, match="chunk_rows"):
        model.score_samples(df, chunk_rows=-5)


def test_top_k_requires_positive_k(fitted):
    model, df = fitted
    with pytest.raises(ValueError, match="k"):
        model.top_k(df, 0)


def test_empty_frame_yields_nothing(fitted):
    model, df = fitted
    assert list(model.score_batches(df[:0])) == []
    idx, sc = model.top_k(df[:0], 5)
    assert len(idx) == 0 and len(sc) == 0
