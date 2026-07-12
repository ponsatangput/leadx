"""Tuner — out-of-fold grid search แบบ nested: hash ครั้งเดียวประเมินทั้ง grid (spec §7)"""
import itertools
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from .hashing import HashEnsemble, mask_codes
from .projections import ProjectionBank
from .scoring import SeedScorer

DEFAULT_GRID = {
    "n_bits": [8, 12, 16, 20, 24],
    "n_tables": [3, 5, 10, 20],
    "m": [1.0, 10.0, 50.0],
    "weights": ["mean", "coverage"],
    "strategy": ["unsupervised", "supervised"],
    "score_type": ["prop", "lift"],
    "alpha": [0.5, 2.0, 10.0],
    "agg": ["mean", "log"],
    "cat_bit_frac": [0.0, "auto"],
}


def stratified_folds(y: np.ndarray, n_folds: int, rng) -> np.ndarray:
    folds = np.empty(len(y), dtype=np.int64)
    for cls in np.unique(y):
        idx = np.nonzero(y == cls)[0]
        folds[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
    return folds


class Tuner:
    """หา config ที่ดีที่สุดจาก out-of-fold validation — hash ครั้งเดียวทั้ง grid (spec §7)

    ประเมิน (strategy, n_bits, n_tables, m, weights) ทั้งกริดจากการ hash ด้วย B_max/
    K_max เพียงครั้งเดียว: n_bits ที่เล็กกว่า = prefix masking, n_tables ที่น้อยกว่า =
    subset ของ tables. two_class วัดด้วย OOF AUC, pu วัดด้วย average precision (ต้องมี
    Z_background เป็น negative proxy).

    Parameters
    ----------
    grid : dict | None
        override DEFAULT_GRID (คีย์: n_bits, n_tables, m, weights, strategy)
    n_folds : int
        จำนวน fold สำหรับ out-of-fold validation
    mode : {"two_class", "pu"}
    random_state : int

    Attributes
    ----------
    results_ : polars.DataFrame
        ทุก config เรียงตาม score (มากไปน้อย)
    best_ : dict
        config ที่ชนะ + score
    """

    def __init__(self, grid=None, n_folds=5, mode="two_class",
                 cat_bit_frac="auto", random_state=0):
        self.grid = {**DEFAULT_GRID, **(grid or {})}
        self.n_folds = n_folds
        self.mode = mode
        self.cat_bit_frac = cat_bit_frac
        self.random_state = random_state

    def tune(self, Z: np.ndarray, y: np.ndarray, Z_background=None,
             cat_mask=None) -> dict:
        if self.mode == "pu" and Z_background is None:
            raise ValueError(
                "PU tuning ต้องการ background sample (unlabeled) เป็น negative proxy "
                "— ส่งผ่าน Z_background"
            )
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        if self.mode == "two_class":
            min_class = int(np.bincount(y.astype(int)).min())
            if min_class < 2:
                raise ValueError(
                    f"OOF tuning ต้องการ seeds อย่างน้อย 2 ต่อ class (class เล็กสุดมี "
                    f"{min_class}) — ไม่งั้น train fold จะว่างเปล่า; เพิ่ม seeds หรือใช้ tune=False"
                )
            n_folds = min(self.n_folds, min_class)
        else:
            n_folds = min(self.n_folds, len(y))
        b_max = max(self.grid["n_bits"])
        k_max = max(self.grid["n_tables"])
        folds = stratified_folds(y, n_folds, rng)
        has_cat = cat_mask is not None and bool(np.any(cat_mask))
        fracs = self.grid["cat_bit_frac"]
        rows = []
        for strategy in self.grid["strategy"]:
            for frac in fracs:
                # dedupe frac: ไม่มี cat columns → frac ไม่มีผล ประเมินค่าแรกค่าเดียว;
                # supervised จัดสรรชนิด bits ด้วย IG เอง → ประเมินครั้งเดียว (ค่าสุดท้าย
                # ซึ่งเปิด typed candidate pool ถ้ามี cat)
                if not has_cat and frac != fracs[0]:
                    continue
                if strategy == "supervised" and frac != fracs[-1]:
                    continue
                frac_val = frac if frac == "auto" else float(frac)
                # bank ต่อ (strategy, frac): frac เปลี่ยน projections → ต้อง re-hash
                # (median mode fit เร็วพอที่จะจ่ายได้; ต่างจาก bits/tables ที่ mask ได้)
                bank = ProjectionBank(
                    n_tables=k_max, n_bits=b_max, strategy=strategy,
                    cat_bit_frac=frac_val, random_state=self.random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bank.fit(Z, y if self.mode == "two_class" else None, cat_mask=cat_mask)
                if strategy == "supervised" and bank.strategy_ == "unsupervised":
                    continue  # guard fallback แล้ว — ประเมินซ้ำกับ unsupervised ไม่มีประโยชน์
                hasher = HashEnsemble(bank.P_, bank.thresholds_)
                codes = hasher.transform(Z)
                codes_bg = hasher.transform(Z_background) if Z_background is not None else None
                for b, k, m, w, st, al, ag in itertools.product(
                    self.grid["n_bits"], self.grid["n_tables"],
                    self.grid["m"], self.grid["weights"],
                    self.grid["score_type"], self.grid["alpha"], self.grid["agg"],
                ):
                    # dedupe: parameter ที่ไม่มีผลกับ score_type นั้นๆ ประเมินค่าแรกค่าเดียว
                    # (prop ไม่ใช้ alpha/agg, lift ไม่ใช้ m) — กัน grid บวมด้วย combo ซ้ำ
                    if st == "prop" and (al != self.grid["alpha"][0] or ag != self.grid["agg"][0]):
                        continue
                    if st == "lift" and m != self.grid["m"][0]:
                        continue
                    fold_scores = []
                    for f in range(n_folds):
                        tr, va = folds != f, folds == f
                        cb = mask_codes(codes, b, b_max)[:, :k]
                        scorer = SeedScorer(mode=self.mode, m=m, weights=w, n_bits=b,
                                            score_type=st, alpha=al, agg=ag)
                        if self.mode == "two_class":
                            scorer.fit(cb[tr], y[tr])
                            fold_scores.append(roc_auc_score(y[va], scorer.score(cb[va])))
                        else:
                            cbg = mask_codes(codes_bg, b, b_max)[:, :k]
                            scorer.fit(cb[tr], y[tr], background_codes=cbg)
                            s = np.concatenate([scorer.score(cb[va]), scorer.score(cbg)])
                            lbl = np.concatenate([np.ones(int(va.sum())), np.zeros(len(cbg))])
                            fold_scores.append(average_precision_score(lbl, s))
                    rows.append({
                        "strategy": strategy, "n_bits": b, "n_tables": k,
                        "m": m, "weights": w, "score_type": st, "alpha": al, "agg": ag,
                        "cat_bit_frac": str(frac), "score": float(np.mean(fold_scores)),
                    })
        self.results_ = pl.DataFrame(rows).sort("score", descending=True)
        self.best_ = self.results_.row(0, named=True)
        return self.best_
