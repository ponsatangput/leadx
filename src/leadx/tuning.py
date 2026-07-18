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
    "n_tables": [3, 5, 10, 20, 40],
    "m": [1.0, 10.0, 50.0],
    "weights": ["mean", "coverage"],
    "strategy": ["unsupervised", "supervised"],
    "score_type": ["prop", "lift"],
    "alpha": [0.5, 2.0, 10.0],
    "agg": ["mean", "log"],
    "cat_bit_frac": [0.0, "auto"],
    "whiten": [False, True],
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
                 cat_bit_frac="auto", hash_mode="median", whiten=False,
                 random_state=0):
        self.grid = {**DEFAULT_GRID, **(grid or {})}
        self.n_folds = n_folds
        self.mode = mode
        self.cat_bit_frac = cat_bit_frac
        self.hash_mode = hash_mode
        self.whiten = whiten
        self.random_state = random_state

    @staticmethod
    def _fold_weights(fw, Z, y, tr_mask, Z_bg_fit):
        """weights จาก train fold เท่านั้น (สูตรเดียวกับ LeadX._compute_weights)"""
        if fw == "corr":
            Zt, yt = Z[tr_mask], y[tr_mask]
            yc = yt - yt.mean()
            denom = np.sqrt((Zt ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-12
            return np.abs((Zt * yc[:, None]).sum(axis=0) / denom).astype(np.float32)
        # contrast: seeds ใน train fold vs background ฝั่ง fit
        w = np.abs(Z[tr_mask].mean(axis=0) - Z_bg_fit.mean(axis=0)) + 1e-3
        return (w / w.max()).astype(np.float32)

    def tune(self, Z: np.ndarray, y: np.ndarray, Z_background=None,
             cat_mask=None, Z_geom=None, feature_weighting="none") -> dict:
        """Z_geom: ข้อมูลสำหรับ fit geometry (parity: PU ใช้ background population);
        default = Z (seeds). feature_weighting: ส่ง Z แบบ *ยังไม่ถ่วงน้ำหนัก* มา —
        tuner จะ cross-fit weights ต่อ fold เอง (ปิด OOF leak — audit P0)"""
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
        # normalize weighting ตาม mode (corr → two_class เท่านั้น, contrast → pu เท่านั้น)
        fw = feature_weighting
        if (fw == "corr" and self.mode != "two_class") or (
                fw == "contrast" and self.mode != "pu"):
            fw = "none"
        # PU: แบ่ง background เป็น fit/eval คนละครึ่ง — ปิด self-scoring leak (audit P0):
        # เดิมแถว bg ที่ใช้สร้างตาราง (denominator) ถูกใช้เป็น negative proxy ตอนวัดด้วย
        if self.mode == "pu":
            bg_perm = np.random.default_rng(self.random_state + 1).permutation(
                Z_background.shape[0])
            bg_fit_idx, bg_eval_idx = bg_perm[::2], bg_perm[1::2]
            Zbg_fit, Zbg_eval = Z_background[bg_fit_idx], Z_background[bg_eval_idx]
        rows, banks = [], {}
        for strategy in self.grid["strategy"]:
          for wh in self.grid["whiten"]:
            # dedupe cat_bit_frac ต่อ (strategy, whiten): เลือก "ชุด fracs ที่ต้องประเมิน"
            # เป็นก้อนเดียว — เดิมใช้เงื่อนไข skip อิสระสามข้อที่ขัดกันเอง ทำให้
            # supervised ไม่เคยถูกประเมินบนข้อมูล numeric-only และ supervised+whiten
            # หายไปทุกกรณี (พบโดย external audit — DESIGN_NOTES §8 P0)
            if not has_cat or wh:
                fracs_eval = [fracs[0]]        # frac ไม่มีผล (ไม่มี cat / whiten ผสมทุกมิติ)
            elif strategy == "supervised":
                fracs_eval = [fracs[-1]]       # supervised จัดสรรชนิด bits ผ่าน IG pool เอง
            else:
                fracs_eval = fracs
            for frac in fracs_eval:
                frac_val = frac if frac == "auto" else float(frac)
                # geometry ที่ใช้ y (supervised strategy / feature weights) ต้อง cross-fit
                # ต่อ fold — validation fold ห้ามเห็น y ของตัวเองผ่าน geometry (audit P0)
                label_dep = (strategy == "supervised") or (fw != "none")

                def make_codes(tr_mask):
                    """สร้าง bank+codes; tr_mask=None → label-free (ใช้ข้อมูลทั้งหมดได้)"""
                    w = (self._fold_weights(fw, Z, y, tr_mask,
                                            Zbg_fit if self.mode == "pu" else None)
                         if (fw != "none" and tr_mask is not None) else None)
                    Zw = Z if w is None else Z * w
                    Zg = Z if Z_geom is None else Z_geom
                    Zg = Zg if w is None else Zg * w
                    bank = ProjectionBank(
                        n_tables=k_max, n_bits=b_max, strategy=strategy,
                        hash_mode=self.hash_mode, cat_bit_frac=frac_val,
                        whiten=wh, random_state=self.random_state,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if strategy == "supervised" and tr_mask is not None:
                            bank.fit(Zw[tr_mask], y[tr_mask], cat_mask=cat_mask)
                        else:
                            bank.fit(Zg, y if self.mode == "two_class" else None,
                                     cat_mask=cat_mask)
                    hasher = HashEnsemble(bank.P_, bank.thresholds_)
                    codes = hasher.transform(Zw)
                    if self.mode == "pu":
                        zf = Zbg_fit if w is None else Zbg_fit * w
                        ze = Zbg_eval if w is None else Zbg_eval * w
                        return bank, codes, hasher.transform(zf), hasher.transform(ze)
                    return bank, codes, None, None

                if label_dep:
                    fold_codes, skip = {}, False
                    for f in range(n_folds):
                        bank_f, c, cf, ce = make_codes(folds != f)
                        if strategy == "supervised" and bank_f.strategy_ == "unsupervised":
                            skip = True
                            break
                        fold_codes[f] = (c, cf, ce)
                    if skip:
                        continue
                    # ไม่มี bank เดี่ยวให้ reuse — facade จะ refit เต็มสำหรับ deploy
                else:
                    bank, c, cf, ce = make_codes(None)
                    if strategy == "supervised" and bank.strategy_ == "unsupervised":
                        continue
                    banks[(strategy, str(frac), wh)] = bank
                    fold_codes = {f: (c, cf, ce) for f in range(n_folds)}
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
                        codes_f, cbgf_f, cbge_f = fold_codes[f]
                        cb = mask_codes(codes_f, b, b_max)[:, :k]
                        scorer = SeedScorer(mode=self.mode, m=m, weights=w, n_bits=b,
                                            score_type=st, alpha=al, agg=ag)
                        if self.mode == "two_class":
                            scorer.fit(cb[tr], y[tr])
                            fold_scores.append(roc_auc_score(y[va], scorer.score(cb[va])))
                        else:
                            cbgf = mask_codes(cbgf_f, b, b_max)[:, :k]
                            cbge = mask_codes(cbge_f, b, b_max)[:, :k]
                            scorer.fit(cb[tr], y[tr], background_codes=cbgf)
                            s = np.concatenate([scorer.score(cb[va]), scorer.score(cbge)])
                            lbl = np.concatenate([np.ones(int(va.sum())), np.zeros(len(cbge))])
                            fold_scores.append(average_precision_score(lbl, s))
                    rows.append({
                        "strategy": strategy, "n_bits": b, "n_tables": k,
                        "m": m, "weights": w, "score_type": st, "alpha": al, "agg": ag,
                        "cat_bit_frac": str(frac), "whiten": wh,
                        "score": float(np.mean(fold_scores)),
                    })
        self.results_ = pl.DataFrame(rows).sort("score", descending=True)
        self.best_ = self.results_.row(0, named=True)
        # เก็บ bank ที่ชนะ — facade ใช้ prefix ของมันเป็นโมเดลสุดท้ายตรงๆ
        # (สิ่งที่ deploy = สิ่งที่ถูกประเมิน ไม่ใช่ refit ใหม่)
        # None เมื่อ winning combo เป็น label-dependent (cross-fit ต่อ fold ไม่มี bank
        # เดี่ยว) — facade ต้อง refit เต็มบนข้อมูลทั้งหมดสำหรับ deploy
        self.best_bank_ = banks.get((self.best_["strategy"], self.best_["cat_bit_frac"],
                                     self.best_["whiten"]))
        return self.best_
