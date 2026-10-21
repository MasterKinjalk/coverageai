"""Predictive model of spray outcomes (WP-10): implements ``contracts.Model``.

Outputs (all from gradient-boosted trees on ``features.py`` features):
  * coverage_pct     point: LightGBM regression; interval: LightGBM quantile 0.1 / 0.9,
                     optionally conformalized (CQR) on a held-out calibration fraction
  * deposition       modeled as retention = deposition / (GPA * 0.09354), then rescaled
  * drift_frac       clipped to [0, 1]
  * displacement_mm

``fit(records)`` follows the loop contract: it receives only NEW records, appends them
to the stored training data (the initial dataset plus all earlier records) and refits.
``fit_frame(df)`` replaces the stored data (initial training from a dataset frame).

Backend (D18): ``backend="lightgbm"`` (default, CPU; the v1 model) or ``backend="xgboost"``
(optional extra ``gpu``; trains on ``device="cuda"`` with depth-wise trees of ``max_depth``,
predicts on the CPU with one thread like the LightGBM path). Same features, targets,
quantile interval and CQR offset. ``n_estimators`` / ``quantile_n_estimators`` may be one
int or a dict per booster name (``TARGETS`` / ``cov_lo``, ``cov_hi``).
Baselines for comparison (``RidgeBaseline``, ``ForestBaseline``) share the feature
builder and point-prediction API but are not used by the loop.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import lightgbm as lgb
import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, Outcome, Prediction, Setting
from coverageai.model import features as F

TARGETS = ("coverage_pct", "deposition_ul_cm2", "drift_frac", "displacement_mm")
DEFAULT_PATH = ROOT / "models" / "spray_model_v1.joblib"
_GPA_UL = float(spec()["physics"]["gpa_to_ul_cm2"])
_GPA_COL = F.feature_index("gpa")
_MIN_FIT_ROWS = 20
BACKENDS = ("lightgbm", "xgboost")


def _per_booster(n, name: str) -> int:
    """Rounds or depth: one int for every booster, or a dict keyed by booster name."""
    return int(n[name]) if isinstance(n, dict) else int(n)


def xgb_train_params(p: dict, n_jobs: int, n_rows: int, name: str, **extra) -> dict:
    """XGBoost parameters matching the LightGBM settings of ``SprayModel._refit``.

    Depth-wise trees: on the RTX 5070 Ti they train about 3x faster on the GPU than on the
    CPU, while leaf-wise (loss-guided) trees are no faster on the GPU (1M rows, 200 rounds).
    min_child_weight = minimum hessian sum = minimum row count for unit weights."""
    return dict(tree_method="hist", device=p["device"], learning_rate=p["learning_rate"],
                max_depth=_per_booster(p["max_depth"], name),
                min_child_weight=min(p["min_child_samples"], max(2, n_rows // 10)),
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, nthread=n_jobs,
                seed=p["seed"], verbosity=0, **extra)


def _xgb_cpu(booster, n_threads: int = 1):
    """Move a booster to CPU prediction (loop workers run in parallel processes)."""
    booster.set_param({"device": "cpu", "nthread": n_threads})
    return booster


def _to_internal(Y: np.ndarray, Fm: np.ndarray) -> np.ndarray:
    """Raw targets (cov, dep, drift, disp) -> training targets (dep becomes retention)."""
    Z = Y.copy()
    Z[:, 1] = Y[:, 1] / (Fm[:, _GPA_COL] * _GPA_UL)
    return Z


def _from_internal(Z: np.ndarray, Fm: np.ndarray) -> np.ndarray:
    out = Z.copy()
    out[:, 0] = np.clip(Z[:, 0], 0.0, 100.0)
    out[:, 1] = np.clip(Z[:, 1], 0.0, 1.5) * Fm[:, _GPA_COL] * _GPA_UL
    out[:, 2] = np.clip(Z[:, 2], 0.0, 1.0)
    out[:, 3] = np.maximum(Z[:, 3], 0.0)
    return out


def records_to_arrays(records: Iterable[tuple[Conditions, Setting, Outcome]]):
    recs = list(records)
    xs = [r[0] for r in recs]
    us = [r[1] for r in recs]
    Y = np.array([[r[2].coverage_pct, r[2].deposition_ul_cm2, r[2].drift_frac,
                   r[2].displacement_mm] for r in recs], dtype=float).reshape(len(recs), 4)
    return F.featurize(xs, us), Y


class SprayModel:
    """LightGBM multi-output predictor implementing the Model protocol."""

    def __init__(self, n_estimators: int | dict = 100, learning_rate: float = 0.15,
                 num_leaves: int = 31, min_child_samples: int = 20,
                 quantile_n_estimators: int | dict = 60, interval: str = "cqr",
                 q_lo: float = 0.1, q_hi: float = 0.9, cal_frac: float = 0.15,
                 record_weight: float = 1.0, seed: int = 0, n_jobs: int | None = None,
                 backend: str = "lightgbm", device: str = "cpu", max_depth: int | dict = 8):
        if interval not in ("quantile", "cqr"):
            raise ValueError("interval must be 'quantile' or 'cqr'")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           quantile_n_estimators=quantile_n_estimators, interval=interval,
                           q_lo=q_lo, q_hi=q_hi, cal_frac=cal_frac,
                           record_weight=record_weight, seed=seed, n_jobs=n_jobs,
                           backend=backend, device=device, max_depth=max_depth)
        self.data_included = True    # False after loading a model saved without its training data
        self._F = np.empty((0, F.N_FEATURES))
        self._Y = np.empty((0, 4))
        self._W = np.empty(0)
        self._cal = np.empty(0, dtype=bool)
        self._rng = np.random.default_rng(seed)
        self.models: dict[str, lgb.Booster] = {}
        self.cqr_offset = 0.0
        self.n_base = 0
        self.fit_seconds = 0.0

    # ------------------------------------------------------------ data
    @property
    def n_train(self) -> int:
        return len(self._Y)

    def _append(self, Fm: np.ndarray, Y: np.ndarray, weight: float) -> None:
        ok = np.isfinite(Y).all(axis=1) & np.isfinite(Fm).all(axis=1)
        Fm, Y = Fm[ok], Y[ok]
        cal = self._rng.random(len(Y)) < self.params["cal_frac"]
        self._F = np.vstack([self._F, Fm])
        self._Y = np.vstack([self._Y, Y])
        self._W = np.concatenate([self._W, np.full(len(Y), weight)])
        self._cal = np.concatenate([self._cal, cal])

    def fit_frame(self, df) -> "SprayModel":
        """Initial training: replace stored data with the rows of a dataset frame."""
        self._F = np.empty((0, F.N_FEATURES))
        self._Y = np.empty((0, 4))
        self._W = np.empty(0)
        self._cal = np.empty(0, dtype=bool)
        self._append(F.featurize_frame(df), df[list(TARGETS)].to_numpy(float), 1.0)
        self.n_base = self.n_train
        self._refit()
        return self

    def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None:
        """Append NEW records to the stored training data and refit (loop contract)."""
        recs = list(records)
        if not recs:
            return
        if not self.data_included:
            raise RuntimeError("this SprayModel was saved without its training data, so it cannot "
                               "be refit; use the residual slow loop (slow_mode='residual')")
        Fm, Y = records_to_arrays(recs)
        self._append(Fm, Y, float(self.params["record_weight"]))
        if self.n_train >= _MIN_FIT_ROWS:
            self._refit()

    # ------------------------------------------------------------ training
    @property
    def _xgb(self) -> bool:
        return self.params.get("backend", "lightgbm") == "xgboost"

    def _refit(self) -> None:
        if self._xgb:
            self._refit_xgb()
            return
        import time
        t0 = time.perf_counter()
        p = self.params
        n_jobs = p["n_jobs"] or os.cpu_count() or 1
        Z = _to_internal(self._Y, self._F)
        base = dict(learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                    min_child_samples=min(p["min_child_samples"], max(2, self.n_train // 10)),
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    num_threads=n_jobs, verbose=-1, seed=p["seed"])
        use_cal = p["interval"] == "cqr" and (self._cal.sum() >= 50) and (~self._cal).sum() >= 50
        fit_mask = ~self._cal if use_cal else np.ones(self.n_train, bool)

        def train(y, w, mask, n_est, **extra):
            ds = lgb.Dataset(self._F[mask], label=y[mask], weight=w[mask], free_raw_data=True)
            return lgb.train({**base, **extra}, ds, num_boost_round=n_est)

        models: dict[str, lgb.Booster] = {}
        for j, name in enumerate(TARGETS):
            models[name] = train(Z[:, j], self._W, np.ones(self.n_train, bool),
                                 _per_booster(p["n_estimators"], name), objective="regression")
        models["cov_lo"] = train(Z[:, 0], self._W, fit_mask, _per_booster(p["quantile_n_estimators"], "cov_lo"),
                                 objective="quantile", alpha=p["q_lo"])
        models["cov_hi"] = train(Z[:, 0], self._W, fit_mask, _per_booster(p["quantile_n_estimators"], "cov_hi"),
                                 objective="quantile", alpha=p["q_hi"])
        self.models = models
        self.cqr_offset = 0.0
        if use_cal:
            Fc, yc = self._F[self._cal], Z[self._cal, 0]
            lo = models["cov_lo"].predict(Fc, num_threads=n_jobs)
            hi = models["cov_hi"].predict(Fc, num_threads=n_jobs)
            self.cqr_offset = self._cqr_offset(lo, hi, yc)
        self.fit_seconds = time.perf_counter() - t0

    def _cqr_offset(self, lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
        p = self.params
        score = np.maximum(lo - y, y - hi)
        level = min(1.0, (p["q_hi"] - p["q_lo"]) * (1 + 1.0 / len(score)))
        return float(np.quantile(score, level))

    def _refit_xgb(self) -> None:
        """Same boosters, interval and CQR offset as the LightGBM path, trained with XGBoost."""
        import time

        import xgboost as xgb
        t0 = time.perf_counter()
        p = self.params
        n_jobs = p["n_jobs"] or os.cpu_count() or 1
        Z = _to_internal(self._Y, self._F)
        use_cal = p["interval"] == "cqr" and (self._cal.sum() >= 50) and (~self._cal).sum() >= 50
        fit_mask = ~self._cal if use_cal else np.ones(self.n_train, bool)

        def train(d, name, n_est, **extra):
            return xgb.train(xgb_train_params(p, n_jobs, self.n_train, name, **extra), d,
                             num_boost_round=n_est)

        models = {}
        d_all = xgb.QuantileDMatrix(self._F, label=Z[:, 0], weight=self._W)
        for j, name in enumerate(TARGETS):
            d_all.set_label(Z[:, j])
            models[name] = train(d_all, name, _per_booster(p["n_estimators"], name),
                                 objective="reg:squarederror")
        del d_all
        d_fit = xgb.QuantileDMatrix(self._F[fit_mask], label=Z[fit_mask, 0], weight=self._W[fit_mask])
        for name, q in (("cov_lo", p["q_lo"]), ("cov_hi", p["q_hi"])):
            models[name] = train(d_fit, name, _per_booster(p["quantile_n_estimators"], name),
                                 objective="reg:quantileerror", quantile_alpha=q)
        del d_fit
        self.models = {k: _xgb_cpu(b, n_jobs) for k, b in models.items()}
        self.cqr_offset = 0.0
        if use_cal:
            Fc, yc = self._F[self._cal], Z[self._cal, 0]
            self.cqr_offset = self._cqr_offset(self._predict("cov_lo", Fc), self._predict("cov_hi", Fc), yc)
        for b in self.models.values():
            _xgb_cpu(b, 1)
        self.fit_seconds = time.perf_counter() - t0

    def _predict(self, name: str, Fm: np.ndarray) -> np.ndarray:
        b = self.models[name]
        if self._xgb:
            return np.asarray(b.inplace_predict(Fm), dtype=float)
        return b.predict(Fm, num_threads=1)

    # ------------------------------------------------------------ inference
    def _check(self) -> None:
        if not self.models:
            raise RuntimeError("SprayModel is not fitted; call fit_frame(df) or load()")

    def predict_features(self, Fm: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays for a feature matrix: coverage_pct, coverage_lo, coverage_hi, deposition_ul_cm2,
        drift_frac, displacement_mm."""
        self._check()
        Z = np.column_stack([self._predict(t, Fm) for t in TARGETS])
        P = _from_internal(Z, Fm)
        lo = self._predict("cov_lo", Fm) - self.cqr_offset
        hi = self._predict("cov_hi", Fm) + self.cqr_offset
        cov = P[:, 0]
        lo = np.clip(np.minimum(lo, cov), 0.0, 100.0)
        hi = np.clip(np.maximum(hi, cov), 0.0, 100.0)
        return {"coverage_pct": cov, "coverage_lo": lo, "coverage_hi": hi,
                "deposition_ul_cm2": P[:, 1], "drift_frac": P[:, 2], "displacement_mm": P[:, 3]}

    def predict_batch(self, xs: Sequence[Conditions], us: Sequence[Setting]) -> list[Prediction]:
        if len(xs) == 0:
            return []
        a = self.predict_features(F.featurize(xs, us))
        cols = [a[k].tolist() for k in ("coverage_pct", "coverage_lo", "coverage_hi",
                                        "deposition_ul_cm2", "drift_frac", "displacement_mm")]
        return [Prediction(*row) for row in zip(*cols)]

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        return self.predict_batch([x], [u])[0]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path = DEFAULT_PATH, include_data: bool = True) -> Path:
        """``include_data=False`` drops the stored training rows (large for big datasets);
        such a model predicts normally but ``fit(records)`` raises."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = include_data and self.data_included
        data = (dict(F=self._F, Y=self._Y, W=self._W, cal=self._cal) if keep else
                dict(F=np.empty((0, F.N_FEATURES)), Y=np.empty((0, 4)), W=np.empty(0),
                     cal=np.empty(0, dtype=bool)))
        if self._xgb:
            models = {k: bytes(v.save_raw("ubj")) for k, v in self.models.items()}
        else:
            models = {k: v.model_to_string() for k, v in self.models.items()}
        state = dict(params=self.params, **data, models=models,
                     cqr_offset=self.cqr_offset, n_base=self.n_base,
                     feature_version=F.FEATURE_VERSION, feature_names=F.FEATURE_NAMES)
        if not keep:   # key absent in v1 files (= data included)
            state["data_included"] = False
        joblib.dump(state, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "SprayModel":
        s = joblib.load(path)
        if s["feature_version"] != F.FEATURE_VERSION or list(s["feature_names"]) != list(F.FEATURE_NAMES):
            raise ValueError("saved model was built with a different feature set; retrain")
        m = cls(**s["params"])
        m._F, m._Y, m._W, m._cal = s["F"], s["Y"], s["W"], s["cal"]
        if m._xgb:
            import xgboost as xgb

            def _load(raw):
                b = xgb.Booster()
                b.load_model(bytearray(raw))
                return _xgb_cpu(b, 1)
            m.models = {k: _load(v) for k, v in s["models"].items()}
        else:
            m.models = {k: lgb.Booster(model_str=v) for k, v in s["models"].items()}
        m.cqr_offset, m.n_base = s["cqr_offset"], s["n_base"]
        m.data_included = bool(s.get("data_included", True))
        return m


# ---------------------------------------------------------------------- baselines

class _FrameBaseline:
    """Point-prediction baselines (coverage, deposition retention, drift, displacement)."""

    def _make(self):
        raise NotImplementedError

    def fit_arrays(self, Fm: np.ndarray, Y: np.ndarray) -> "_FrameBaseline":
        self._est = self._make()
        self._est.fit(Fm, _to_internal(Y, Fm))
        return self

    def predict_arrays(self, Fm: np.ndarray) -> np.ndarray:
        return _from_internal(np.asarray(self._est.predict(Fm)), Fm)


class RidgeBaseline(_FrameBaseline):
    """Linear baseline: ridge on standardized features (the proposal's linear model)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _make(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))


class ForestBaseline(_FrameBaseline):
    def __init__(self, n_estimators: int = 150, min_samples_leaf: int = 3, seed: int = 0):
        self.n_estimators, self.min_samples_leaf, self.seed = n_estimators, min_samples_leaf, seed

    def _make(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=self.n_estimators,
                                     min_samples_leaf=self.min_samples_leaf, max_features=0.5,
                                     n_jobs=-1, random_state=self.seed)
