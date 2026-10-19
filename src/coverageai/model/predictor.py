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

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.15,
                 num_leaves: int = 31, min_child_samples: int = 20,
                 quantile_n_estimators: int = 60, interval: str = "cqr",
                 q_lo: float = 0.1, q_hi: float = 0.9, cal_frac: float = 0.15,
                 record_weight: float = 1.0, seed: int = 0, n_jobs: int | None = None):
        if interval not in ("quantile", "cqr"):
            raise ValueError("interval must be 'quantile' or 'cqr'")
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           quantile_n_estimators=quantile_n_estimators, interval=interval,
                           q_lo=q_lo, q_hi=q_hi, cal_frac=cal_frac,
                           record_weight=record_weight, seed=seed, n_jobs=n_jobs)
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
        Fm, Y = records_to_arrays(recs)
        self._append(Fm, Y, float(self.params["record_weight"]))
        if self.n_train >= _MIN_FIT_ROWS:
            self._refit()

    # ------------------------------------------------------------ training
    def _refit(self) -> None:
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
            models[name] = train(Z[:, j], self._W, np.ones(self.n_train, bool), p["n_estimators"],
                                 objective="regression")
        models["cov_lo"] = train(Z[:, 0], self._W, fit_mask, p["quantile_n_estimators"],
                                 objective="quantile", alpha=p["q_lo"])
        models["cov_hi"] = train(Z[:, 0], self._W, fit_mask, p["quantile_n_estimators"],
                                 objective="quantile", alpha=p["q_hi"])
        self.models = models
        self.cqr_offset = 0.0
        if use_cal:
            Fc, yc = self._F[self._cal], Z[self._cal, 0]
            lo = models["cov_lo"].predict(Fc, num_threads=n_jobs)
            hi = models["cov_hi"].predict(Fc, num_threads=n_jobs)
            score = np.maximum(lo - yc, yc - hi)
            n = len(score)
            level = min(1.0, (p["q_hi"] - p["q_lo"]) * (1 + 1.0 / n))
            self.cqr_offset = float(np.quantile(score, level))
        self.fit_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------ inference
    def _check(self) -> None:
        if not self.models:
            raise RuntimeError("SprayModel is not fitted; call fit_frame(df) or load()")

    def predict_features(self, Fm: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays for a feature matrix: coverage_pct, coverage_lo, coverage_hi, deposition_ul_cm2,
        drift_frac, displacement_mm."""
        self._check()
        m = self.models
        Z = np.column_stack([m[t].predict(Fm, num_threads=1) for t in TARGETS])
        P = _from_internal(Z, Fm)
        lo = m["cov_lo"].predict(Fm, num_threads=1) - self.cqr_offset
        hi = m["cov_hi"].predict(Fm, num_threads=1) + self.cqr_offset
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
    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(params=self.params, F=self._F, Y=self._Y, W=self._W, cal=self._cal,
                     models={k: v.model_to_string() for k, v in self.models.items()},
                     cqr_offset=self.cqr_offset, n_base=self.n_base,
                     feature_version=F.FEATURE_VERSION, feature_names=F.FEATURE_NAMES)
        joblib.dump(state, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "SprayModel":
        s = joblib.load(path)
        if s["feature_version"] != F.FEATURE_VERSION or list(s["feature_names"]) != list(F.FEATURE_NAMES):
            raise ValueError("saved model was built with a different feature set; retrain")
        m = cls(**s["params"])
        m._F, m._Y, m._W, m._cal = s["F"], s["Y"], s["W"], s["cal"]
        m.models = {k: lgb.Booster(model_str=v) for k, v in s["models"].items()}
        m.cqr_offset, m.n_base = s["cqr_offset"], s["n_base"]
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
