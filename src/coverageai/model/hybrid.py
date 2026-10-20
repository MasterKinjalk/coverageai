"""Physics-first hybrid predictor: nominal physics + learned residual + conformal interval.

    prediction = physics_base(x, u) + residual_model(features(x, u), physics_base(x, u))

Components
  * ``PhysicsBase``: the simulator's equations (catalog -> droplet spectrum -> fall with
    evaporation -> drift -> deposition -> Boolean-model coverage) with NO hidden mismatch and
    the WSP spread factor at its mean (the same computation as ``integration.NominalPhysicsModel``
    and ``app_logic.NominalSimModel``; duplicated here so the model package does not depend on
    either). Its own interval is the coverage at the 10 % / 90 % quantiles of the spread factor.
    It can be deliberately MISSPECIFIED (``misspec``): evaporation coefficient, drift threshold,
    spread-factor range and VMD scaled by fixed factors, to mimic the error a real physics model
    has against a real plant (see ``MISSPEC_PRESETS``).
  * ``PhysicsOnlyModel``: Model protocol around ``PhysicsBase`` alone (no learning).
  * ``PhysicsResidualModel``: Model protocol. Learns additive residuals
        coverage      y - C_phys                       (pp)
        deposition    y / (GPA k) - D_phys / (GPA k)   (retention, as in SprayModel)
        drift         y - drift_phys
        displacement  y - disp_phys
    with small, strongly regularized LightGBM boosters on the SprayModel features plus the
    physics outputs (C_phys, its spread-factor band, retention, drift, displacement). The
    coverage interval is conformalized quantile regression (CQR) of the coverage residual,
    with a separate one-sided conformal offset for each bound computed on a held-out
    calibration fraction, so each bound misses about ``q_lo`` (resp. ``1 - q_hi``) of
    outcomes on exchangeable data. The selector uses only the lower bound.

Slow loop. ``fit(records)`` follows the loop contract of ``SprayModel.fit``: it appends the
NEW records (weight ``record_weight``) to the stored data and refits the residual boosters
and the conformal offsets. The closed-loop experiments do NOT use it: there,
``loop.run_passes_slow(slow_mode="residual")`` wraps whatever model is in use in
``feedback.ResidualCorrectedModel`` (frozen base + shrunk per-regime offset), which is the
recommended slow loop for this model too.

Circularity warning. In this project the plant IS the simulator, so with ``misspec=None``
the physics base is the plant's own equations minus the hidden mismatch factors. Any
accuracy it shows against simulator data is an upper bound, not evidence about real fields.
The misspecified variants are the honest check: they show how much of a realistic physics
error the residual learner can recover from data.
"""
from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, Derived, Outcome, Prediction, Setting
from coverageai.model import features as F

TARGETS = ("coverage_pct", "deposition_ul_cm2", "drift_frac", "displacement_mm")
DEFAULT_PATH = ROOT / "models" / "hybrid_v1.joblib"
MISSPEC_PATH = ROOT / "models" / "hybrid_misspec_v1.joblib"
_GPA_UL = float(spec()["physics"]["gpa_to_ul_cm2"])
_GPA_COL = F.feature_index("gpa")
_MIN_FIT_ROWS = 200
_VMD_REF_UM = 300.0
SCHEMA = "hybrid/1"

# Physics-base outputs, in this column order, as a (n, 6) matrix.
PHYS_KEYS = ("coverage_pct", "coverage_lo", "coverage_hi", "deposition_ul_cm2", "drift_frac",
             "displacement_mm")
PHYS_FEATURE_NAMES = ("p_cov", "p_cov_lo", "p_cov_hi", "p_band", "p_logit_cov", "p_retention",
                      "p_drift", "p_disp")

# Deliberate physics errors (multiplicative factors on the true simulator parameters):
# evaporation coefficient +-25-30 %, drift threshold +-25 %, WSP spread factor +-10 %,
# VMD (catalog droplet size) -15 / +20 %. On dataset v1 (no mismatch):
#   optimistic  : physics PREDICTS TOO MUCH coverage (stains spread more, drops smaller) while
#                 overstating evaporation/drift losses. Coverage bias about +4 pp, drift MAE
#                 about 0.07, ~41 % of outcomes below the physics lower bound. The dangerous
#                 direction for the selector (it under-sprays). Used for "hybrid_misspec".
#   pessimistic : the opposite signs (coverage bias about -5 pp; conservative physics).
MISSPEC_PRESETS: dict[str, dict[str, float]] = {
    "optimistic": {"evaporation_lambda_scale": 1.30, "drift_threshold_scale": 0.75,
                   "spread_factor_scale": 1.10, "vmd_scale": 0.85, "vmd_power": 1.0},
    "pessimistic": {"evaporation_lambda_scale": 0.75, "drift_threshold_scale": 1.25,
                    "spread_factor_scale": 0.90, "vmd_scale": 1.20, "vmd_power": 1.0},
}
DEFAULT_MISSPEC = "optimistic"
_MISSPEC_KEYS = ("evaporation_lambda_scale", "drift_threshold_scale", "spread_factor_scale",
                 "vmd_scale", "vmd_power")


def resolve_misspec(misspec: str | Mapping[str, float] | None) -> dict[str, float]:
    """None / "none" -> {} (exact nominal physics); a preset name or a dict of factors."""
    if misspec is None or misspec == "none":
        return {}
    if isinstance(misspec, str):
        if misspec not in MISSPEC_PRESETS:
            raise ValueError(f"unknown misspec preset {misspec!r}; one of {tuple(MISSPEC_PRESETS)}")
        return dict(MISSPEC_PRESETS[misspec])
    bad = set(misspec) - set(_MISSPEC_KEYS)
    if bad:
        raise ValueError(f"unknown misspec keys {sorted(bad)}")
    return {k: float(v) for k, v in misspec.items()}


def _default_derive() -> Callable[[Conditions, Setting], Derived]:
    from coverageai import catalog
    return catalog.derive


# ====================================================================== physics base

class PhysicsBase:
    """Nominal-physics outputs (no hidden mismatch, mean spread factor), optionally misspecified.

    ``arrays(xs, us)`` -> dict of arrays PHYS_KEYS. Deterministic and vectorized.
    """

    def __init__(self, misspec: str | Mapping[str, float] | None = None,
                 derive_fn: Callable[[Conditions, Setting], Derived] | None = None,
                 interval_q: float = 0.1):
        from coverageai.simulator import SpraySimulator
        self.misspec = resolve_misspec(misspec)
        self.derive_fn = derive_fn or _default_derive()
        self.interval_q = float(interval_q)
        p = spec()["physics"]
        m = self.misspec
        over: dict = {}
        if "evaporation_lambda_scale" in m:
            over["evaporation_lambda_coeff"] = float(p["evaporation_lambda_coeff"]) * m["evaporation_lambda_scale"]
        if "drift_threshold_scale" in m:
            over["drift_threshold_m"] = float(p["drift_threshold_m"]) * m["drift_threshold_scale"]
        if "spread_factor_scale" in m:
            over["spread_factor"] = [float(v) * m["spread_factor_scale"] for v in p["spread_factor"]]
        self.overrides = over
        self.vmd_scale = float(m.get("vmd_scale", 1.0))
        self.vmd_power = float(m.get("vmd_power", 1.0))
        self.sim = SpraySimulator(derive_fn=self.derive_fn, physics_overrides=over or None)

    @property
    def exact(self) -> bool:
        return not self.overrides and self.vmd_scale == 1.0 and self.vmd_power == 1.0

    def arrays(self, xs: Sequence[Conditions], us: Sequence[Setting],
               derived: Sequence[Derived] | None = None) -> dict[str, np.ndarray]:
        """Same computation as ``SpraySimulator.nominal_arrays`` (no mismatch, mean spread
        factor, coverage_lo / coverage_hi at the interval_q / 1 - interval_q spread-factor
        quantiles); a VMD misspecification enters as the simulator's per-row VMD factor."""
        from coverageai.physics import coverage as cov_mod
        n = len(xs)
        if n == 0:
            return {k: np.empty(0) for k in PHYS_KEYS}
        ders = derived if derived is not None else [self.derive_fn(x, u) for x, u in zip(xs, us)]
        one = np.ones(n)
        vf = one
        if not (self.vmd_scale == 1.0 and self.vmd_power == 1.0):
            v = np.array([d.vmd_um for d in ders], float)
            vf = self.vmd_scale * (v / _VMD_REF_UM) ** (self.vmd_power - 1.0)
        sim = self.sim
        a, b = (float(t) for t in sim.p["spread_factor"])
        r = sim._arrays_from_hidden(list(xs), list(us), ders, one, one, one, vf,
                                    one * 0.5 * (a + b), want_bins=True)
        d_f, n_cm2 = r.pop("d_f"), r.pop("n_cm2")
        md = sim.p["min_detectable_droplet_um"]
        for key, q in (("coverage_lo", self.interval_q), ("coverage_hi", 1.0 - self.interval_q)):
            c, _ = cov_mod.boolean_coverage(n_cm2, d_f, one * (a + q * (b - a)), md)
            r[key] = c * 100.0
        return {k: np.asarray(r[k], float) for k in PHYS_KEYS}

    def matrix(self, xs, us, derived=None) -> np.ndarray:
        a = self.arrays(xs, us, derived)
        return np.column_stack([a[k] for k in PHYS_KEYS]) if len(xs) else np.empty((0, 6))

    def matrix_frame(self, df) -> np.ndarray:
        """Physics outputs for a dataset frame; uses its Derived columns when present."""
        xs = [Conditions(*r) for r in df[list(F.X_FIELDS)].itertuples(index=False)]
        us = [Setting(r.family, r.size, float(r.pressure_psi)) for r in
              df[["family", "size", "pressure_psi"]].itertuples(index=False)]
        dcols = [f.name for f in dataclasses.fields(Derived)]
        ders = ([Derived(*r) for r in df[dcols].itertuples(index=False)]
                if all(c in df.columns for c in dcols) else None)
        return self.matrix(xs, us, ders)


def _phys_features(P: np.ndarray, Fm: np.ndarray) -> np.ndarray:
    cov, lo, hi = P[:, 0], P[:, 1], P[:, 2]
    c = np.clip(cov / 100.0, 1e-4, 1 - 1e-4)
    ret = P[:, 3] / (Fm[:, _GPA_COL] * _GPA_UL)
    return np.column_stack([cov, lo, hi, hi - lo, np.log(c / (1 - c)), ret, P[:, 4], P[:, 5]])


def _prediction_rows(a: dict[str, np.ndarray]) -> list[Prediction]:
    cols = [a[k].tolist() for k in ("coverage_pct", "coverage_lo", "coverage_hi",
                                    "deposition_ul_cm2", "drift_frac", "displacement_mm")]
    return [Prediction(*row) for row in zip(*cols)]


class PhysicsOnlyModel:
    """Model protocol: physics base alone. Interval = spread-factor q / 1-q band. ``fit`` no-op.

    With ``misspec=None`` this equals ``integration.NominalPhysicsModel`` (option
    ``oracle_nominal``)."""

    def __init__(self, misspec: str | Mapping[str, float] | None = None,
                 derive_fn: Callable | None = None, interval_q: float = 0.1):
        self.base = PhysicsBase(misspec, derive_fn, interval_q)

    def predict_arrays(self, xs, us) -> dict[str, np.ndarray]:
        return self.base.arrays(xs, us)

    def predict_batch(self, xs: Sequence[Conditions], us: Sequence[Setting]) -> list[Prediction]:
        if len(xs) == 0:
            return []
        return _prediction_rows(self.base.arrays(xs, us))

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        return self.predict_batch([x], [u])[0]

    def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None:
        pass


# ====================================================================== hybrid

class PhysicsResidualModel:
    """Physics base + regularized LightGBM residuals + CQR interval (Model protocol)."""

    def __init__(self, misspec: str | Mapping[str, float] | None = None,
                 n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 15,
                 min_child_samples: int = 200, reg_lambda: float = 10.0,
                 quantile_n_estimators: int = 200, q_lo: float = 0.1, q_hi: float = 0.9,
                 cal_frac: float = 0.15, record_weight: float = 1.0, seed: int = 0,
                 n_jobs: int | None = None, derive_fn: Callable | None = None):
        self.params = dict(misspec=resolve_misspec(misspec), n_estimators=n_estimators,
                           learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, reg_lambda=reg_lambda,
                           quantile_n_estimators=quantile_n_estimators, q_lo=q_lo, q_hi=q_hi,
                           cal_frac=cal_frac, record_weight=record_weight, seed=seed,
                           n_jobs=n_jobs)
        self.base = PhysicsBase(self.params["misspec"], derive_fn, interval_q=q_lo)
        self._F = np.empty((0, F.N_FEATURES))
        self._P = np.empty((0, 6))
        self._Y = np.empty((0, 4))
        self._W = np.empty(0)
        self._cal = np.empty(0, dtype=bool)
        self._rng = np.random.default_rng(seed)
        self.models: dict[str, lgb.Booster] = {}
        self.offset_lo = 0.0
        self.offset_hi = 0.0
        self.n_base = 0
        self.fit_seconds = 0.0

    # ------------------------------------------------------------ data
    @property
    def n_train(self) -> int:
        return len(self._Y)

    @property
    def misspec(self) -> dict[str, float]:
        return dict(self.params["misspec"])

    def _append(self, Fm: np.ndarray, P: np.ndarray, Y: np.ndarray, weight: float) -> None:
        ok = np.isfinite(Y).all(axis=1) & np.isfinite(Fm).all(axis=1) & np.isfinite(P).all(axis=1)
        Fm, P, Y = Fm[ok], P[ok], Y[ok]
        cal = self._rng.random(len(Y)) < self.params["cal_frac"]
        self._F = np.vstack([self._F, Fm])
        self._P = np.vstack([self._P, P])
        self._Y = np.vstack([self._Y, Y])
        self._W = np.concatenate([self._W, np.full(len(Y), float(weight))])
        self._cal = np.concatenate([self._cal, cal])

    def _reset_data(self) -> None:
        self._F = np.empty((0, F.N_FEATURES))
        self._P = np.empty((0, 6))
        self._Y = np.empty((0, 4))
        self._W = np.empty(0)
        self._cal = np.empty(0, dtype=bool)
        self._rng = np.random.default_rng(self.params["seed"])

    def fit_arrays(self, Fm: np.ndarray, P: np.ndarray, Y: np.ndarray) -> "PhysicsResidualModel":
        """Initial training from precomputed features, physics-base matrix and targets."""
        self._reset_data()
        self._append(np.asarray(Fm, float), np.asarray(P, float), np.asarray(Y, float), 1.0)
        self.n_base = self.n_train
        self._refit()
        return self

    def fit_frame(self, df, P: np.ndarray | None = None) -> "PhysicsResidualModel":
        """Initial training on a dataset frame (replaces stored data). ``P`` = precomputed
        ``self.base.matrix_frame(df)`` (optional, to share it between models)."""
        P = self.base.matrix_frame(df) if P is None else P
        return self.fit_arrays(F.featurize_frame(df), P, df[list(TARGETS)].to_numpy(float))

    def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None:
        """Loop contract: append NEW records (weight ``record_weight``) and refit residuals.
        (The closed-loop slow loop wraps the model in feedback.ResidualCorrectedModel instead.)"""
        recs = list(records)
        if not recs:
            return
        xs, us = [r[0] for r in recs], [r[1] for r in recs]
        Y = np.array([[r[2].coverage_pct, r[2].deposition_ul_cm2, r[2].drift_frac,
                       r[2].displacement_mm] for r in recs], float).reshape(len(recs), 4)
        self._append(F.featurize(xs, us), self.base.matrix(xs, us), Y,
                     float(self.params["record_weight"]))
        if self.n_train >= _MIN_FIT_ROWS:
            self._refit()

    # ------------------------------------------------------------ training
    def _design(self, Fm: np.ndarray, P: np.ndarray) -> np.ndarray:
        return np.hstack([Fm, _phys_features(P, Fm)])

    def _residual_targets(self) -> np.ndarray:
        Y, P, Fm = self._Y, self._P, self._F
        k = Fm[:, _GPA_COL] * _GPA_UL
        return np.column_stack([Y[:, 0] - P[:, 0], (Y[:, 1] - P[:, 3]) / k,
                                Y[:, 2] - P[:, 4], Y[:, 3] - P[:, 5]])

    def _refit(self) -> None:
        t0 = time.perf_counter()
        p = self.params
        n_jobs = p["n_jobs"] or min(4, os.cpu_count() or 1)
        X = self._design(self._F, self._P)
        R = self._residual_targets()
        base = dict(learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                    min_child_samples=min(p["min_child_samples"], max(5, self.n_train // 20)),
                    reg_lambda=p["reg_lambda"], subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, num_threads=n_jobs, verbose=-1, seed=p["seed"])
        use_cal = (self._cal.sum() >= 50) and ((~self._cal).sum() >= 50)
        fit_mask = ~self._cal if use_cal else np.ones(self.n_train, bool)

        def train(y, mask, n_est, **extra):
            ds = lgb.Dataset(X[mask], label=y[mask], weight=self._W[mask], free_raw_data=True)
            return lgb.train({**base, **extra}, ds, num_boost_round=n_est)

        allm = np.ones(self.n_train, bool)
        models = {name: train(R[:, j], allm, p["n_estimators"], objective="regression")
                  for j, name in enumerate(TARGETS)}
        models["cov_lo"] = train(R[:, 0], fit_mask, p["quantile_n_estimators"],
                                 objective="quantile", alpha=p["q_lo"])
        models["cov_hi"] = train(R[:, 0], fit_mask, p["quantile_n_estimators"],
                                 objective="quantile", alpha=p["q_hi"])
        self.models = models
        self.offset_lo = self.offset_hi = 0.0
        if use_cal:
            Xc, rc = X[self._cal], R[self._cal, 0]
            lo = models["cov_lo"].predict(Xc, num_threads=n_jobs)
            hi = models["cov_hi"].predict(Xc, num_threads=n_jobs)
            n = len(rc)
            # one-sided split-conformal offset per bound (finite-sample corrected level)
            self.offset_lo = float(np.quantile(lo - rc, min(1.0, (1 - p["q_lo"]) * (1 + 1.0 / n))))
            self.offset_hi = float(np.quantile(rc - hi, min(1.0, p["q_hi"] * (1 + 1.0 / n))))
        self.fit_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------ inference
    def _check(self) -> None:
        if not self.models:
            raise RuntimeError("PhysicsResidualModel is not fitted; call fit_frame(df) or load()")

    def predict_from(self, Fm: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays coverage_pct, coverage_lo, coverage_hi, deposition_ul_cm2, drift_frac,
        displacement_mm (plus physics_coverage_pct) from features and the physics matrix."""
        self._check()
        m = self.models
        X = self._design(Fm, P)
        k = Fm[:, _GPA_COL] * _GPA_UL
        r = {t: m[t].predict(X, num_threads=1) for t in TARGETS}
        cov = np.clip(P[:, 0] + r["coverage_pct"], 0.0, 100.0)
        lo = P[:, 0] + m["cov_lo"].predict(X, num_threads=1) - self.offset_lo
        hi = P[:, 0] + m["cov_hi"].predict(X, num_threads=1) + self.offset_hi
        lo = np.clip(np.minimum(lo, cov), 0.0, 100.0)
        hi = np.clip(np.maximum(hi, cov), 0.0, 100.0)
        ret = np.clip(P[:, 3] / k + r["deposition_ul_cm2"], 0.0, 1.5)
        return {"coverage_pct": cov, "coverage_lo": lo, "coverage_hi": hi,
                "deposition_ul_cm2": ret * k,
                "drift_frac": np.clip(P[:, 4] + r["drift_frac"], 0.0, 1.0),
                "displacement_mm": np.maximum(P[:, 5] + r["displacement_mm"], 0.0),
                "physics_coverage_pct": P[:, 0]}

    def predict_arrays(self, xs: Sequence[Conditions], us: Sequence[Setting]) -> dict[str, np.ndarray]:
        return self.predict_from(F.featurize(xs, us), self.base.matrix(xs, us))

    def predict_batch(self, xs: Sequence[Conditions], us: Sequence[Setting]) -> list[Prediction]:
        if len(xs) == 0:
            return []
        return _prediction_rows(self.predict_arrays(xs, us))

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        return self.predict_batch([x], [u])[0]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(schema=SCHEMA, params=self.params, F=self._F, P=self._P, Y=self._Y,
                     W=self._W, cal=self._cal,
                     models={k: v.model_to_string() for k, v in self.models.items()},
                     offset_lo=self.offset_lo, offset_hi=self.offset_hi, n_base=self.n_base,
                     feature_version=F.FEATURE_VERSION, feature_names=list(F.FEATURE_NAMES))
        joblib.dump(state, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH, derive_fn: Callable | None = None
             ) -> "PhysicsResidualModel":
        s = joblib.load(path)
        if s.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} artifact")
        if s["feature_version"] != F.FEATURE_VERSION or list(s["feature_names"]) != list(F.FEATURE_NAMES):
            raise ValueError("saved hybrid model was built with a different feature set; retrain")
        m = cls(**s["params"], derive_fn=derive_fn)
        m._F, m._P, m._Y, m._W, m._cal = s["F"], s["P"], s["Y"], s["W"], s["cal"]
        m.models = {k: lgb.Booster(model_str=v) for k, v in s["models"].items()}
        m.offset_lo, m.offset_hi, m.n_base = s["offset_lo"], s["offset_hi"], s["n_base"]
        return m


def find_hybrid_path(misspec: bool = False, path: str | Path | None = None) -> Path | None:
    """Explicit path, else models/hybrid_v1.joblib (or hybrid_misspec_v1.joblib) in ROOT or a
    parent directory (worktrees live below the main checkout)."""
    if path:
        p = Path(path)
        return p if p.is_file() else None
    name = (MISSPEC_PATH if misspec else DEFAULT_PATH).name
    return next((d / "models" / name for d in [ROOT, *ROOT.parents]
                 if (d / "models" / name).is_file()), None)


def load_hybrid_model(path: str | Path | None = None, misspec: bool = False,
                      derive_fn: Callable | None = None, **params) -> PhysicsResidualModel:
    """Load a trained hybrid (``experiments/eval_hybrid.py --train`` writes the artifacts);
    keyword params update model.params (e.g. record_weight, n_jobs)."""
    p = find_hybrid_path(misspec, path)
    if p is None:
        raise FileNotFoundError("trained hybrid model not found (run experiments/eval_hybrid.py "
                                f"--stage train); looked for {path or (MISSPEC_PATH if misspec else DEFAULT_PATH).name}")
    m = PhysicsResidualModel.load(p, derive_fn=derive_fn)
    m.params.update({k: v for k, v in params.items() if v is not None})
    return m
