"""Learned card feedback (opt-in, decision D29): a model that predicts the next card's error.

The fast loop of the proposal (eq. 3, ``feedback.BiasFeedback``) corrects the coverage model
with a running average of card errors, kept per hand-set regime (crosswind range x droplet
group). The coverage model itself never sees a card. This module replaces that running average
with a trained estimate: a gradient-boosted model (XGBoost or LightGBM) that predicts

    e_k = card reading_k - model prediction_k

for the section about to be sprayed, from

* the section: wind, temperature, humidity, speed, boom height and the chosen setting's flow,
  spray volume and droplet size, and
* the cards read so far in this pass: how many, the last errors, their running averages, the
  same on a deposit-ratio scale, and what the eq. 3 rule would apply (so the model can fall
  back on the rule and only has to learn where the rule is wrong).

It can learn what the rule cannot express: that the hidden losses scale with the coverage level
instead of adding a constant, that the last error says more than older ones when gusts persist,
and what a typical error is before any card has been read.

``LearnedFeedback`` has the interface ``loop.run_pass`` expects of its ``feedback`` argument, so
it plugs in with ``run_pass(..., feedback=LearnedFeedback(model))`` and nothing changes when it
is not passed. Training uses simulated passes on development seeds only
(``experiments/train_feedback_model.py``). Everything here is simulation: the model learns the
simulator's hidden mismatch, not a real field.

One code path builds the features for training (``rows_from_logs``) and at run time
(``LearnedFeedback``): both replay the cards through ``HistoryState``. A feature row for section
k uses the cards of sections before k only.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import CardMeasurement, Conditions, Derived
from coverageai.feedback import BiasFeedback, retreat_flag
from coverageai.physics.droplets import terminal_velocity

FEATURE_VERSION = 1
SECTION_FEATURES = ("wind_speed", "abs_u_cross", "u_along", "temp_c", "rh_pct", "speed_kmh", "boom_height",
                    "gpa", "vmd_um", "class_ord", "flow_gpm", "gpa_over_vmd", "drift_index")
HISTORY_FEATURES = ("n_cards", "e_last", "e_last2", "e_last3", "e_mean", "e_std", "e_ewma_slow", "e_ewma_fast",
                    "r_last", "r_mean", "r_ewma", "pred_mean", "card_mean",
                    "group_n", "group_ewma", "rule_bias",
                    "last_abs_u_cross", "last_vmd_um", "last_gpa_over_vmd", "d_wind", "d_abs_u_cross")
FEATURE_NAMES = SECTION_FEATURES + HISTORY_FEATURES
N_FEATURES = len(FEATURE_NAMES)
MODEL_DIR = ROOT / "models"
# Trained files by version. v1 learned from raw card readings (D29). v2 learned from readings with
# the reader's known error removed (D30) and is the one the app uses (D31). Same features in both.
MODEL_PATHS = {v: {"xgboost": MODEL_DIR / f"feedback_xgb_v{v}.joblib",
                   "lightgbm": MODEL_DIR / f"feedback_lgbm_v{v}.joblib"} for v in (1, 2)}
DEFAULT_VERSION = 2
DEFAULT_PATHS = MODEL_PATHS[DEFAULT_VERSION]
_CLASS_IDX = {c: i for i, c in enumerate(spec()["droplets"]["class_order"])}
_EWMA_FAST = 0.6          # weight of the newest card in the fast running average (the rule uses 0.3)
_NAN = float("nan")


def deposit_ratio(card_pct: float, model_pct: float) -> float:
    """Card against model on a deposit scale. For droplets landing at random the covered share is
    1 - exp(-deposit x const), so ln(1 - C) is proportional to the deposit and the ratio of the
    two logs estimates (deposit that arrived) / (deposit the model expected). A hidden loss such
    as canopy interception shows up as a ratio below 1 at every coverage level, which an additive
    error does not."""
    c = min(max(card_pct, 0.5), 99.5) / 100.0
    m = min(max(model_pct, 0.5), 99.5) / 100.0
    return float(np.clip(math.log1p(-c) / math.log1p(-m), 0.2, 3.0))


def section_features(x: Conditions, derived: Derived) -> list[float]:
    """The part of the feature row that describes the section and the chosen setting."""
    vmd = float(derived.vmd_um)
    cross = abs(float(derived.u_cross_ms))
    gpa = float(derived.app_rate_gpa)
    vt = float(terminal_velocity(np.array([vmd]))[0])
    return [float(x.wind_speed_ms), cross, float(derived.u_along_ms), float(x.temp_c), float(x.rh_pct),
            float(x.travel_speed_kmh), float(x.boom_height_m), gpa, vmd,
            float(_CLASS_IDX.get(derived.droplet_class, -1)), float(derived.flow_gpm),
            gpa / max(vmd, 1.0), cross * float(x.boom_height_m) / max(vt, 1e-6)]


class HistoryState:
    """The cards read so far in a pass, and the features they give for the next section."""

    def __init__(self) -> None:
        self.rule = BiasFeedback()          # the eq. 3 rule, run alongside: a feature and a fallback
        self.reset()

    def reset(self) -> None:
        self.rule.reset()
        self.errors: list[float] = []       # card - model, in section order
        self.ratios: list[float] = []
        self.preds: list[float] = []
        self.cards: list[float] = []
        self._slow = self._fast = self._r = 0.0
        self._last_section: tuple[float, float, float, float] | None = None   # wind, |cross|, vmd, gpa/vmd

    def regime(self, x: Conditions, derived: Derived) -> str:
        return self.rule.regime(x, derived)

    def features(self, x: Conditions, derived: Derived) -> np.ndarray:
        """Feature row for a section about to be sprayed. Uses earlier cards only."""
        sec = section_features(x, derived)
        e, n = self.errors, len(self.errors)
        if n:
            hist = [float(n), e[-1], e[-2] if n > 1 else _NAN, e[-3] if n > 2 else _NAN,
                    float(np.mean(e)), float(np.std(e)) if n > 1 else _NAN, self._slow, self._fast,
                    self.ratios[-1], float(np.mean(self.ratios)), self._r,
                    float(np.mean(self.preds)), float(np.mean(self.cards))]
        else:
            hist = [0.0] + [_NAN] * 12
        regime = self.regime(x, derived)
        group = self.rule.state()["regimes"].get(regime)
        hist += [float(group["n"]) if group else 0.0, float(group["bias"]) if group else _NAN,
                 float(self.rule.bias(regime))]
        if self._last_section is not None:
            w, c, v, g = self._last_section
            hist += [c, v, g, sec[0] - w, sec[1] - c]
        else:
            hist += [_NAN] * 5
        return np.array(sec + hist, dtype=float)

    def update(self, x: Conditions, derived: Derived, predicted_cov: float, measured_cov: float) -> float:
        """Add the card of the section just sprayed. ``predicted_cov`` is the RAW model prediction."""
        e = float(measured_cov) - float(predicted_cov)
        r = deposit_ratio(measured_cov, predicted_cov)
        a = float(self.rule.alpha)
        first = not self.errors
        self._slow = e if first else (1 - a) * self._slow + a * e
        self._fast = e if first else (1 - _EWMA_FAST) * self._fast + _EWMA_FAST * e
        self._r = r if first else (1 - a) * self._r + a * r
        self.errors.append(e)
        self.ratios.append(r)
        self.preds.append(float(predicted_cov))
        self.cards.append(float(measured_cov))
        self.rule.update(self.regime(x, derived), measured_cov, predicted_cov)
        sec = section_features(x, derived)
        self._last_section = (sec[0], sec[1], sec[8], sec[11])
        return e


def rows_from_logs(logs: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Training rows from one pass: (features, target) with target = card reading - model
    prediction of each section. The features of section k are built before its card is added."""
    state = HistoryState()
    X, y = [], []
    for lg in logs:
        X.append(state.features(lg.x, lg.derived))
        y.append(state.update(lg.x, lg.derived, lg.pred.coverage_pct, lg.meas.coverage_pct))
    return np.array(X, dtype=float).reshape(len(X), N_FEATURES), np.array(y, dtype=float)


# ------------------------------------------------------------------ the model

class FeedbackModel:
    """Gradient-boosted regressor for the next card's error (pp). Trains on the GPU with XGBoost
    when asked; predicts on the CPU with one thread, one row at a time, inside the loop."""

    def __init__(self, backend: str, booster: Any, meta: dict | None = None):
        if backend not in ("xgboost", "lightgbm"):
            raise ValueError("backend must be 'xgboost' or 'lightgbm'")
        self.backend, self.booster, self.meta = backend, booster, dict(meta or {})

    @classmethod
    def train(cls, X: np.ndarray, y: np.ndarray, backend: str = "xgboost", device: str = "cpu",
              n_estimators: int = 600, max_depth: int = 8, learning_rate: float = 0.05,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              early_stopping_rounds: int = 50, seed: int = 0, **params) -> "FeedbackModel":
        """Fit on (X, y). With a validation set, boosting stops when its error stops improving."""
        meta = dict(n_rows=int(len(y)), n_estimators=int(n_estimators), max_depth=int(max_depth),
                    learning_rate=float(learning_rate), device=device, features=list(FEATURE_NAMES),
                    feature_version=FEATURE_VERSION)
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if backend == "xgboost":
            import xgboost as xgb
            p = dict(objective="reg:squarederror", tree_method="hist", device=device, max_depth=max_depth,
                     eta=learning_rate, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                     reg_lambda=1.0, max_bin=256, seed=seed, verbosity=0)
            p.update(params)
            d = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
            evals = [(xgb.DMatrix(X_val, label=y_val, feature_names=list(FEATURE_NAMES)), "val")] if has_val else []
            bst = xgb.train(p, d, num_boost_round=n_estimators, evals=evals, verbose_eval=False,
                            early_stopping_rounds=early_stopping_rounds if has_val else None)
            if has_val:                     # keep the trees up to the best round only
                bst = bst[: bst.best_iteration + 1]
            bst.set_param({"device": "cpu", "nthread": 1})
            meta["trees"] = int(bst.num_boosted_rounds())
        elif backend == "lightgbm":
            import lightgbm as lgb
            p = dict(objective="regression", learning_rate=learning_rate, num_leaves=2 ** min(max_depth, 6) - 1,
                     max_depth=max_depth, min_child_samples=50, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, reg_lambda=1.0, seed=seed, verbose=-1, num_threads=0)
            p.update(params)
            d = lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=True)
            cb = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if has_val else []
            bst = lgb.train(p, d, num_boost_round=n_estimators, callbacks=cb,
                            valid_sets=[lgb.Dataset(X_val, label=y_val, reference=d)] if has_val else None)
            meta["trees"] = int(bst.current_iteration() if not has_val else (bst.best_iteration or bst.current_iteration()))
        else:
            raise ValueError("backend must be 'xgboost' or 'lightgbm'")
        return cls(backend, bst, meta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, N_FEATURES)
        if self.backend == "xgboost":
            return np.asarray(self.booster.inplace_predict(X), dtype=float)
        n = self.meta.get("trees") or None
        return np.asarray(self.booster.predict(X, num_iteration=n, num_threads=1), dtype=float)

    def importance(self) -> dict[str, float]:
        """Share of the total split gain per feature (sums to 1)."""
        if self.backend == "xgboost":
            g = self.booster.get_score(importance_type="total_gain")
            raw = np.array([g.get(f, 0.0) for f in FEATURE_NAMES], dtype=float)
        else:
            raw = np.asarray(self.booster.feature_importance(importance_type="gain"), dtype=float)
        tot = float(raw.sum()) or 1.0
        return {f: float(v / tot) for f, v in zip(FEATURE_NAMES, raw)}

    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (bytes(self.booster.save_raw("ubj")) if self.backend == "xgboost"
               else self.booster.model_to_string(num_iteration=self.meta.get("trees") or -1))
        joblib.dump({"backend": self.backend, "raw": raw, "meta": self.meta}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeedbackModel":
        import joblib
        blob = joblib.load(Path(path))
        meta = blob.get("meta", {})
        if list(meta.get("features", FEATURE_NAMES)) != list(FEATURE_NAMES):
            raise ValueError(f"{path}: trained on a different feature list; retrain it "
                             "(experiments/train_feedback_model.py)")
        if blob["backend"] == "xgboost":
            import xgboost as xgb
            bst = xgb.Booster()
            bst.load_model(bytearray(blob["raw"]))
            bst.set_param({"device": "cpu", "nthread": 1})
        else:
            import lightgbm as lgb
            bst = lgb.Booster(model_str=blob["raw"])
        return cls(blob["backend"], bst, meta)


def load_feedback_model(name: str) -> FeedbackModel:
    """``"xgboost"`` / ``"lightgbm"`` (the shipped files) or a path."""
    path = DEFAULT_PATHS.get(name, Path(name))
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Train it with: "
                                "uv run --no-sync python experiments/train_feedback_model.py")
    return FeedbackModel.load(path)


# ------------------------------------------------------------------ loop adapter

class LearnedFeedback:
    """Card feedback from a trained ``FeedbackModel``, with the interface ``loop.run_pass`` uses
    (``global_bias``, ``regime``, ``bias``, ``update``, ``retreat_flag``, ``reset``, ``state``).

    The correction for a section is the model's prediction of that section's card error, limited
    to +-``clip_pp`` like the rule's. ``regime`` returns the same key as the rule (it is what the
    logs carry) and remembers the section, so ``bias(key)`` can be asked before the card (the
    correction to apply) and after it (the correction a section like this would get now)."""

    def __init__(self, model: FeedbackModel, clip_pp: float | None = None):
        self.model = model
        self.clip_pp = float(spec()["feedback"]["bias_clip_pp"] if clip_pp is None else clip_pp)
        self.history = HistoryState()
        self.reset()

    def reset(self) -> None:
        self.history.reset()
        self._sections: dict[str, tuple[Conditions, Derived]] = {}
        self._hint = 0.0

    def regime(self, x: Conditions, derived: Derived) -> str:
        key = self.history.regime(x, derived)
        self._sections[key] = (x, derived)
        return key

    def _predict(self, x: Conditions, derived: Derived) -> float:
        b = float(self.model.predict(self.history.features(x, derived)[None, :])[0])
        return float(np.clip(b, -self.clip_pp, self.clip_pp)) if math.isfinite(b) else 0.0

    def bias(self, regime: str | None = None) -> float:
        sec = self._sections.get(regime) if regime is not None else None
        return self._predict(*sec) if sec is not None else self._hint

    def global_bias(self) -> float:
        """First guess for the selector: the correction the last section would get now."""
        return self._hint

    def update(self, regime: str, measured_cov: float, predicted_cov: float) -> float:
        x, derived = self._sections[regime]
        e = self.history.update(x, derived, predicted_cov, measured_cov)
        self._hint = self._predict(x, derived)
        return e

    def retreat_flag(self, meas: CardMeasurement) -> bool:
        return retreat_flag(meas)

    def state(self) -> dict:
        h = self.history
        return {"learned": True, "backend": self.model.backend, "clip_pp": self.clip_pp,
                "n_cards": len(h.errors), "last_error": h.errors[-1] if h.errors else None,
                "mean_error": float(np.mean(h.errors)) if h.errors else None, "rule": h.rule.state()}
