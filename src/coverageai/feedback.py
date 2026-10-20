"""Fast-loop feedback: running bias correction of the model's coverage prediction.

Proposal eq. (3):  b_{k+1} = (1 - alpha) * b_k + alpha * e_k,
with e_k = C_meas,k - C_pred,k where C_pred is the RAW model prediction (never the
bias-corrected one, otherwise the update would chase its own correction).

Design choices (all configurable via config/spec.yaml `feedback`):
  * clipping: |b| <= clip_pp after every update.
  * warm-up: a bias is only acted on (returned by `bias()`) once at least
    `warmup_cards` cards have been seen for that regime. Updates always accumulate.
  * regimes: default = bin of |u_cross| (spec edges) x droplet group (fine/coarse).
    Each regime has its own bias, plus a global bias updated by every card.
  * global fallback: if a regime has NO data yet, `bias()` returns the global bias
    damped by 0.5 (a regime we know nothing about should get a cautious share of the
    pass-wide correction). A regime with 1..warmup-1 cards returns 0 (still warming).
    With `per_regime=False` a single global bias is used with no damping.
  * retreat flag: a card classified "critical" (coverage < marginal_min) flags the
    segment for re-treatment.

Slow loop (``ResidualCorrectedModel``, proposal section 4.4): between passes every verified
record (x, u, measured C) joins the slow-loop training set and a correction model on top
of the frozen offline model is refit on all records so far. See the class docstring for
why a residual model is used instead of refitting the LightGBM model itself.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from coverageai.config import spec
from coverageai.contracts import CardMeasurement, Conditions, Derived, Outcome, Prediction, Setting

GLOBAL_KEY = "global"
FALLBACK_DAMPING = 0.5
SLOW_PRIOR_SD_PP = 5.0      # prior sd of the systematic (cross-pass) coverage offset


def retreat_flag(meas: CardMeasurement) -> bool:
    """True if the card is in the critical coverage class (re-treatment candidate)."""
    return meas.coverage_class == "critical"


def default_regime_fn() -> Callable[[Conditions, Derived], str]:
    cfg = spec()["feedback"]["regime_bins"]
    edges = list(cfg["u_cross_ms"])
    group_of = {c: g for g, classes in cfg["droplet_group"].items() for c in classes}

    def regime(x: Conditions, derived: Derived) -> str:
        v = abs(derived.u_cross_ms)
        i = int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2))
        return f"u{edges[i]:g}-{edges[i + 1]:g}|{group_of.get(derived.droplet_class, 'unknown')}"

    return regime


@dataclass
class _Acc:
    b: float = 0.0
    n: int = 0


class BiasFeedback:
    def __init__(self, alpha: float | None = None, clip_pp: float | None = None,
                 warmup_cards: int | None = None,
                 regime_fn: Callable[[Conditions, Derived], str] | None = None,
                 per_regime: bool = True):
        fb = spec()["feedback"]
        self.alpha = fb["alpha"] if alpha is None else alpha
        self.clip_pp = fb["bias_clip_pp"] if clip_pp is None else clip_pp
        self.warmup_cards = fb["warmup_cards"] if warmup_cards is None else warmup_cards
        self.per_regime = per_regime
        self._regime_fn = regime_fn or default_regime_fn()
        self.reset()

    # ------------------------------------------------------------------ API
    def regime(self, x: Conditions, derived: Derived) -> str:
        return self._regime_fn(x, derived) if self.per_regime else GLOBAL_KEY

    def global_bias(self) -> float:
        """Pass-wide bias (all regimes pooled), gated by warm-up. Used as a selection hint."""
        return self._glob.b if self._glob.n >= max(self.warmup_cards, 1) else 0.0

    def bias(self, regime: str | None = None) -> float:
        """Correction (pp) to add to the raw model coverage for this regime."""
        if not self.per_regime or regime is None:
            return self.global_bias()
        acc = self._reg.get(regime)
        warm = max(self.warmup_cards, 1)
        if acc is not None and acc.n >= warm:
            return acc.b
        if (acc is None or acc.n == 0) and self._glob.n >= warm:
            return FALLBACK_DAMPING * self._glob.b
        return 0.0

    def update(self, regime: str, measured_cov: float, predicted_cov: float) -> float:
        """Eq. (3) update. `predicted_cov` must be the RAW model prediction. Returns e_k."""
        e = float(measured_cov) - float(predicted_cov)
        self._step(self._glob, e)
        if self.per_regime:
            self._step(self._reg.setdefault(regime, _Acc()), e)
        return e

    def retreat_flag(self, meas: CardMeasurement) -> bool:
        return retreat_flag(meas)

    def reset(self) -> None:
        self._glob = _Acc()
        self._reg: dict[str, _Acc] = {}

    def state(self) -> dict:
        return {
            "alpha": self.alpha, "clip_pp": self.clip_pp, "warmup_cards": self.warmup_cards,
            "per_regime": self.per_regime,
            "global": {"bias": self._glob.b, "n": self._glob.n},
            "regimes": {r: {"bias": a.b, "n": a.n} for r, a in self._reg.items()},
        }

    # ------------------------------------------------------------- internal
    def _step(self, acc: _Acc, e: float) -> None:
        acc.b = float(np.clip((1 - self.alpha) * acc.b + self.alpha * e,
                              -self.clip_pp, self.clip_pp))
        acc.n += 1


class ResidualCorrectedModel:
    """Slow loop: frozen base model + additive coverage correction refit between passes.

    Model protocol wrapper (predict / predict_batch / fit). ``fit(records)`` receives the
    NEW verified records (x, u, Outcome with the MEASURED coverage) of the pass just
    finished, appends them to its buffer and refits the correction on ALL records so far:

        r_i   = C_meas,i - C_base(x_i, u_i)          (residual against the FROZEN base)
        two-level normal model: global offset g ~ N(0, prior_sd_pp^2), regime deviation
        d_j ~ N(0, tau^2), regimes = the fast loop's bins of |u_cross| x droplet group.
        g     = sum(r) / (n + s^2 / prior_sd_pp^2)                  (posterior mean)
        tau^2 = max(0, mean_j[(mean(r_j) - g)^2 - s^2 / n_j])       (method of moments)
        c_j   = g + tau^2 / (tau^2 + s^2 / n_j) * (mean(r_j) - g)   (shrunk to g)

    with s^2 the pooled within-regime residual variance (card noise + measurement error).
    A regime never seen gets g. Predictions are base + c (point, lower and upper bound all
    shifted, clipped to [0, 100]); the interval WIDTH stays the base model's, so a refit
    can never narrow the selector's margin.

    Why a residual model rather than refitting the base model (the midterm slow loop):
    the base SprayModel is trained on 50,000 simulated rows. Appending a few dozen noisy
    cards with weight 1 changes almost nothing; with weight 30 the boosters chase the
    noise of individual cards and the CQR offset is recomputed on a calibration split that
    is almost all offline data, so the interval narrowed (half-width 8.0 to 5.2 pp) while
    the error grew, and true coverage fell below the lower bound in 21-27 % of segments
    instead of about 8 %. What field records can actually teach in a few passes is a
    low-dimensional offset that persists across passes (the mean effect of the hidden
    mismatch and the model's own bias near the operating point). A shrunk per-regime
    offset estimates that from a few dozen cards, and the shrinkage toward 0 and toward
    the global offset keeps it from fitting card noise.

    Known limitation: records of one pass share that pass's canopy/wear draw, so they are
    not independent; with one or two earlier passes the between-pass variance cannot be
    separated, and the shrinkage is weaker than an exact hierarchical model would give.
    The fast loop absorbs the pass-specific remainder.
    """

    def __init__(self, base, derive_fn: Callable[[Conditions, Setting], Derived],
                 regime_fn: Callable[[Conditions, Derived], str] | None = None,
                 per_regime: bool = True, prior_sd_pp: float = SLOW_PRIOR_SD_PP):
        self.base = base
        self.derive_fn = derive_fn
        self.per_regime = per_regime
        self._regime_fn = regime_fn or default_regime_fn()
        self.prior_sd_pp = float(prior_sd_pp)
        self._r: list[float] = []
        self._g: list[str] = []
        self.global_corr = 0.0
        self.regime_corr: dict[str, float] = {}
        self.tau_regime_pp = 0.0
        self.noise_sd_pp = float("nan")
        self.history: list[dict] = []

    # ------------------------------------------------------------------ data
    @property
    def n_records(self) -> int:
        return len(self._r)

    def residuals(self) -> np.ndarray:
        return np.asarray(self._r, float)

    def regime(self, x: Conditions, u: Setting, derived: Derived | None = None) -> str:
        if not self.per_regime:
            return GLOBAL_KEY
        return self._regime_fn(x, derived if derived is not None else self.derive_fn(x, u))

    def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None:
        """Append the NEW verified records and refit the correction on all records."""
        recs = [r for r in records if np.isfinite(r[2].coverage_pct)]
        if not recs:
            return
        preds = self.base.predict_batch([r[0] for r in recs], [r[1] for r in recs])
        for (x, u, o), p in zip(recs, preds):
            self._r.append(float(o.coverage_pct) - float(p.coverage_pct))
            self._g.append(self.regime(x, u))
        self._refit()

    def _refit(self) -> None:
        r, g = np.asarray(self._r, float), np.asarray(self._g)
        n = len(r)
        keys = sorted(set(g.tolist()))
        ss, dof = 0.0, 0
        for key in keys:
            rk = r[g == key]
            if len(rk) > 1:
                ss += float(((rk - rk.mean()) ** 2).sum())
                dof += len(rk) - 1
        if dof > 0:
            s2 = ss / dof
        elif n > 1:
            s2 = float(r.var(ddof=1))
        else:
            s2 = self.prior_sd_pp ** 2
        glob = float(r.sum() / (n + s2 / self.prior_sd_pp ** 2))
        reg: dict[str, float] = {}
        tau2 = 0.0
        if self.per_regime and len(keys) >= 2:
            d = np.array([r[g == key].mean() - glob for key in keys])
            v = np.array([s2 / int((g == key).sum()) for key in keys])
            tau2 = max(0.0, float(np.mean(d ** 2 - v)))
            for key, dk, vk in zip(keys, d, v):
                reg[key] = glob + (tau2 / (tau2 + vk) if tau2 > 0 else 0.0) * float(dk)
        self.global_corr, self.regime_corr = glob, reg
        self.tau_regime_pp, self.noise_sd_pp = float(np.sqrt(tau2)), float(np.sqrt(s2))
        self.history.append(self.state())

    # ------------------------------------------------------------------ inference
    def correction(self, x: Conditions, u: Setting, derived: Derived | None = None) -> float:
        """Additive coverage correction (pp) applied to the base prediction for (x, u)."""
        if not self.regime_corr:
            return self.global_corr
        return self.regime_corr.get(self.regime(x, u, derived), self.global_corr)

    def predict_batch(self, xs: Sequence[Conditions], us: Sequence[Setting]) -> list[Prediction]:
        preds = self.base.predict_batch(xs, us)
        if self.n_records == 0:
            return preds
        if self.regime_corr:
            deltas = [self.correction(x, u) for x, u in zip(xs, us)]
        else:
            deltas = [self.global_corr] * len(preds)

        def clip(v: float) -> float:
            return float(min(max(v, 0.0), 100.0))

        return [dataclasses.replace(p, coverage_pct=clip(p.coverage_pct + d),
                                    coverage_lo=clip(p.coverage_lo + d),
                                    coverage_hi=clip(p.coverage_hi + d))
                for p, d in zip(preds, deltas)]

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        return self.predict_batch([x], [u])[0]

    def state(self) -> dict:
        r = self.residuals()
        return {"n_records": self.n_records, "global_corr": self.global_corr,
                "regime_corr": dict(self.regime_corr), "tau_regime_pp": self.tau_regime_pp,
                "noise_sd_pp": self.noise_sd_pp, "prior_sd_pp": self.prior_sd_pp,
                "per_regime": self.per_regime,
                "resid_mean": float(r.mean()) if len(r) else float("nan"),
                "resid_var": float(r.var(ddof=1)) if len(r) > 1 else float("nan")}
