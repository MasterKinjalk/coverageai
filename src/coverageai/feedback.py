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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from coverageai.config import spec
from coverageai.contracts import CardMeasurement, Conditions, Derived

GLOBAL_KEY = "global"
FALLBACK_DAMPING = 0.5


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
