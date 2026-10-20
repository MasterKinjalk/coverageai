"""Short-horizon wind forecaster for the closed loop (no ML training: statistics only).

Problem. The selector for segment k plans with the conditions x_k read at the segment start.
The plant then applies a hidden gust multiplier g_k (simulator: AR(1) state, phi = 0.7 per
segment, sd = mismatch gust_sigma), so the wind the spray actually meets is x_k.wind * g_k.
The gust persists from segment to segment, so the realized wind of the segments already
sprayed carries information about the next one.

Observation model (anemometer). After segment j has been sprayed the controller observes
its realized, segment-averaged wind with sensor noise:  y_j = x_j.wind * g_j + nu_j,
nu_j ~ N(0, obs_sd^2) (obs_sd = 0.3 m/s by default, a typical cup/sonic anemometer
accuracy). It never observes g_k of the segment it is about to spray (``loop.run_pass``
calls ``observe`` only after the plant call of that segment).

Forecast target. Because the gust is multiplicative, the forecaster works on the gust ratio
r_j = y_j / x_j.wind (observations with x_j.wind < ``min_wind_ms`` are skipped: the ratio is
dominated by sensor noise there) and forecasts r_k for the next segment; the forecast wind
is x_k.wind * r_hat_k. Direction is taken from x_k unchanged (the gust model only scales the
speed). Methods:

  nominal      r_hat = 1 (today's behaviour: plan with x_k). Still logs forecast vs realized
               and tracks the residual sd, so it is the "forecaster off" arm with logging and
               the base of an uncertainty-only (gust-aware nominal) control.
  last         persistence of the realized wind: r_hat_k = r_{k-1} (last valid observation).
  ewma         r_hat_k = level, level <- (1 - beta) level + beta r_j  (level starts at 1).
  ar1          AR(1) on the ratio, fitted online by shrunk least squares on consecutive pairs:
               mu  = (sum r + n_mu * 1) / (n + n_mu)            (prior mean 1: no gust bias)
               phi = [sum a b / sum a^2] * n / (n + n_phi),  a = r_{j-1} - mu, b = r_j - mu
               (n pairs; n_phi pseudo-pairs with phi = 0 shrink toward the mean forecast
               while data are few), clipped to [-0.95, 0.95];  r_hat_k = mu + phi^h (r_last - mu), h = steps since
               the last valid observation. Fitted on NOISY ratios, so phi is the best linear
               predictor from the last noisy reading (attenuated relative to the true gust phi
               by the signal-to-noise ratio), which is what a one-step forecast should use.

Uncertainty. sd of the one-step ratio forecast error from the most recent ``window``
residuals (r_j - r_hat_j), blended with a prior (``sd_prior`` with weight ``n_sd_prior``).
The residual contains the sensor noise of the observation; when ``deduct_obs_noise`` the
known sensor variance (obs_sd / x_j.wind)^2 is deducted (floored at ``sd_floor``^2), so sd
estimates the error of the forecast of the TRUE gust ratio.

Gust-aware planning. With ``gust_aware`` the selector plans for a conservative wind
x_k.wind * max(0, r_hat + z sd) (more wind = more drift and less deposition in this plant),
while the prediction that the fast-loop bias and the slow loop learn from is still made at
the point forecast, so the feedback does not learn to cancel the conservatism.

History. ``begin_pass`` clears the within-pass state (last observation, EWMA level) but keeps
the AR(1) sufficient statistics and residuals unless ``keep_history`` is False, so in the
fast + slow loop the gust model keeps learning across passes (the gust process of a field is
the same across passes; its realization is not).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

from coverageai.contracts import Conditions

METHODS = ("nominal", "last", "ewma", "ar1")


def _cfg() -> dict:
    try:
        from coverageai.config import spec
        return dict(spec().get("forecast", {}) or {})
    except Exception:   # pragma: no cover - spec unavailable
        return {}


@dataclass(frozen=True)
class WindForecast:
    """Forecast for one segment (all speeds in m/s, ratio = forecast / nominal x.wind)."""
    k: int
    nominal_wind_ms: float        # x_k.wind_speed_ms (what the controller used before)
    ratio: float                  # forecast gust ratio r_hat_k
    sd_ratio: float               # forecast-error sd of the ratio
    wind_ms: float                # point forecast  = nominal * ratio
    sd_wind_ms: float             # = nominal * sd_ratio
    plan_wind_ms: float           # wind the selector plans with (point, or point + z sd)
    n_obs: int                    # valid observations used so far
    method: str                   # forecaster name, e.g. "ar1+gust"
    gust_aware: bool

    def as_extra(self) -> dict:
        return {"fc_method": self.method,
                "fc_nominal_wind_ms": self.nominal_wind_ms, "fc_ratio": self.ratio,
                "fc_sd_ratio": self.sd_ratio, "fc_wind_ms": self.wind_ms,
                "fc_sd_wind_ms": self.sd_wind_ms, "fc_plan_wind_ms": self.plan_wind_ms,
                "fc_n_obs": self.n_obs}


@dataclass
class WindForecaster:
    method: str = "ar1"
    gust_aware: bool = False
    z: float | None = None
    obs_sd_ms: float | None = None
    min_wind_ms: float | None = None
    ewma_beta: float | None = None
    n_mu_prior: float | None = None
    n_phi_prior: float | None = None
    sd_prior: float | None = None
    n_sd_prior: float | None = None
    sd_floor: float | None = None
    window: int | None = None
    deduct_obs_noise: bool | None = None
    keep_history: bool = True
    # ---- state
    _n: int = field(default=0, init=False)
    _sum_r: float = field(default=0.0, init=False)
    _pairs: list = field(default_factory=list, init=False)       # (r_prev, r) consecutive pairs
    _resid: deque = field(default=None, init=False)               # (residual, obs-noise var)
    _last_r: float | None = field(default=None, init=False)
    _last_k: int | None = field(default=None, init=False)
    _level: float = field(default=1.0, init=False)
    _pending: dict = field(default_factory=dict, init=False)      # k -> WindForecast issued

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"forecast method must be one of {METHODS}, got {self.method!r}")
        c = _cfg()
        dflt = dict(z=1.0, obs_sd_ms=0.3, min_wind_ms=0.5, ewma_beta=0.5, n_mu_prior=5.0,
                    n_phi_prior=5.0, sd_prior=0.15, n_sd_prior=3.0, sd_floor=0.02, window=10,
                    deduct_obs_noise=True)
        for k, v in dflt.items():
            if getattr(self, k) is None:
                setattr(self, k, type(v)(c.get(k, v)))
        self._resid = deque(maxlen=int(self.window))

    # ------------------------------------------------------------ factory
    @classmethod
    def from_name(cls, name: str, **kw) -> "WindForecaster":
        """'ar1', 'ewma', 'last', 'nominal'; suffix '+gust' (or '_gust') turns on gust-aware."""
        n = name.replace("_gust", "+gust")
        ga = n.endswith("+gust")
        return cls(method=n[:-5] if ga else n, gust_aware=ga, **kw)

    @property
    def name(self) -> str:
        return self.method + ("+gust" if self.gust_aware else "")

    # ------------------------------------------------------------ pass bookkeeping
    def begin_pass(self, seed: int | None = None, mismatch: str | None = None,
                   simulator=None) -> None:
        """New pass (run_pass passes its seed, mismatch and simulator; only the
        GustOracleForecaster reference uses them)."""
        self._last_r = self._last_k = None
        self._level = 1.0
        self._pending.clear()
        if not self.keep_history:
            self._n, self._sum_r = 0, 0.0
            self._pairs.clear()
            self._resid.clear()

    def reset(self) -> None:
        keep, self.keep_history = self.keep_history, False
        self.begin_pass()
        self.keep_history = keep

    # ------------------------------------------------------------ estimation
    def _mu(self) -> float:
        return (self._sum_r + self.n_mu_prior * 1.0) / (self._n + self.n_mu_prior)

    def _var_ratio(self) -> float:
        n = len(self._resid)
        s = sum(e * e - (v if self.deduct_obs_noise else 0.0) for e, v in self._resid)
        var = (s + self.n_sd_prior * self.sd_prior ** 2) / (n + self.n_sd_prior)
        return max(var, self.sd_floor ** 2)

    def phi(self) -> float:
        """Shrunk least-squares AR(1) coefficient: OLS * n / (n + n_phi_prior)
        (n_phi_prior pseudo-pairs with phi = 0), clipped to [-0.95, 0.95]."""
        if self.method != "ar1" or not self._pairs:
            return 0.0
        mu = self._mu()
        sab = sum((a - mu) * (b - mu) for a, b in self._pairs)
        saa = sum((a - mu) ** 2 for a, _ in self._pairs)
        n = len(self._pairs)
        if saa <= 1e-12:
            return 0.0
        return float(min(max(sab / saa * n / (n + self.n_phi_prior), -0.95), 0.95))

    def _ratio_hat(self, k: int) -> float:
        if self.method == "nominal":
            return 1.0
        if self._last_r is None:            # start of a pass: no reading yet
            return self._mu() if self.method == "ar1" else 1.0
        if self.method == "last":
            return self._last_r
        if self.method == "ewma":
            return self._level
        mu = self._mu()
        h = max(1, k - self._last_k) if self._last_k is not None else 1
        return mu + self.phi() ** h * (self._last_r - mu)

    # ------------------------------------------------------------ API
    def forecast(self, k: int, x: Conditions) -> WindForecast:
        """Forecast for segment k with nominal conditions x (uses only observations < k)."""
        r = max(0.0, float(self._ratio_hat(k)))
        sd = math.sqrt(self._var_ratio())
        w = float(x.wind_speed_ms)
        plan = w * max(0.0, r + self.z * sd) if self.gust_aware else w * r
        f = WindForecast(k=k, nominal_wind_ms=w, ratio=r, sd_ratio=sd, wind_ms=w * r,
                         sd_wind_ms=w * sd, plan_wind_ms=plan, n_obs=self._n,
                         method=self.name, gust_aware=self.gust_aware)
        self._pending[k] = f
        return f

    def observe(self, k: int, x: Conditions, observed_wind_ms: float) -> dict:
        """Anemometer reading of segment k's realized wind (call only after spraying k)."""
        if self._last_k is not None and k <= self._last_k:
            raise ValueError(f"observations must arrive in segment order (got {k} after {self._last_k})")
        w = float(x.wind_speed_ms)
        out: dict = {"fc_obs_wind_ms": float(observed_wind_ms)}
        f = self._pending.pop(k, None)
        if w < self.min_wind_ms:
            out["fc_obs_used"] = False
            return out
        r = float(observed_wind_ms) / w
        if f is not None:
            self._resid.append((r - f.ratio, (self.obs_sd_ms / w) ** 2))
        if self._last_r is not None and self._last_k == k - 1:
            self._pairs.append((self._last_r, r))
        self._n += 1
        self._sum_r += r
        self._level = (1 - self.ewma_beta) * self._level + self.ewma_beta * r
        self._last_r, self._last_k = r, k
        out["fc_obs_used"] = True
        out["fc_obs_ratio"] = r
        return out

    def conditions(self, x: Conditions, f: WindForecast) -> tuple[Conditions, Conditions]:
        """(x_point, x_plan): x with the point-forecast wind, and with the planning wind."""
        return replace(x, wind_speed_ms=f.wind_ms), replace(x, wind_speed_ms=f.plan_wind_ms)

    def state(self) -> dict:
        return {"method": self.method, "gust_aware": self.gust_aware, "z": self.z,
                "n_obs": self._n, "n_pairs": len(self._pairs), "mu": self._mu(),
                "phi": self.phi(), "sd_ratio": math.sqrt(self._var_ratio()),
                "obs_sd_ms": self.obs_sd_ms}


class GustOracleForecaster(WindForecaster):
    """REFERENCE ONLY, not deployable: knows the realized wind of segment k before it is
    sprayed (reads the plant's hidden gust from the same random stream, like the fair
    oracle), with zero error. Its result bounds what ANY wind forecaster can gain inside
    this simulator; everything else (canopy, wear, card noise) stays unknown to it.
    Requires a SpraySimulator-like plant (``_draw_segment``) and run_pass's begin_pass
    context; otherwise it falls back to the nominal wind."""

    def __init__(self, **kw):
        super().__init__(method="nominal", gust_aware=False, **kw)
        self._sim = self._seed = self._mm = None

    @property
    def name(self) -> str:
        return "gust_oracle"

    def begin_pass(self, seed=None, mismatch=None, simulator=None) -> None:
        super().begin_pass()
        self._seed, self._mm, self._sim = seed, mismatch, simulator

    def forecast(self, k: int, x: Conditions) -> WindForecast:
        g = 1.0
        if self._sim is not None and hasattr(self._sim, "_draw_segment") and self._seed is not None:
            import numpy as np
            # same stream as loop.segment_rng(seed, k); _draw_segment is idempotent for the
            # pass's AR(1) gust state, so the plant call later sees the same gust.
            hid, _ = self._sim._draw_segment(np.random.default_rng([self._seed, k]), self._mm)
            g = float(hid.gust_mult)
        w = float(x.wind_speed_ms)
        f = WindForecast(k=k, nominal_wind_ms=w, ratio=g, sd_ratio=0.0, wind_ms=w * g,
                         sd_wind_ms=0.0, plan_wind_ms=w * g, n_obs=self._n, method=self.name,
                         gust_aware=False)
        self._pending[k] = f
        return f


def make_forecaster(spec_or_obj) -> WindForecaster | None:
    """None -> None; a WindForecaster -> itself; a name ('ar1', 'ar1+gust', ...) -> new one."""
    if spec_or_obj is None or isinstance(spec_or_obj, WindForecaster):
        return spec_or_obj
    if isinstance(spec_or_obj, str):
        if spec_or_obj == "gust_oracle":
            return GustOracleForecaster()
        return WindForecaster.from_name(spec_or_obj)
    raise TypeError(f"forecaster must be None, a name or a WindForecaster, got {type(spec_or_obj)}")
