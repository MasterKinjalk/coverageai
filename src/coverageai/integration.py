"""WP-20 integration: wire the real modules into ``loop.Modules``.

    build_modules(model="spray", verifier="perfect") -> loop.Modules

model:
  spray           learned SprayModel (WP-10), loaded from models/spray_model_v1.joblib
                  (searched in the repo root, its parents, or $COVERAGEAI_MODEL).
  oracle_nominal  NominalPhysicsModel: the simulator with mismatch "none" and the WSP
                  spread factor at its mean. Physics-perfect but blind to the hidden
                  mismatch (gust, canopy, wear). Stand-in until / alongside WP-10 and the
                  clean "model error = mismatch only" control.
  mock            mocks.MockModel with the whole mock stack (fast unit tests only).
verifier:
  perfect         measurement = true outcome coverage (no image).
  noisy           true coverage + Gaussian error (bias, sd from results/b0_synthetic.json
                  or b1_synthetic.json when present; default sd 1.5 pp). No image.
  b0              WSPRenderer card image -> vision.classical.ClassicalVerifier.
  b1              WSPRenderer card image -> learned U-Net verifier (when merged).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import (CardMeasurement, Conditions, Outcome, Prediction, Setting,
                                  coverage_class)
from coverageai.loop import Modules, default_mock_modules

MODEL_CHOICES = ("spray", "oracle_nominal", "mock")
VERIFIER_CHOICES = ("perfect", "noisy", "b0", "b1")
DEFAULT_NOISE_SD_PP = 1.5
MODEL_FILENAME = "spray_model_v1.joblib"


# ------------------------------------------------------------------ models

class NominalPhysicsModel:
    """Model protocol wrapper around SpraySimulator at mismatch "none" (vectorized).

    Knows the physics and the catalog exactly, but not the hidden mismatch. The only
    randomness left in the "none" simulator is the WSP spread factor; the point prediction
    uses its mean and the prediction interval its 10/90 % quantiles. ``fit`` is a no-op
    (nothing to learn: the model is already the nominal truth).
    """

    def __init__(self, simulator=None):
        from coverageai.simulator import SpraySimulator
        self.simulator = simulator or SpraySimulator()

    def predict_batch(self, xs: Sequence[Conditions], us: Sequence[Setting]) -> list[Prediction]:
        if len(xs) == 0:
            return []
        a = self.simulator.nominal_arrays(xs, us)
        cov = a["coverage_pct"]
        return [Prediction(float(cov[i]), float(a["coverage_lo"][i]), float(a["coverage_hi"][i]),
                           float(a["deposition_ul_cm2"][i]), float(a["drift_frac"][i]),
                           float(a["displacement_mm"][i])) for i in range(len(xs))]

    def predict(self, x: Conditions, u: Setting) -> Prediction:
        return self.predict_batch([x], [u])[0]

    def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None:
        pass


def find_spray_model_path(path: str | Path | None = None) -> Path | None:
    """Explicit path, $COVERAGEAI_MODEL, or models/spray_model_v1.joblib in ROOT or a parent
    directory (worktrees live below the main checkout)."""
    cands: list[Path] = []
    if path:
        cands.append(Path(path))
    if os.environ.get("COVERAGEAI_MODEL"):
        cands.append(Path(os.environ["COVERAGEAI_MODEL"]))
    for d in [ROOT, *ROOT.parents]:
        cands.append(d / "models" / MODEL_FILENAME)
    return next((c for c in cands if c.is_file()), None)


def load_spray_model(path: str | Path | None = None):
    try:
        from coverageai.model.predictor import SprayModel
    except ImportError as e:   # WP-10 not merged
        raise RuntimeError("model 'spray' needs coverageai.model.predictor (WP-10); "
                           "use model='oracle_nominal' until it is merged") from e
    p = find_spray_model_path(path)
    if p is None:
        raise FileNotFoundError(f"trained SprayModel not found (looked for {MODEL_FILENAME})")
    return SprayModel.load(p)


def spray_model_available() -> bool:
    try:
        import coverageai.model.predictor  # noqa: F401
    except ImportError:
        return False
    return find_spray_model_path() is not None


# ------------------------------------------------------------------ verifiers

@dataclass
class NoisyVerifier:
    """Perfect coverage plus Gaussian error: meas = clip(true + bias + sd * N(0,1), 0, 100).

    Image-free stand-in for a rendered card read by a vision model; lets the experiment grid
    run without rendering. Read by loop.run_pass through ``measure_outcome``.
    """
    sd_pp: float = DEFAULT_NOISE_SD_PP
    bias_pp: float = 0.0
    source: str = "default"

    def measure_outcome(self, outcome: Outcome, rng: np.random.Generator) -> CardMeasurement:
        c = float(np.clip(outcome.coverage_pct + self.bias_pp + self.sd_pp * rng.standard_normal(),
                          0.0, 100.0))
        s = spec()["coverage_classes"]
        return CardMeasurement(c, outcome.stains_per_cm2, None, None, None,
                               coverage_class(c, s["acceptable_min"], s["marginal_min"]),
                               False, 0.0)

    def measure(self, image, px_per_mm):  # pragma: no cover - protocol placeholder
        raise NotImplementedError("NoisyVerifier reads the true outcome via measure_outcome()")


def noise_params_from_results(which: str | None = None, results_dir: Path | None = None
                              ) -> tuple[float, float, str]:
    """(sd_pp, bias_pp, source) from results/{b0,b1}_synthetic.json (overall block): sd is
    `rmse_pp` if present, else the Gaussian-equivalent mae * sqrt(pi/2); bias is `bias_pp`.
    Falls back to (1.5, 0, "default") when no file is present."""
    d = results_dir or ROOT / "results"
    for name in ([which] if which else ["b0", "b1"]):
        f = d / f"{name}_synthetic.json"
        if not f.is_file():
            continue
        o = json.loads(f.read_text(encoding="utf8")).get("overall", {})
        if "mae_pp" not in o and "rmse_pp" not in o:
            continue
        bias = float(o.get("bias_pp", 0.0))
        sd = float(o["rmse_pp"]) if "rmse_pp" in o else float(o["mae_pp"]) * float(np.sqrt(np.pi / 2))
        if "rmse_pp" in o:
            sd = float(np.sqrt(max(sd ** 2 - bias ** 2, 0.0)))
        return sd, bias, f.name
    return DEFAULT_NOISE_SD_PP, 0.0, "default"


def image_verifier_available(name: str) -> bool:
    try:
        make_image_verifier(name)
        return True
    except (ImportError, NotImplementedError, AttributeError, FileNotFoundError):
        return False


def make_image_verifier(name: str):
    if name == "b0":
        from coverageai.vision.classical import ClassicalVerifier
        return ClassicalVerifier()
    if name == "b1":
        from coverageai.vision.unet import UNetVerifier   # WP-13
        return UNetVerifier()
    raise ValueError(name)


# ------------------------------------------------------------------ factory

def build_modules(model: str = "spray", verifier: str = "perfect", *,
                  model_path: str | Path | None = None,
                  noise_sd_pp: float | None = None, noise_bias_pp: float | None = None,
                  noise_source: str | None = None, results_dir: Path | None = None,
                  model_obj=None) -> Modules:
    """Assemble loop.Modules from real components (see module docstring)."""
    if model not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}, got {model!r}")
    if verifier not in VERIFIER_CHOICES:
        raise ValueError(f"verifier must be one of {VERIFIER_CHOICES}, got {verifier!r}")

    if model == "mock":
        m = default_mock_modules()
        sim, mdl, sel, rend, derive = m.simulator, m.model, m.selector, m.renderer, m.derive_fn
    else:
        from coverageai import catalog
        from coverageai.selector import ConstrainedSelector
        from coverageai.simulator import SpraySimulator
        from coverageai.vision.render import WSPRenderer
        sim = SpraySimulator(derive_fn=catalog.derive)
        sel = ConstrainedSelector(candidates_fn=catalog.candidates, derive_fn=catalog.derive)
        rend, derive = WSPRenderer(), catalog.derive
        if model_obj is not None:
            mdl = model_obj
        elif model == "oracle_nominal":
            mdl = NominalPhysicsModel(sim)
        else:
            mdl = load_spray_model(model_path)

    if verifier == "perfect":
        ver = None
    elif verifier == "noisy":
        sd, bias, src = noise_params_from_results(None, results_dir)
        if noise_sd_pp is not None or noise_bias_pp is not None:
            sd = DEFAULT_NOISE_SD_PP if noise_sd_pp is None else noise_sd_pp
            bias = 0.0 if noise_bias_pp is None else noise_bias_pp
            src = noise_source or "explicit"
        ver = NoisyVerifier(sd, bias, src)
    else:
        ver = make_image_verifier(verifier)
    return Modules(simulator=sim, model=mdl, selector=sel, verifier=ver, renderer=rend,
                   derive_fn=derive)


def fresh_model(modules: Modules, template=None):
    """A deep copy of the model so a slow-loop run does not leak refits into other runs."""
    return copy.deepcopy(template if template is not None else modules.model)
