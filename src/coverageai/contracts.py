"""Shared records and module interfaces for CoverageVision AI.

Frozen at schema version 1.0. Only the orchestrator changes this file; bump
SCHEMA_VERSION on any change. Notation follows the proposal: x = conditions the
controller cannot set, u = spray setting it chooses.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

import numpy as np

SCHEMA_VERSION = "1.0"

CoverageClass = Literal["acceptable", "marginal", "critical"]
MismatchLevel = Literal["none", "low", "med", "high"]


@dataclass(frozen=True)
class Conditions:
    """x: environment and machine state at a segment."""
    wind_speed_ms: float
    wind_dir_deg: float          # direction the airflow moves toward (decision D1)
    heading_deg: float
    temp_c: float
    rh_pct: float
    travel_speed_kmh: float
    boom_height_m: float


@dataclass(frozen=True)
class Setting:
    """u: spray setting chosen by the selector."""
    family: str                  # TP65, ULD, 3D, XR, DGX40
    size: str                    # manufacturer size code, e.g. "03"
    pressure_psi: float


@dataclass(frozen=True)
class Derived:
    """Catalog lookups and geometry derived from (x, u)."""
    flow_gpm: float
    app_rate_gpa: float
    app_rate_lha: float
    droplet_class: str
    vmd_um: float
    spray_angle_deg: float
    rel_wind_deg: float          # wind_dir - heading, wrapped to (-180, 180]
    u_cross_ms: float
    u_along_ms: float


@dataclass(frozen=True)
class DropletSample:
    """Droplets deposited on the card, used by the renderer. Card coordinates in mm."""
    diameters_um: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    spread_factor: float


@dataclass(frozen=True)
class Outcome:
    """Simulator ground truth for one segment."""
    coverage_pct: float
    deposition_ul_cm2: float
    drift_frac: float
    displacement_mm: float
    stains_per_cm2: float
    droplets: DropletSample | None = None


@dataclass(frozen=True)
class Prediction:
    coverage_pct: float
    coverage_lo: float
    coverage_hi: float
    deposition_ul_cm2: float
    drift_frac: float
    displacement_mm: float


@dataclass(frozen=True)
class CardMeasurement:
    """Vision output for one WSP card."""
    coverage_pct: float
    stains_per_cm2: float | None
    dv01_um: float | None
    dv05_um: float | None
    dv09_um: float | None
    coverage_class: CoverageClass
    saturated: bool
    runtime_s: float


@dataclass
class SegmentLog:
    """One row per segment of a field pass."""
    pass_id: str
    seed: int
    k: int
    policy: str
    x: Conditions
    u: Setting
    derived: Derived
    pred: Prediction
    corrected_pred_cov: float
    outcome: Outcome
    meas: CardMeasurement
    residual: float
    bias: float
    flagged_retreat: bool
    selector_infeasible: bool
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- interfaces

@runtime_checkable
class Model(Protocol):
    def predict(self, x: Conditions, u: Setting) -> Prediction: ...
    def predict_batch(self, xs: list[Conditions], us: list[Setting]) -> list[Prediction]: ...
    def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None: ...


@runtime_checkable
class Selector(Protocol):
    def select(self, x: Conditions, model: Model, bias: float, c_req: float) -> tuple[Setting, dict]:
        """Return the chosen setting and info: {"infeasible": bool, "candidates": [...]}."""
        ...


@runtime_checkable
class Verifier(Protocol):
    def measure(self, image: np.ndarray, px_per_mm: float) -> CardMeasurement: ...


@runtime_checkable
class Simulator(Protocol):
    def simulate(self, x: Conditions, u: Setting, rng: np.random.Generator,
                 mismatch: MismatchLevel = "none", sample_droplets: bool = False) -> Outcome: ...


@runtime_checkable
class Renderer(Protocol):
    def render(self, outcome: Outcome, rng: np.random.Generator,
               degrade: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return (RGB uint8 card image, boolean ground-truth stain mask)."""
        ...


# ---------------------------------------------------------------- helpers

def coverage_class(coverage_pct: float, acceptable_min: float = 30.0,
                   marginal_min: float = 15.0) -> CoverageClass:
    if coverage_pct >= acceptable_min:
        return "acceptable"
    if coverage_pct >= marginal_min:
        return "marginal"
    return "critical"


def to_dict(obj: Any) -> Any:
    """JSON-safe dict of a contract record; numpy arrays become lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def segment_log_to_json(log: SegmentLog, include_droplets: bool = False) -> str:
    d = to_dict(log)
    if not include_droplets:
        d["outcome"]["droplets"] = None
    d["schema_version"] = SCHEMA_VERSION
    return json.dumps(d)


def flatten_log(log: SegmentLog) -> dict[str, Any]:
    """Flat dict for DataFrames: prefixes x_, u_, d_, pred_, out_, meas_."""
    row: dict[str, Any] = {
        "pass_id": log.pass_id, "seed": log.seed, "k": log.k, "policy": log.policy,
        "corrected_pred_cov": log.corrected_pred_cov, "residual": log.residual,
        "bias": log.bias, "flagged_retreat": log.flagged_retreat,
        "selector_infeasible": log.selector_infeasible,
    }
    for prefix, rec in (("x_", log.x), ("u_", log.u), ("d_", log.derived),
                        ("pred_", log.pred), ("out_", log.outcome), ("meas_", log.meas)):
        for f in dataclasses.fields(rec):
            if f.name == "droplets":
                continue
            row[prefix + f.name] = getattr(rec, f.name)
    row.update({f"extra_{k}": v for k, v in log.extra.items()})
    return row
