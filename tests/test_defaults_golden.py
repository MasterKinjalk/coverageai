"""Golden digests of default-option runs (every opt-in feature off).

The constants below pin the default behaviour: with the nozzle lock, speed/boom control, forecaster
and card capture all off, results must stay bit-identical. Regenerate them only from a commit where
a deliberate change of the defaults was decided (and record it in Docs/decisions.md).

Policies are run in a fixed order on one module stack per model, as the stack is reused between runs.
"""
from __future__ import annotations

import hashlib

import pytest

from coverageai import app_logic as al
from coverageai import integration
from coverageai.integration import build_modules
from coverageai.loop import run_pass


def digest(logs) -> str:
    rows = [(lg.u.family, lg.u.size, lg.u.pressure_psi, lg.u.travel_speed_kmh, lg.u.boom_height_m,
             round(lg.pred.coverage_pct, 6), round(lg.outcome.coverage_pct, 6), round(lg.meas.coverage_pct, 6),
             round(lg.bias, 6), round(lg.outcome.drift_frac, 8), lg.flagged_retreat, lg.selector_infeasible)
            for lg in logs]
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def _run_all(model: str, verifier: str) -> dict[str, str]:
    conds = al.make_conditions("gusty_afternoon", 1)
    mods = build_modules(model=model, verifier=verifier)
    out = {}
    for policy in ("open", "fast", "oracle"):
        logs = run_pass(list(conds), policy, mods, 1, "med", 30.0, f"g_{policy}", render=False)
        out[f"run_pass:{policy}"] = digest(logs)
    for policy in ("fast", "fast_slow", "best_fixed"):
        res = al.run_pass_wrapper(mods, conds, policy, 1, "med", 30.0, 0.6)
        out[f"wrapper:{policy}"] = digest(res.logs)
    return out


def test_golden_oracle_nominal_perfect():
    assert len(al.make_conditions("gusty_afternoon", 1)) == 40
    got = _run_all("oracle_nominal", "perfect")
    assert got == {
        "run_pass:open": "8b02f28c3aa8288d",
        "run_pass:fast": "e58a6fac16461905",
        "run_pass:oracle": "c77f4a2e31a77c47",
        "wrapper:fast": "e58a6fac16461905",
        "wrapper:fast_slow": "b44e2d12448f4587",
        "wrapper:best_fixed": "91d15b0aee422c02",
    }


def test_golden_spray_noisy():
    if not integration.spray_model_available():
        pytest.skip("trained spray model file is missing")
    got = _run_all("spray", "noisy")
    assert got["run_pass:open"] == "7280286332fa9957"
    assert got["run_pass:fast"] == "9f5575a10991ecb1"
    assert got["run_pass:oracle"] == "c1f9e7c06d4460bb"
    assert got["wrapper:fast_slow"] == "b21da6069fab00f4"
    assert got["wrapper:best_fixed"] == "8be4029a13fbd657"
