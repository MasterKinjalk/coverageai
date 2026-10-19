"""End-to-end smoke test. Runs a short pass with mock modules and writes a
SegmentLog JSONL. WP-06 replaces the inline loop with coverageai.loop.run_pass.

Usage: uv run python -m coverageai.e2e --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coverageai.config import ROOT, px_per_mm, spec
from coverageai.contracts import Conditions, SegmentLog, segment_log_to_json
from coverageai.mocks import (MockModel, MockRenderer, MockSelector, MockSimulator,
                              MockVerifier, mock_derive)


def smoke(n_segments: int = 10, seed: int = 0, out: Path | None = None) -> list[SegmentLog]:
    rng = np.random.default_rng(seed)
    sim, model, sel, ren, ver = MockSimulator(), MockModel(), MockSelector(), MockRenderer(), MockVerifier()
    c_req = spec()["c_req_pct"]
    bias, logs = 0.0, []
    for k in range(n_segments):
        x = Conditions(wind_speed_ms=float(rng.uniform(0, 5)), wind_dir_deg=90.0, heading_deg=0.0,
                       temp_c=25.0, rh_pct=50.0, travel_speed_kmh=16.0, boom_height_m=0.5)
        u, info = sel.select(x, model, bias, c_req)
        pred = model.predict(x, u)
        outcome = sim.simulate(x, u, rng, mismatch="med")
        img, _ = ren.render(outcome, rng)
        meas = ver.measure(img, px_per_mm())
        resid = meas.coverage_pct - (pred.coverage_pct + bias)
        bias = 0.7 * bias + 0.3 * (meas.coverage_pct - pred.coverage_pct)
        logs.append(SegmentLog("smoke", seed, k, "fast", x, u, mock_derive(x, u), pred,
                               pred.coverage_pct + bias, outcome, meas, resid, bias,
                               meas.coverage_class == "critical", info["infeasible"]))
    out = out or ROOT / "results" / "smoke.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(segment_log_to_json(l) for l in logs) + "\n", encoding="utf8")
    return logs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--segments", type=int, default=10)
    args = ap.parse_args()
    logs = smoke(args.segments)
    met = sum(l.meas.coverage_pct >= spec()["c_req_pct"] for l in logs)
    print(f"smoke ok: {len(logs)} segments, {met} met C_req, final bias {logs[-1].bias:+.2f} pp")


if __name__ == "__main__":
    main()
