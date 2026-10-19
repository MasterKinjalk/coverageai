"""End-to-end smoke test: runs every policy through coverageai.loop with mock modules.

Usage: uv run python -m coverageai.e2e --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coverageai.config import ROOT, spec
from coverageai.contracts import Conditions, SegmentLog, Setting
from coverageai.loop import (Modules, default_mock_modules, run_pass, run_passes_slow,
                             save_logs, segment_rng, summarize)
from coverageai.mocks import mock_candidates


def make_conditions(n: int, seed: int) -> list[Conditions]:
    """Toy pass: random wind, heading reversals every segment (back-and-forth)."""
    rng = np.random.default_rng(seed)
    return [Conditions(wind_speed_ms=float(rng.uniform(0, 5)), wind_dir_deg=90.0,
                       heading_deg=0.0 if k % 2 == 0 else 180.0, temp_c=25.0, rh_pct=50.0,
                       travel_speed_kmh=16.0, boom_height_m=0.5) for k in range(n)]


def pick_best_fixed(conditions: list[Conditions], modules: Modules, c_req: float,
                    mismatch: str, seed: int, candidates: list[Setting]) -> Setting:
    """Fixed setting with the highest requirement-met rate on the scenario (ties: lowest
    drift). Smoke-test helper only; WP-11 owns the real best_fixed baseline."""
    best, best_key = candidates[0], None
    for u in candidates:
        outs = [modules.simulator.simulate(x, u, segment_rng(seed, k), mismatch)
                for k, x in enumerate(conditions)]
        key = (-np.mean([o.coverage_pct >= c_req for o in outs]),
               np.mean([o.drift_frac for o in outs]))
        if best_key is None or key < best_key:
            best, best_key = u, key
    return best


def run_all_policies(n_segments: int = 10, seed: int = 0, modules: Modules | None = None,
                     mismatch: str = "med", out_dir: Path | None = None
                     ) -> dict[str, list[SegmentLog]]:
    m = modules or default_mock_modules()
    c_req = spec()["c_req_pct"]
    conds = make_conditions(n_segments, seed)
    fixed = pick_best_fixed(conds, m, c_req, mismatch, seed, mock_candidates())
    res = {
        pol: run_pass(conds, pol, m, seed, mismatch, c_req, f"smoke_{pol}", fixed_setting=fixed)
        for pol in ("open", "fast", "oracle", "best_fixed")
    }
    res["fast_slow"] = run_passes_slow(conds, m, seed, mismatch, c_req, n_passes=2)
    if out_dir is not None:
        for pol, logs in res.items():
            save_logs(logs, out_dir / f"smoke_{pol}")
    return res


def smoke(n_segments: int = 10, seed: int = 0, out: Path | None = None) -> list[SegmentLog]:
    """Run all policies with mocks; write the `fast` policy log as JSONL to `out`
    (default results/smoke.jsonl, plus parquet) and parquet+JSONL for every policy to results/runs/smoke when
    `out` is None. Returns the `fast` logs."""
    out_dir = ROOT / "results" / "runs" / "smoke" if out is None else None
    res = run_all_policies(n_segments, seed, out_dir=out_dir)
    fast = res["fast"]
    path = out or ROOT / "results" / "smoke.jsonl"
    save_logs(fast, path)
    return fast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--segments", type=int, default=10)
    args = ap.parse_args()
    res = run_all_policies(args.segments, out_dir=ROOT / "results" / "runs" / "smoke")
    c_req = spec()["c_req_pct"]
    print(f"smoke ok: {args.segments} segments per policy, mismatch=med, C_req={c_req}")
    for pol, logs in res.items():
        s = summarize(logs, c_req)
        print(f"  {pol:10s} n={s['n']:3d} met={s['frac_met']:.2f} shortfall={s['mean_shortfall_pp']:.2f}pp "
              f"vol={s['mean_volume_gpa']:.1f}GPA drift={s['mean_drift_frac']:.3f} "
              f"retreat={s['n_retreat']} infeasible={s['n_infeasible']} final_bias={logs[-1].bias:+.2f}")


if __name__ == "__main__":
    main()
