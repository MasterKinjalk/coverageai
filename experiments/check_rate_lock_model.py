"""Is SprayModel v1 accurate along constant-application-rate lines?

A rate-lock controller fixes nozzle and rate (gal/acre) and moves pressure, with travel speed
following so the rate stays constant. This script scores the default model on those lines against
the mismatch-free simulator (nominal_arrays). Development seed 30000+; never evaluation seeds.

    uv run --no-sync python experiments/check_rate_lock_model.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from coverageai import catalog, units
from coverageai.config import ROOT, spec
from coverageai.contracts import Setting
from coverageai.integration import load_spray_model
from coverageai.scenarios import random_conditions
from coverageai.simulator import SpraySimulator

SEED = 30017
N_COND = 300
PER_LINE = 8            # conditions drawn per line (subsample for runtime)
GPAS = (5, 10, 15, 20, 30)
S0 = 15.0               # reference speed for the GPA formula (km/h)
FEAS = 30.0             # coverage feasibility limit (pp)
OUT = ROOT / "results" / "rate_lock_model_check.md"
REF_JSON = ROOT / "results" / "model_v1.json"


def lines():
    """(family, size, gpa, [Setting...]) with speeds inside the spec range."""
    s = spec()
    lo_s, hi_s = s["ranges"]["travel_speed_kmh"]
    sp = s["boom"]["nozzle_spacing_in"]
    by = defaultdict(list)
    for st in catalog.candidates(5.0):
        by[(st.family, st.size)].append(st.pressure_psi)
    out = []
    for (fam, size), ps in by.items():
        for g in GPAS:
            us = []
            for p in ps:
                flow, _ = catalog.flow_and_class(fam, size, p)
                g0 = units.gpa(flow, units.kmh_to_mph(S0), sp)
                v = S0 * g0 / g
                if lo_s - 1e-9 <= v <= hi_s + 1e-9:
                    us.append(Setting(fam, size, p, travel_speed_kmh=float(v)))
            if len(us) >= 3:
                out.append((fam, size, g, us))
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    conds = random_conditions(N_COND, SEED)
    model = load_spray_model()
    sim = SpraySimulator()
    xs, us, meta = [], [], []
    all_lines = lines()
    for li, (fam, size, g, line) in enumerate(all_lines):
        for ci in rng.choice(N_COND, PER_LINE, replace=False):
            for u in line:
                xs.append(conds[ci])
                us.append(u)
                meta.append((li, int(ci), fam, g))
    n_lines = len(all_lines)
    truth = sim.nominal_arrays(xs, us)
    pr = model.predict_batch(xs, us)
    tc, td = truth["coverage_pct"], truth["drift_frac"]
    pc = np.array([p.coverage_pct for p in pr])
    pd_ = np.array([p.drift_frac for p in pr])
    li_a = np.array([m[0] for m in meta])
    ci_a = np.array([m[1] for m in meta])
    fam_a = np.array([m[2] for m in meta])
    g_a = np.array([m[3] for m in meta])

    def stats(mask):
        ec, ed = (pc - tc)[mask], (pd_ - td)[mask]
        return (int(mask.sum()), np.abs(ec).mean(), ec.mean(), np.abs(ed).mean(), ed.mean())

    keys = defaultdict(list)
    for i, (l, c) in enumerate(zip(li_a, ci_a)):
        keys[(l, c)].append(i)
    rho, ok, nfeas, regret = [], 0, 0, []
    for idx in keys.values():
        idx = np.array(idx)
        if np.ptp(tc[idx]) > 1e-9 and np.ptp(pc[idx]) > 1e-9:
            rho.append(spearmanr(pc[idx], tc[idx])[0])
        f = idx[pc[idx] >= FEAS]
        if len(f):
            nfeas += 1
            b = f[np.argmin(pd_[f])]
            ok += tc[b] >= FEAS
            ft = idx[tc[idx] >= FEAS]
            if len(ft) and tc[b] >= FEAS:
                regret.append(td[b] - td[ft].min())
    rho = np.array(rho)

    rows_g = [(f"{g}",) + stats(g_a == g) for g in GPAS]
    rows_f = [(f,) + stats(fam_a == f) for f in sorted(set(fam_a))]
    allr = stats(np.ones(len(tc), bool))

    ref = None
    try:
        d = json.load(open(REF_JSON))["splits"]["random"]["models"]["lightgbm"]
        ref = (d["coverage_pct"]["mae"], d["coverage_pct"]["mae_vs_expected"],
               d["drift_frac"]["mae"])
    except Exception:
        pass

    def tab(rows, head):
        s = f"| {head} | rows | cov MAE (pp) | cov bias (pp) | drift MAE | drift bias |\n"
        s += "|---|---|---|---|---|---|\n"
        for r in rows:
            s += f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:+.3f} | {r[4]:.4f} | {r[5]:+.4f} |\n"
        return s

    ratio = allr[1] / ref[1] if ref else float("nan")
    retrain = ratio > 1.5
    refmae = ref[1] if ref else float("nan")
    refblock = ""
    if ref:
        refblock = (
            f"- Coverage MAE vs sampled outcomes: {ref[0]:.3f} pp (noise floor about 2.77 pp, "
            "so this is mostly sampling noise).\n"
            f"- Coverage MAE vs the expected (mismatch-free) value: {ref[1]:.3f} pp. This is the "
            "fair reference, because the truth here is also the expected value.\n"
            f"- Drift MAE: {ref[2]:.4f}.\n"
            f"- Rate-lock coverage MAE is {ratio:.2f} times the vs-expected reference.\n")
    ranking = "good" if rho.mean() > 0.9 else "weaker than hoped"
    md = f"""# Rate-lock model check

Question: is the default SprayModel v1 accurate along constant-application-rate lines?
Nozzle and rate are fixed, pressure moves, travel speed follows
(speed = s0 * GPA(P at s0) / target, s0 = {S0:g} km/h, kept inside the spec speed range).
Simulation only.

## Setup
- Truth: `SpraySimulator.nominal_arrays` (mismatch-free, deterministic).
- Model: `models/spray_model_v1.joblib` through `integration.load_spray_model`.
- {N_COND} conditions from `random_conditions(n, seed={SEED})` (development seed).
- {n_lines} lines (nozzle x target GPA on the 5 psi grid, at least 3 points).
  Each line uses {PER_LINE} random conditions. {len(tc)} predictions.
- Target GPA levels: {', '.join(map(str, GPAS))}.

## Error against the nominal truth
{tab([("all",) + allr], "group")}
{tab(rows_g, "GPA")}
{tab(rows_f, "family")}
## Reference (general test set)
Source: `results/model_v1.json`, LightGBM v1, random split ({"found" if ref else "NOT FOUND"}).
{refblock}
## Ranking along pressure (same condition, nozzle and GPA)
- Spearman correlation, predicted vs true coverage: mean {rho.mean():.3f}, median {np.median(rho):.3f}, 10th percentile {np.percentile(rho, 10):.3f} ({len(rho)} lines with variation).
- Among {nfeas} lines with a predicted-feasible point (cov >= {FEAS:g}), the predicted-lowest-drift point is truly feasible in {ok / nfeas:.1%}.
- When it is feasible, its true drift exceeds the best truly feasible point by {np.mean(regret):.4f} on average.

## Verdict
{"Retrain needed." if retrain else "Retrain not needed."} Coverage MAE on rate-lock lines is {allr[1]:.2f} pp against {refmae:.2f} pp on the general test set (vs expected), a ratio of {ratio:.2f}. Bias is {allr[2]:+.2f} pp. Within a line, pressure ranking is {ranking} (mean Spearman {rho.mean():.2f}), and the drift-minimising feasible pick is truly feasible {ok / nfeas:.0%} of the time. Not tested: closed-loop performance with a rate-lock selector, hidden mismatch (gust, canopy, wear) on these lines, other s0 values, other seeds, boom-height changes, the XGBoost v2 or hybrid models, and any field data.
"""
    OUT.write_text(md, encoding="utf-8")
    print(md)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
