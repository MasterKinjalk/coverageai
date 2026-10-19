"""One-off diagnostic for the slow loop (WP-21): interval calibration and error of the SprayModel
before/after weighted refits (6 seeds, gusty_afternoon, mismatch none, record weight 30).
Usage: uv run python experiments/slow_loop_diagnostic.py   (about 10 min)
"""
import copy
import numpy as np
from coverageai import integration, scenarios
from coverageai.loop import run_passes_slow

m = integration.build_modules("spray", "noisy", record_weight=30.0, n_jobs=2)
base = copy.deepcopy(m.model)
rows = []
for s in range(6):
    m.model = copy.deepcopy(base)
    conds = scenarios.generate(scenarios.get("gusty_afternoon"), s)
    logs = run_passes_slow(conds, m, s, "none", 30.0, 3)
    n = len(conds)
    for i in range(3):
        L = logs[i * n:(i + 1) * n]
        true = np.array([l.outcome.coverage_pct for l in L])
        lo = np.array([l.pred.coverage_lo for l in L])
        c = np.array([l.pred.coverage_pct for l in L])
        hi = np.array([l.pred.coverage_hi for l in L])
        rows.append((i, np.mean(true < lo), np.mean(true < 30.0), np.mean(c - lo), np.mean(true - c), np.mean(np.abs(true - c))))
rows = np.array(rows)
for i in range(3):
    r = rows[rows[:, 0] == i].mean(axis=0)
    print(f"pass {i}: frac true<lo={r[1]:.3f} frac true<30={r[2]:.3f} mean(c-lo)={r[3]:.2f} mean err={r[4]:.2f} MAE={r[5]:.2f}")
