"""Multi-seed policy comparison (app_logic.compare_seeds / comparison_stats) and the bootstrap
helpers in coverageai.stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coverageai import app_logic as al
from coverageai import stats

POLS = ("open", "fast", "oracle")
COLS = ["seed", "policy", "met_pct", "shortfall_pp", "volume_gpa", "drift_pct", "n_retreat",
        "n_infeasible"]


@pytest.fixture(scope="module")
def modules():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def three_seeds(modules):
    calls = []
    out = al.compare_seeds(modules, "gusty_afternoon", [0, 1, 2], "high", 30.0, policies=POLS,
                           progress=lambda d, t: calls.append((d, t)))
    return out, calls


def test_one_seed_reproduces_compare_policies(modules):
    pols = ("open", "fast", "fast_slow", "oracle", "best_fixed")
    out = al.compare_seeds(modules, "gusty_afternoon", [3], "med", 30.0, policies=pols, n_learn=1)
    conds = al.make_conditions("gusty_afternoon", 3)
    ref = al.comparison_table(al.compare_policies(
        modules, conds, 3, "med", 30.0, policies=pols, n_learn=1,
        fixed_setting=al.best_fixed_for(modules, conds, 30.0)))
    ps = out["per_seed"]
    assert list(ps.columns) == COLS and list(ps["policy"]) == list(pols)
    assert np.allclose(ps["met_pct"], ref["requirement met (%)"])
    assert np.allclose(ps["volume_gpa"], ref["mean volume (GPA)"])
    assert np.allclose(ps["drift_pct"], 100 * ref["mean drift (frac)"])
    assert np.allclose(ps["shortfall_pp"], ref["mean shortfall (pp)"])
    assert list(ps["n_retreat"]) == list(ref["re-treatment flags"])
    assert set(out["first"]) == set(pols) and out["seeds"] == [3]
    # every policy, including fast_slow, ran on the same field
    assert {r.seed for r in out["first"].values()} == {3}
    st = al.comparison_stats(ps, reference="oracle")
    assert st[["met_lo", "met_hi", "volume_lo", "drift_hi", "d_met_lo", "d_met_hi"]].isna().all().all()
    assert np.allclose(st["met_pct"], ps["met_pct"])


def test_seeds_are_paired_and_progress_counts_runs(three_seeds):
    out, calls = three_seeds
    ps = out["per_seed"]
    assert out["seeds"] == [0, 1, 2] and out["policies"] == list(POLS)
    assert len(ps) == 9
    assert ps.groupby("seed")["policy"].apply(list).tolist() == [list(POLS)] * 3
    assert set(out["first"]) == set(POLS) and all(r.seed == 0 for r in out["first"].values())
    assert calls == [(i, 9) for i in range(1, 10)]


def test_comparison_stats_bounds_and_paired_difference(three_seeds):
    ps = three_seeds[0]["per_seed"]
    st = al.comparison_stats(ps, reference="oracle")
    assert list(st["policy"]) == list(POLS) and (st["n_seeds"] == 3).all()
    for m in ("met", "volume", "drift"):
        mean = st["met_pct" if m == "met" else f"{m}_gpa" if m == "volume" else "drift_pct"]
        assert (st[f"{m}_lo"] <= mean + 1e-12).all() and (mean <= st[f"{m}_hi"] + 1e-12).all()
    ref = st.set_index("policy").loc["oracle"]
    assert np.isnan(ref["d_met_pp"]) and np.isnan(ref["d_met_lo"]) and np.isnan(ref["d_met_hi"])
    op = st.set_index("policy").loc["open"]
    piv = ps.pivot(index="seed", columns="policy", values="met_pct")
    assert op["d_met_pp"] == pytest.approx((piv["open"] - piv["oracle"]).mean())
    assert op["d_met_lo"] <= op["d_met_pp"] + 1e-12 <= op["d_met_hi"] + 2e-12
    # deterministic for a given rng seed; no reference -> no paired columns
    pd.testing.assert_frame_equal(st, al.comparison_stats(ps, reference="oracle"))
    assert "d_met_pp" not in al.comparison_stats(ps).columns
    assert "d_met_pp" not in al.comparison_stats(ps, reference="missing").columns


def test_stats_helpers_deterministic_and_degenerate():
    v = np.array([1.0, 4.0, 2.0, 8.0, 5.0])
    a = stats.bootstrap_mean_ci(v, 500, np.random.default_rng(1))
    b = stats.bootstrap_mean_ci(v, 500, np.random.default_rng(1))
    assert a == b and a[0] == pytest.approx(v.mean()) and a[1] <= a[0] <= a[2]
    c = stats.bootstrap_mean_ci(np.full(6, 0.1), 300, np.random.default_rng(2))
    assert c[1] == c[0] == c[2]                                  # zero width for constant input
    one = stats.bootstrap_mean_ci([3.0], 100, np.random.default_rng(0))
    assert one[0] == 3.0 and np.isnan(one[1]) and np.isnan(one[2])
    w = v + np.array([0.5, -0.2, 0.1, 0.3, 0.0])
    p = stats.paired_bootstrap_mean_ci(w, v, 500, np.random.default_rng(3))
    assert p == stats.paired_bootstrap_mean_ci(w, v, 500, np.random.default_rng(3))
    assert p[0] == pytest.approx((w - v).mean()) and p[1] <= p[0] <= p[2]
    z = stats.paired_bootstrap_mean_ci(v, v, 200, np.random.default_rng(4))
    assert z == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        stats.paired_bootstrap_mean_ci([1.0, 2.0], [1.0], 10, np.random.default_rng(0))


def test_compare_seeds_restores_the_shared_selector(modules):
    sel = modules.selector
    before = (sel.lam, list(sel.speed_options_kmh), list(sel.boom_height_options_m), sel.nozzle_lock)
    out = al.compare_seeds(modules, "calm_morning", [0, 1], "low", 30.0, lam=0.15,
                           policies=("open", "fast_slow", "best_fixed"), n_learn=1,
                           speed_control=True, boom_control=True, forecaster="ar1",
                           fixed_nozzle="auto")
    after = (sel.lam, list(sel.speed_options_kmh), list(sel.boom_height_options_m), sel.nozzle_lock)
    assert after == before
    # within a seed the nozzle picked for the first strategy is held for the others (checked on the
    # first seed, the only one whose full results are kept)
    first = out["first"]
    assert first["open"].meta["nozzle"]["mode"] == "auto"
    assert first["fast_slow"].meta["nozzle"]["family"] == first["open"].meta["nozzle"]["family"]
    assert len(out["per_seed"]) == 6
