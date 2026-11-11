"""Nozzle lock (decision D19): ConstrainedSelector.nozzle_lock, best_nozzle, and the wrapper option."""
from __future__ import annotations

import pytest

from coverageai import app_logic as al
from coverageai import loop
from coverageai.selector import ConstrainedSelector


@pytest.fixture(scope="module")
def modules():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def conds():
    return al.make_conditions("gusty_afternoon", 1, n_segments=12)


def _nozzles(logs):
    return {(lg.u.family, lg.u.size) for lg in logs}


def test_lock_restricts_selection_but_pressure_varies(modules, conds):
    sel = modules.selector
    assert sel.nozzle_lock is None
    free = loop.run_pass(list(conds), "fast", modules, 1, "med", 30.0, "free", render=False)
    lock = ("ULD", "06")            # a nozzle with several usable pressures that free choice avoids
    assert lock not in _nozzles(free)
    sel.nozzle_lock = lock
    try:
        logs = loop.run_pass(list(conds), "fast", modules, 1, "med", 30.0, "locked", render=False)
    finally:
        sel.nozzle_lock = None
    assert _nozzles(logs) == {lock}
    assert len({lg.u.pressure_psi for lg in logs}) > 1


def test_candidates_for_with_lock_and_speed_options(modules, conds):
    sel = ConstrainedSelector(candidates_fn=modules.selector._candidates_fn,
                              derive_fn=modules.derive_fn,
                              nozzle_lock=("XR", "04"), speed_options_kmh=[12.0, 14.0, 16.0, 18.0])
    assert sel.nozzle_lock == ("XR", "04")
    cands = sel.candidates_for(conds[0])
    assert cands and {(u.family, u.size) for u in cands} == {("XR", "04")}
    assert len({u.travel_speed_kmh for u in cands}) > 1
    # lock off: more nozzles are offered
    sel.nozzle_lock = None
    assert len({(u.family, u.size) for u in sel.candidates_for(conds[0])}) > 1


def test_unknown_lock_raises_value_error(modules, conds):
    sel = ConstrainedSelector(candidates_fn=modules.selector._candidates_fn,
                              derive_fn=modules.derive_fn, nozzle_lock=("ZZ", "99"))
    with pytest.raises(ValueError):
        sel.select(conds[0], modules.model, 0.0, 30.0)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_best_nozzle_ranking(modules, conds, lam):
    sel = ConstrainedSelector(candidates_fn=modules.selector._candidates_fn,
                              derive_fn=modules.derive_fn, lam=lam)
    (fam, size), stats = sel.best_nozzle(conds, modules.model, 30.0)
    rows = stats["nozzles"]
    assert len(rows) == len({(u.family, u.size) for u in sel.candidates})
    succ = [r["success"] for r in rows]
    assert succ == sorted(succ, reverse=True)
    assert (rows[0]["family"], rows[0]["size"]) == (fam, size)
    top = [r for r in rows if r["success"] >= max(succ) - 1e-12]
    assert rows[0]["success"] == max(succ)
    if len(top) > 1 and lam == 1.0:
        assert rows[0]["mean_drift"] == min(r["mean_drift"] for r in top)
    if len(top) > 1 and lam == 0.0:
        assert rows[0]["mean_gpa"] == min(r["mean_gpa"] for r in top)


def test_best_nozzle_tiebreak_differs_by_lambda(modules, conds):
    """The lambda ends really change the choice among tied nozzles (guards a no-op score)."""
    def top(lam):
        sel = ConstrainedSelector(candidates_fn=modules.selector._candidates_fn,
                                  derive_fn=modules.derive_fn, lam=lam)
        return sel.best_nozzle(conds, modules.model, 5.0)[1]["nozzles"]    # easy target: many ties
    for lam, key in ((1.0, "mean_drift"), (0.0, "mean_gpa")):
        rows = top(lam)
        tied = [r for r in rows if r["success"] == rows[0]["success"]]
        assert rows[0][key] == min(r[key] for r in tied)


def test_wrapper_auto_holds_one_nozzle_and_restores_lock(modules, conds):
    res = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.6, fixed_nozzle="auto")
    nz = res.meta["nozzle"]
    assert nz["mode"] == "auto" and nz["ranking"]
    assert _nozzles(res.logs) == {(nz["family"], nz["size"])}
    assert modules.selector.nozzle_lock is None
    assert getattr(modules.selector, "inner", modules.selector).nozzle_lock is None


def test_wrapper_fixed_xr04(modules, conds):
    res = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, fixed_nozzle="XR-04")
    assert _nozzles(res.logs) == {("XR", "04")}
    assert res.meta["nozzle"]["mode"] == "fixed"
    assert (res.meta["nozzle"]["family"], res.meta["nozzle"]["size"]) == ("XR", "04")
    res2 = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, fixed_nozzle=("XR", "04"))
    assert [lg.u for lg in res2.logs] == [lg.u for lg in res.logs]


def test_wrapper_free_unchanged(modules, conds):
    a = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0)
    b = al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, fixed_nozzle="free")
    assert [lg.u for lg in a.logs] == [lg.u for lg in b.logs]
    assert a.meta["nozzle"]["mode"] == "free"


def test_wrapper_bad_nozzle_raises_before_mutation(modules, conds):
    inner = getattr(modules.selector, "inner", modules.selector)
    lam0 = inner.lam
    with pytest.raises(ValueError):
        al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, 0.2, fixed_nozzle="NOPE-99")
    assert inner.nozzle_lock is None and inner.lam == lam0
    with pytest.raises(ValueError):
        al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, fixed_nozzle="garbage")


def test_lock_restored_when_run_raises(modules, conds, monkeypatch):
    inner = getattr(modules.selector, "inner", modules.selector)

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(al, "run_pass", boom)
    with pytest.raises(RuntimeError):
        al.run_pass_wrapper(modules, conds, "fast", 1, "med", 30.0, fixed_nozzle="XR-04")
    assert inner.nozzle_lock is None


def test_best_fixed_ignores_fixed_nozzle(modules, conds):
    a = al.run_pass_wrapper(modules, conds, "best_fixed", 1, "med", 30.0)
    b = al.run_pass_wrapper(modules, conds, "best_fixed", 1, "med", 30.0, fixed_nozzle="XR-04")
    assert b.meta["nozzle"]["mode"] == "best_fixed"
    assert [lg.u for lg in a.logs] == [lg.u for lg in b.logs]


def test_compare_policies_auto_uses_same_nozzle(modules, conds):
    out = al.compare_policies(modules, conds, 1, "med", 30.0, 0.6,
                              policies=("open", "fast", "oracle"), fixed_nozzle="auto")
    nozzles = set()
    for p, r in out.items():
        assert len(_nozzles(r.logs)) == 1, p
        nozzles |= _nozzles(r.logs)
    assert len(nozzles) == 1
