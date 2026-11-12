"""Learned card feedback (D29): features, the model wrapper, the loop adapter and the app runner.

None of these tests needs the trained files in ``models/``: each builds a tiny model of its own.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from coverageai import app_logic as al
from coverageai import feedback_learned as fl
from coverageai.loop import run_pass


@pytest.fixture(scope="module")
def modules():
    return al.build_modules_adapter("oracle_nominal", "perfect")


@pytest.fixture(scope="module")
def fast_logs(modules):
    conds = al.make_conditions("gusty_afternoon", 20001)
    return al.run_pass_wrapper(modules, conds, "fast", 20001, "med", 30.0, 0.6, fixed_nozzle="auto").logs


class Constant:
    """Stand-in for a trained model: always predicts the same correction."""
    backend, meta = "lightgbm", {"trees": 1, "n_rows": 1}

    def __init__(self, value: float):
        self.value = value
        self.seen: list[np.ndarray] = []

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(-1, fl.N_FEATURES)
        self.seen.append(X.copy())
        return np.full(len(X), self.value)


def _tiny_rows(modules, n_seeds: int = 3):
    Xs, ys = [], []
    for seed in range(20100, 20100 + n_seeds):
        conds = al.make_conditions("hot_dry", seed, n_segments=20)
        for policy in ("open", "fast"):
            res = al.run_pass_wrapper(modules, conds, policy, seed, "high", 30.0, 0.6)
            X, y = fl.rows_from_logs(res.logs)
            Xs.append(X)
            ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


# ------------------------------------------------------------------ features

def test_rows_have_the_documented_shape_and_target(fast_logs):
    X, y = fl.rows_from_logs(fast_logs)
    assert X.shape == (len(fast_logs), fl.N_FEATURES) == (len(fast_logs), len(fl.FEATURE_NAMES))
    assert len(set(fl.FEATURE_NAMES)) == fl.N_FEATURES
    for lg, target in zip(fast_logs, y):
        assert target == pytest.approx(lg.meas.coverage_pct - lg.pred.coverage_pct)   # the e_k of eq. 3
    i = fl.FEATURE_NAMES.index
    assert X[0, i("n_cards")] == 0 and np.isnan(X[0, i("e_last")]) and np.isnan(X[0, i("d_wind")])
    assert X[5, i("n_cards")] == 5
    assert X[5, i("e_last")] == pytest.approx(y[4]) and X[5, i("e_last2")] == pytest.approx(y[3])
    assert X[5, i("e_mean")] == pytest.approx(np.mean(y[:5]))
    assert np.isfinite(X[:, : len(fl.SECTION_FEATURES)]).all()


def test_a_section_never_sees_its_own_card(fast_logs):
    """Leak check: changing the card of section k must leave the feature rows up to k as they
    were and change the rows after it."""
    k = 7
    X0, _ = fl.rows_from_logs(fast_logs)
    logs = list(fast_logs)
    lg = logs[k]
    logs[k] = dataclasses.replace(lg, meas=dataclasses.replace(lg.meas, coverage_pct=lg.meas.coverage_pct + 12.0))
    X1, y1 = fl.rows_from_logs(logs)
    assert np.array_equal(X0[: k + 1], X1[: k + 1], equal_nan=True)
    assert not np.array_equal(X0[k + 1], X1[k + 1], equal_nan=True)
    assert X1[k + 1, fl.FEATURE_NAMES.index("e_last")] == pytest.approx(y1[k])


def test_the_rule_feature_is_exactly_what_the_loop_applied(fast_logs):
    """``rule_bias`` lets the model fall back on eq. 3, so it must be the correction the fast loop
    really used for that section, not an approximation of it."""
    X, _ = fl.rows_from_logs(fast_logs)
    rule = X[:, fl.FEATURE_NAMES.index("rule_bias")]
    assert any(abs(lg.bias) > 1e-9 for lg in fast_logs)
    for lg, b in zip(fast_logs, rule):
        assert b == pytest.approx(lg.bias, abs=1e-9)


def test_deposit_ratio():
    assert fl.deposit_ratio(30.0, 30.0) == pytest.approx(1.0)
    assert fl.deposit_ratio(20.0, 30.0) < 1.0 < fl.deposit_ratio(40.0, 30.0)
    # a fixed loss of deposit gives the same ratio at every coverage level
    lose = lambda c: 100 * (1 - (1 - c / 100) ** 0.8)      # noqa: E731  (80 % of the deposit arrives)
    assert fl.deposit_ratio(lose(20.0), 20.0) == pytest.approx(0.8) == pytest.approx(fl.deposit_ratio(lose(55.0), 55.0))
    assert 0.2 <= fl.deposit_ratio(0.0, 99.9) and fl.deposit_ratio(99.9, 0.1) <= 3.0


# ------------------------------------------------------------------ model wrapper

@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_train_predict_save_load(modules, tmp_path, backend):
    if backend == "xgboost":
        pytest.importorskip("xgboost")
    X, y = _tiny_rows(modules)
    n = len(y) * 3 // 4
    # the defaults ask for many rows per leaf (they are meant for millions of rows): relax them here
    small = {"min_child_samples": 5} if backend == "lightgbm" else {"min_child_weight": 1}
    m = fl.FeedbackModel.train(X[:n], y[:n], backend=backend, device="cpu", n_estimators=40, max_depth=4,
                               X_val=X[n:], y_val=y[n:], early_stopping_rounds=10, **small)
    p = m.predict(X[n:])
    assert p.shape == (len(y) - n,) and np.isfinite(p).all()
    assert m.predict(X[0]).shape == (1,)                       # one row, as the loop asks
    assert 1 <= m.meta["trees"] <= 40 and m.meta["n_rows"] == n
    imp = m.importance()
    assert set(imp) == set(fl.FEATURE_NAMES) and sum(imp.values()) == pytest.approx(1.0)
    again = fl.FeedbackModel.load(m.save(tmp_path / f"fb_{backend}.joblib"))
    assert again.backend == backend
    assert np.allclose(again.predict(X[n:]), p, atol=1e-5)
    # a file trained on another feature list is refused, not silently mis-read
    import joblib
    blob = joblib.load(tmp_path / f"fb_{backend}.joblib")
    blob["meta"]["features"] = list(fl.FEATURE_NAMES[:-1])
    joblib.dump(blob, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="different feature list"):
        fl.FeedbackModel.load(tmp_path / "old.joblib")
    with pytest.raises(FileNotFoundError, match="train_feedback_model"):
        fl.load_feedback_model(str(tmp_path / "missing.joblib"))


# ------------------------------------------------------------------ loop adapter

def test_learned_feedback_drives_the_loop(modules):
    conds = al.make_conditions("gusty_afternoon", 20002, n_segments=10)
    model = Constant(-4.0)
    mods = copy.copy(modules)
    mods.simulator.begin_pass(20002, "med")
    logs = run_pass(conds, "fast", mods, 20002, "med", 30.0, "learned", render=False,
                    feedback=fl.LearnedFeedback(model))
    assert all(lg.bias == -4.0 for lg in logs)                 # no warm-up: the model decides from card 0
    assert all(lg.corrected_pred_cov == pytest.approx(lg.pred.coverage_pct - 4.0) for lg in logs)
    assert all(lg.extra["bias_after"] == -4.0 and al.regime_text(lg.extra["regime"]) for lg in logs)
    # the model was asked with the history as it stood: the first question has no card in it
    first = model.seen[0][0]
    assert first[fl.FEATURE_NAMES.index("n_cards")] == 0
    assert max(x[0, fl.FEATURE_NAMES.index("n_cards")] for x in model.seen) == len(conds)


def test_correction_is_clipped_and_a_broken_prediction_is_ignored(modules):
    conds = al.make_conditions("gusty_afternoon", 20002, n_segments=3)
    x = conds[0]
    derived = modules.derive_fn(x, modules.selector.candidates[0])
    big = fl.LearnedFeedback(Constant(99.0), clip_pp=15.0)
    assert big.bias(big.regime(x, derived)) == 15.0
    nan = fl.LearnedFeedback(Constant(float("nan")))
    assert nan.bias(nan.regime(x, derived)) == 0.0
    assert big.global_bias() == 0.0 and big.bias(None) == 0.0   # nothing seen yet
    key = big.regime(x, derived)
    assert big.update(key, 25.0, 30.0) == pytest.approx(-5.0)
    assert big.global_bias() == 15.0 and big.state()["n_cards"] == 1
    big.reset()
    assert big.state()["n_cards"] == 0 and big.global_bias() == 0.0


# ------------------------------------------------------------------ app runner

def _digest(res):
    return tuple((lg.u.family, lg.u.size, lg.u.pressure_psi, lg.outcome.coverage_pct, lg.bias) for lg in res.logs)


def test_runner_default_is_the_rule_and_is_left_untouched(modules):
    conds = al.make_conditions("gusty_afternoon", 20003, n_segments=12)
    run = lambda **kw: al.run_pass_wrapper(modules, conds, "fast", 20003, "med", 30.0, 0.6, **kw)   # noqa: E731
    base = run()
    assert base.meta["feedback"] == {"kind": "rule"}
    assert _digest(run(feedback_model=None)) == _digest(run(feedback_model="rule")) == _digest(base)
    learned = run(feedback_model=Constant(-6.0))
    assert learned.meta["feedback"]["kind"] == "learned" and all(lg.bias == -6.0 for lg in learned.logs)
    assert _digest(run()) == _digest(base)                     # a learned run leaves nothing behind
    assert all(learned.info_for(k) is not None for k in range(learned.n))


def test_runner_strategies_without_feedback_ignore_the_model(modules):
    conds = al.make_conditions("gusty_afternoon", 20003, n_segments=8)
    for policy in ("open", "oracle", "best_fixed"):
        a = al.run_pass_wrapper(modules, conds, policy, 20003, "med", 30.0, 0.6)
        b = al.run_pass_wrapper(modules, conds, policy, 20003, "med", 30.0, 0.6, feedback_model=Constant(-6.0))
        assert _digest(a) == _digest(b), policy
        assert b.meta["feedback"] == {"kind": "none"}


def test_learning_strategy_keeps_the_rule(modules):
    """With the slow loop the trained model's correction stacks on the learned offset and
    overshoots (adoption check, D31), so ``fast_slow`` ignores the trained model."""
    conds = al.make_conditions("gusty_afternoon", 20003, n_segments=8)
    kw = dict(n_learn=1, same_field=True)
    a = al.run_pass_wrapper(modules, conds, "fast_slow", 20003, "med", 30.0, 0.6, **kw)
    b = al.run_pass_wrapper(modules, conds, "fast_slow", 20003, "med", 30.0, 0.6, feedback_model=Constant(-3.0), **kw)
    assert b.meta["feedback"] == {"kind": "rule"} and _digest(a) == _digest(b)
    assert al.TRAINED_FEEDBACK_POLICIES == ("fast",)


def test_page_text_for_a_learned_run(modules):
    conds = al.make_conditions("gusty_afternoon", 20003, n_segments=8)
    res = al.run_pass_wrapper(modules, conds, "fast", 20003, "med", 30.0, 0.6, feedback_model=Constant(-2.0))
    basis = al.feedback_basis(res)
    assert all(b["uses_feedback"] and b["source"] == "learned" for b in basis)
    assert "no card yet" in basis[0]["text"] and [b["cards_before"] for b in basis] == list(range(res.n))
    e0 = al.loop_numbers(res.logs[0])["raw_error_pp"]
    assert f"1 earlier cards (last {e0:+.1f} pp" in basis[1]["text"]
    rule = al.feedback_rule_text(res)
    assert "trained" in rule and "running-average rule" in rule
    assert "70%" in al.feedback_rule_text() and "70%" not in rule      # the rule's own text is for rule runs
    from coverageai.field_anim import field_animation_payload
    p = field_animation_payload(res, 30.0)
    assert p["feedback"] is True and p["next_label"] == "correction for a section like this now"
    assert p["sections"][3]["basis"] == basis[3]["text"]
    for raw in ("xgboost", "lightgbm", "regime", "bias"):
        assert raw not in basis[2]["text"] and raw not in rule.lower().replace("xgboost", "").replace("lightgbm", "")


def test_feedback_choice_needs_its_file_and_its_package(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setattr(fl, "DEFAULT_PATHS", {"xgboost": tmp_path / "a.joblib", "lightgbm": tmp_path / "b.joblib"})
    assert al.feedback_model_available("rule")
    assert not al.feedback_model_available("lightgbm") and not al.feedback_model_available("nonsense")
    assert al.default_feedback_choice() == "rule"              # nothing trained on disk: the rule
    (tmp_path / "b.joblib").write_bytes(b"x")
    assert al.feedback_model_available("lightgbm") and not al.feedback_model_available("xgboost")
    assert al.default_feedback_choice() == "lightgbm"          # D31: the app prefers a trained model
    (tmp_path / "a.joblib").write_bytes(b"x")
    real = importlib.util.find_spec
    has_xgb = real("xgboost") is not None
    assert al.feedback_model_available("xgboost") is has_xgb
    assert al.default_feedback_choice() == ("xgboost" if has_xgb else "lightgbm")
    # xgboost is an optional package: without it its file is not offered and LightGBM is used
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "xgboost" else real(name, *a))
    assert not al.feedback_model_available("xgboost") and al.default_feedback_choice() == "lightgbm"


def test_library_default_is_still_the_rule(modules):
    """D31 changes the app's default only. A plain library call must keep the eq. 3 rule, whatever
    is on disk, so the reported experiments and the golden digests do not move."""
    import inspect
    assert inspect.signature(al.run_pass_wrapper).parameters["feedback_model"].default is None
    assert inspect.signature(al.compare_policies).parameters["feedback_model"].default is None
    conds = al.make_conditions("gusty_afternoon", 20003, n_segments=6)
    assert al.run_pass_wrapper(modules, conds, "fast", 20003, "med", 30.0, 0.6).meta["feedback"] == {"kind": "rule"}
