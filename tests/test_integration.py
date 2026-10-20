import json
import time

import numpy as np
import pytest

from coverageai import catalog, integration, scenarios
from coverageai.contracts import CardMeasurement, coverage_class
from coverageai.integration import (NominalPhysicsModel, NoisyVerifier, build_modules,
                                    noise_params_from_results)
from coverageai.loop import OracleModel, run_pass, run_passes_slow, segment_rng, summarize
from coverageai.simulator import SpraySimulator

C_REQ = 30.0
SEED = 3


@pytest.fixture(scope="module")
def conds():
    return scenarios.generate(scenarios.get("gusty_afternoon"), 1)


@pytest.fixture(scope="module")
def mods():
    return build_modules("oracle_nominal", "noisy")


def _models():
    out = ["oracle_nominal", "mock"]
    if integration.spray_model_available():
        out.append("spray")
    return out


@pytest.mark.parametrize("model", _models())
@pytest.mark.parametrize("verifier", ["perfect", "noisy"])
def test_build_modules_options(model, verifier, conds):
    m = build_modules(model, verifier)
    assert m.simulator and m.selector and m.model and m.derive_fn
    logs = run_pass(conds[:3], "open", m, SEED, "med", C_REQ, "t", render=False)
    assert len(logs) == 3
    assert all(np.isfinite(l.meas.coverage_pct) for l in logs)


def test_build_modules_rejects_unknown():
    with pytest.raises(ValueError):
        build_modules("nope", "perfect")
    with pytest.raises(ValueError):
        build_modules("oracle_nominal", "nope")


@pytest.mark.parametrize("verifier", ["b0", "b1"])
def test_build_modules_image_verifiers_if_available(verifier):
    if not integration.image_verifier_available(verifier):
        pytest.skip(f"{verifier} verifier not merged")
    m = build_modules("oracle_nominal", verifier)
    assert hasattr(m.verifier, "measure") and m.renderer is not None


def test_nominal_model_shapes_and_interval(conds):
    sim = SpraySimulator()
    mdl = NominalPhysicsModel(sim)
    cands = catalog.candidates()[:40]
    x = conds[0]
    preds = mdl.predict_batch([x] * len(cands), cands)
    assert len(preds) == len(cands)
    for p in preds:
        assert p.coverage_lo <= p.coverage_pct + 1e-9 <= p.coverage_hi + 2e-9
    one = mdl.predict(x, cands[5])
    assert one.coverage_pct == pytest.approx(preds[5].coverage_pct)
    mdl.fit([])  # no-op


def test_vectorized_oracle_matches_per_candidate_simulate(conds):
    sim = SpraySimulator()
    cands = catalog.candidates()[::7]
    sim.begin_pass(SEED, "med")
    for k in (0, 1, 2, 5):
        x = conds[k]
        a = sim.simulate_candidates(x, cands, segment_rng(SEED, k), "med")
        for i in (0, len(cands) // 2, len(cands) - 1):
            o = sim.simulate(x, cands[i], segment_rng(SEED, k), "med")
            assert a["coverage_pct"][i] == pytest.approx(o.coverage_pct, rel=1e-9, abs=1e-9)
            assert a["drift_frac"][i] == pytest.approx(o.drift_frac, rel=1e-9, abs=1e-12)
            assert a["deposition_ul_cm2"][i] == pytest.approx(o.deposition_ul_cm2, rel=1e-9)


@pytest.mark.parametrize("mismatch", ["none", "med", "high"])
def test_oracle_prediction_equals_realized_outcome(conds, mismatch):
    """The oracle must see exactly the hidden realization the real segment sees."""
    m = build_modules("oracle_nominal", "perfect")
    sim = m.simulator
    cands = m.selector.candidates
    sim.begin_pass(SEED, mismatch)
    oracle = OracleModel(sim, mismatch, SEED)
    rng = np.random.default_rng(0)
    for k in range(8):
        oracle.k = k
        x = conds[k]
        preds = oracle.predict_batch([x] * len(cands), cands)
        u_sel, _ = m.selector.select(x, oracle, 0.0, C_REQ)
        i = cands.index(u_sel)
        realized = sim.simulate(x, u_sel, segment_rng(SEED, k), mismatch)
        assert preds[i].coverage_pct == pytest.approx(realized.coverage_pct, abs=1e-9)
        assert preds[i].drift_frac == pytest.approx(realized.drift_frac, abs=1e-12)
        j = int(rng.integers(len(cands)))                       # and for a random candidate
        r2 = sim.simulate(x, cands[j], segment_rng(SEED, k), mismatch)
        assert preds[j].coverage_pct == pytest.approx(r2.coverage_pct, abs=1e-9)
    sim.end_pass()


def test_oracle_policy_pass_logs_consistent(conds):
    m = build_modules("oracle_nominal", "perfect")
    logs = run_pass(conds[:10], "oracle_card", m, SEED, "med", C_REQ, "o", render=False)
    sim, oracle = m.simulator, OracleModel(m.simulator, "med", SEED)
    sim.begin_pass(SEED, "med")
    for l in logs:
        oracle.k = l.k
        p = oracle.predict_batch([l.x] * 3, [l.u] * 3)[0]
        assert p.coverage_pct == pytest.approx(l.outcome.coverage_pct, abs=1e-9)
    sim.end_pass()
    # oracle never selects an infeasible-by-truth setting unless nothing is feasible
    assert summarize(logs, C_REQ)["frac_met"] >= 0.9


def test_pass_state_persists_canopy_across_segments(conds):
    """begin_pass makes the hidden canopy/wear shared by all segments of a pass."""
    sim = SpraySimulator()
    sim.begin_pass(SEED, "high")
    u = catalog.candidates()[100]
    hids = [sim._draw_segment(segment_rng(SEED, k), "high")[0] for k in range(5)]
    sim.end_pass()
    assert len({h.canopy_eta for h in hids}) == 1
    assert len({h.flow_factor for h in hids}) == 1
    assert len({h.gust_mult for h in hids}) > 1


@pytest.mark.parametrize("policy", ["open", "fast", "oracle"])
def test_full_pass_speed_no_rendering(conds, mods, policy):
    t = time.perf_counter()
    logs = run_pass(conds, policy, mods, SEED, "med", C_REQ, "speed")
    dt = time.perf_counter() - t
    assert len(logs) == 40
    assert dt < 10.0, f"{policy} pass took {dt:.1f}s (target <= 5s)"


def test_slow_loop_runs(conds, mods):
    logs = run_passes_slow(conds[:6], mods, SEED, "med", C_REQ, n_passes=2)
    assert len(logs) == 12 and {l.extra["slow_pass"] for l in logs} == {0, 1}


class _StubVerifier:
    """Blue-pixel fraction; stands in for B0 so the rendered path is exercised."""

    def measure(self, image, px_per_mm):
        blue = image[..., 2].astype(int) > image[..., 0].astype(int) + 20
        c = 100.0 * float(blue.mean())
        return CardMeasurement(c, None, None, None, None, coverage_class(c), False, 0.0)


def test_short_real_pass_with_rendered_cards(conds):
    m = build_modules("oracle_nominal", "perfect")
    ver = integration.make_image_verifier("b0") if integration.image_verifier_available("b0") \
        else _StubVerifier()
    m.verifier = ver
    logs = run_pass(conds[:3], "fast", m, SEED, "med", C_REQ, "real", render=True)
    assert len(logs) == 3
    for l in logs:
        assert l.outcome.droplets is not None          # droplets sampled for the renderer
        assert l.extra["t_render_s"] > 0
        assert 0.0 <= l.meas.coverage_pct <= 100.0


def test_e2e_run_real_writes_logs(tmp_path):
    from coverageai.e2e import run_real
    logs = run_real(model="oracle_nominal", verifier="noisy", policy="open", out=tmp_path / "r")
    assert len(logs) == 40
    assert (tmp_path / "r.jsonl").exists() and (tmp_path / "r.parquet").exists()


def test_noisy_verifier_statistics_and_determinism():
    from coverageai.contracts import Outcome
    v = NoisyVerifier(sd_pp=2.0, bias_pp=-1.0)
    o = Outcome(40.0, 0.5, 0.01, 1.0, 100.0)
    a = [v.measure_outcome(o, np.random.default_rng(i)).coverage_pct for i in range(2000)]
    assert np.mean(a) == pytest.approx(39.0, abs=0.2)
    assert np.std(a) == pytest.approx(2.0, abs=0.15)
    assert v.measure_outcome(o, np.random.default_rng(7)) == v.measure_outcome(o, np.random.default_rng(7))
    lo = v.measure_outcome(Outcome(0.0, 0, 0, 0, 0), np.random.default_rng(1))
    assert lo.coverage_pct >= 0.0


def test_noise_params_from_json(tmp_path):
    assert noise_params_from_results("b0", tmp_path) == (integration.DEFAULT_NOISE_SD_PP, 0.0, "default")
    (tmp_path / "b0_synthetic.json").write_text(
        json.dumps({"overall": {"mae_pp": 2.0, "bias_pp": -0.5}}), encoding="utf8")
    sd, bias, src = noise_params_from_results("b0", tmp_path)
    assert sd == pytest.approx(2.0 * np.sqrt(np.pi / 2)) and bias == -0.5 and src == "b0_synthetic.json"
    (tmp_path / "b1_synthetic.json").write_text(
        json.dumps({"overall": {"rmse_pp": 3.0, "bias_pp": 0.0}}), encoding="utf8")
    assert noise_params_from_results("b1", tmp_path)[0] == pytest.approx(3.0)


def test_noise_bands_parse_and_apply(tmp_path):
    (tmp_path / "b0_synthetic.json").write_text(json.dumps({
        "overall": {"mae_pp": 1.0, "bias_pp": 0.0},
        "by_coverage_band": {"0-15": {"mae_pp": 0.5, "bias_pp": 0.5},
                             "15-30": {"mae_pp": 1.0, "bias_pp": 0.0},
                             "50-80+": {"mae_pp": 3.0, "bias_pp": -3.0}}}), encoding="utf8")
    bands, src = integration.noise_bands_from_results("b0", tmp_path)
    assert [b[:2] for b in bands] == [(0.0, 15.0), (15.0, 30.0), (50.0, 1e9)]
    assert bands[0][3] == 0.5 and bands[2][3] == -3.0
    v = NoisyVerifier(bands=bands)
    assert v.error_params(5.0)[1] == 0.5 and v.error_params(70.0)[1] == -3.0
    assert v.error_params(40.0) == (bands[-1][2], bands[-1][3])   # uncovered gap -> last band
