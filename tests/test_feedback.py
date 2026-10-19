import pytest

from coverageai.contracts import CardMeasurement, Conditions
from coverageai.feedback import BiasFeedback, default_regime_fn
from coverageai.mocks import mock_derive
from coverageai.contracts import Setting


def meas(cov, cls):
    return CardMeasurement(cov, None, None, None, None, cls, False, 0.0)


def test_converges_to_constant_offset_within_10_segments():
    fb = BiasFeedback(alpha=0.3, clip_pp=15, warmup_cards=2, per_regime=False)
    for _ in range(10):
        fb.update("any", measured_cov=28.0, predicted_cov=20.0)   # verifier reads +8 pp
    assert fb.bias("any") == pytest.approx(8.0, abs=0.5)


def test_update_uses_raw_prediction_and_returns_residual():
    fb = BiasFeedback(alpha=0.5, clip_pp=15, warmup_cards=1, per_regime=False)
    assert fb.update("r", 30.0, 20.0) == pytest.approx(10.0)
    assert fb.bias("r") == pytest.approx(5.0)          # (1-0.5)*0 + 0.5*10
    fb.update("r", 30.0, 20.0)
    assert fb.bias("r") == pytest.approx(7.5)


def test_clipping():
    fb = BiasFeedback(alpha=1.0, clip_pp=5.0, warmup_cards=1, per_regime=False)
    fb.update("r", 50.0, 10.0)
    assert fb.bias("r") == 5.0
    fb.update("r", 0.0, 40.0)
    assert fb.bias("r") == -5.0


def test_warmup():
    fb = BiasFeedback(alpha=0.5, clip_pp=15, warmup_cards=3, per_regime=True)
    for n in range(1, 4):
        fb.update("A", 30.0, 20.0)
        assert fb.bias("A") == (0.0 if n < 3 else pytest.approx(fb.state()["regimes"]["A"]["bias"]))
    assert fb.bias("A") > 0


def test_regime_isolation():
    fb = BiasFeedback(alpha=0.3, clip_pp=15, warmup_cards=2, per_regime=True)
    for _ in range(4):
        fb.update("B", 25.0, 20.0)
    b_before = fb.bias("B")
    for _ in range(10):
        fb.update("A", 5.0, 30.0)            # gust-like disturbance only in regime A
    assert fb.bias("B") == b_before
    assert fb.bias("A") < -5


def test_global_fallback_damped():
    fb = BiasFeedback(alpha=1.0, clip_pp=15, warmup_cards=2, per_regime=True)
    assert fb.bias("never_seen") == 0.0       # nothing known yet
    fb.update("A", 28.0, 20.0)
    fb.update("A", 28.0, 20.0)
    assert fb.bias("A") == pytest.approx(8.0)
    assert fb.bias("never_seen") == pytest.approx(4.0)   # 0.5 * global


def test_reset_and_state():
    fb = BiasFeedback()
    fb.update("A", 30.0, 20.0)
    st = fb.state()
    assert st["global"]["n"] == 1 and "A" in st["regimes"]
    fb.reset()
    assert fb.state()["global"]["n"] == 0 and fb.state()["regimes"] == {}


def test_retreat_flag():
    fb = BiasFeedback()
    assert fb.retreat_flag(meas(10.0, "critical"))
    assert not fb.retreat_flag(meas(20.0, "marginal"))
    assert not fb.retreat_flag(meas(35.0, "acceptable"))


def test_default_regime_bins():
    regime = default_regime_fn()
    x = Conditions(3.0, 90.0, 0.0, 25, 50, 16, 0.5)
    fine = mock_derive(x, Setting("TP65", "02", 40.0))      # F
    coarse = mock_derive(x, Setting("DGX40", "04", 40.0))   # UC
    assert regime(x, fine).endswith("|fine") and regime(x, coarse).endswith("|coarse")
    assert regime(x, fine).startswith("u2-4")
    calm = Conditions(0.5, 90.0, 0.0, 25, 50, 16, 0.5)
    assert BiasFeedback().regime(calm, mock_derive(calm, Setting("TP65", "02", 40.0))).startswith("u0-2")
