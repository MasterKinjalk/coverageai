import json

import pytest

from coverageai import units
from coverageai.contracts import (Conditions, Model, Renderer, Selector, Simulator, Verifier,
                                  coverage_class)
from coverageai.e2e import smoke
from coverageai.mocks import MockModel, MockRenderer, MockSelector, MockSimulator, MockVerifier


def test_unit_roundtrips():
    assert units.gpa_to_lha(15) == pytest.approx(140.3, abs=0.1)
    assert units.gpa_to_ul_cm2(15) == pytest.approx(1.403, abs=0.001)
    assert units.mph_to_kmh(units.kmh_to_mph(16.0)) == pytest.approx(16.0)
    assert units.lmin_to_gpm(units.gpm_to_lmin(0.4)) == pytest.approx(0.4)
    # TP6504 at 40 psi, 10 mph, 20 in spacing: chart says 11.9 GPA
    assert units.gpa(0.40, 10, 20) == pytest.approx(11.9, abs=0.05)
    # ASABE SI form agrees with the US form
    q_lmin = units.gpm_to_lmin(0.40)
    assert units.lha(q_lmin, units.mph_to_kmh(10), 0.508) == pytest.approx(units.gpa_to_lha(11.88), rel=0.01)


def test_wrap():
    assert units.wrap_deg(270) == -90
    assert units.wrap_deg(-180) == 180
    assert units.wrap_deg(45) == 45


def test_coverage_class():
    assert coverage_class(30) == "acceptable"
    assert coverage_class(29.9) == "marginal"
    assert coverage_class(14.9) == "critical"


def test_mocks_satisfy_protocols():
    assert isinstance(MockModel(), Model)
    assert isinstance(MockSelector(), Selector)
    assert isinstance(MockSimulator(), Simulator)
    assert isinstance(MockRenderer(), Renderer)
    assert isinstance(MockVerifier(), Verifier)


def test_smoke_pass(tmp_path):
    out = tmp_path / "smoke.jsonl"
    logs = smoke(5, out=out)
    assert len(logs) == 5
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["schema_version"] == "1.2"
    assert set(rows[0]["x"]) == set(Conditions.__dataclass_fields__)
