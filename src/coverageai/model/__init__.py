"""Predictive models. Exports are lazy so that importing ``coverageai.model.features`` does not
pull in LightGBM or the simulator."""
from __future__ import annotations

_EXPORTS = {
    "SprayModel": "coverageai.model.predictor",
    "PhysicsBase": "coverageai.model.hybrid",
    "PhysicsOnlyModel": "coverageai.model.hybrid",
    "PhysicsResidualModel": "coverageai.model.hybrid",
    "load_hybrid_model": "coverageai.model.hybrid",
    "MISSPEC_PRESETS": "coverageai.model.hybrid",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'coverageai.model' has no attribute {name!r}")
