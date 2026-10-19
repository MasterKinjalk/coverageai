"""First-order spray physics for the CoverageVision AI simulator (WP-02).

Modules: droplets (spectrum, terminal velocity), evaporation (d^2 law, wet bulb),
drift (displacement and drift partition), coverage (Boolean-model coverage and
droplet sampling). All functions are NumPy-vectorized and take constants from
config/spec.yaml. These are published-trend fits, not a validated drift model.
"""
