# AI usage log

The course permits AI use. Each line records one AI-assisted work session.

| Date | WP | Model | Output |
|---|---|---|---|
| 2026-10-18 | Plan, WP-00 | Claude Opus 5.5 | PLAN.md, contracts, spec.yaml, units, mocks, smoke test |
| 2026-10-18 | WP-11 | Claude Sonnet 5.5 | selector.py (constrained selector, best_fixed_setting), experiments/eval_selector.py, tests/test_selector.py |
| 2026-10-18 | WP-12 | Claude Sonnet | Scenario library (scenarios.py: six presets, AR(1) wind with gusts, random_conditions), plot script, Docs/scenarios.md, tests |
| 2026-10-18 | WP-01 | Claude Sonnet | Nozzle catalog: data/catalog/nozzles.csv (316 rows, 5 families; Hypro rows from official Pentair PDFs), gpa_checks.csv, experiments/validate_catalog.py, catalog.py (load/derive/candidates), tests/test_catalog.py |
| 2026-10-18 | WP-06 | Claude Sonnet | feedback.py (eq. 3 bias, per-regime, warm-up, clip), loop.py (run_pass, oracle wrapper, slow loop, logs/summary), e2e rewired to loop, tests |
| 2026-10-18 | WP-05 | Claude Sonnet 5.5 | Zenodo investigation of the Simoes 2025 dataset, `vision/real_data.py` loader (download, YOLOv8-polygon rasterization, tiling, summary CLI), tests, example figure, decision D9, draft email to authors |
| 2026-10-18 | WP-02 | Claude Sonnet 5.5 | physics/ (droplets, evaporation, drift, coverage), simulator.py (SpraySimulator, begin_pass, simulate_batch), tests/test_simulator.py, experiments/sim_trends.py + figure |
| 2026-10-18 | WP-03 | Claude Sonnet | vision/render.py synthetic WSP renderer (vectorized rasterizer, scanner/phone degradations), stratified dataset CLI, tests/test_render.py, example figure |
| 2026-10-18 | WP-02b | Claude Sonnet 5.5 | Exit-velocity + linear-drag relaxation in physics/drift.py (decision D10), anchor tests, refreshed sim_trends.png |
| 2026-10-18 | WP-10 | Claude Sonnet | Dataset generator (experiments/make_dataset.py, results/dataset_v1.md), model/features.py, model/predictor.py (SprayModel LightGBM + CQR interval, ridge/RF baselines), experiments/train_eval_model.py, results/model_v1.json, parity and sensitivity figures, tests/test_model.py |
