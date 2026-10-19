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
| 2026-10-18/19 | WP-20/21 | Claude Sonnet 5.5 | integration.py (build_modules, NominalPhysicsModel, NoisyVerifier), vectorized oracle (simulator.simulate_candidates/nominal_arrays), loop pass-state + measure_outcome hooks, e2e --real, experiments/closed_loop.py (grid, ablations, figures, summary), results/closed_loop*, tests/test_integration.py |
| 2026-10-18 | WP-04 | Claude Sonnet 5.5 | vision/classical.py ClassicalVerifier B0 (ROI quad + perspective, flatten, Otsu/fixed threshold, watershed stains, Dv), vision/simple_cards.py test-card generator, experiments/eval_classical.py (WP-03 dataset aware), tests/test_classical.py, results/b0_synthetic.json + parity plot |
| 2026-10-18 | WP-40 | Claude Sonnet 5.5 | Midterm report draft: reports/midterm/template.html, build.py (fills numbers from results/*.json, copies figures, renders PDF via headless Edge), midterm_report.html/.pdf |
| 2026-10-18 | WP-22 | Claude Sonnet | Streamlit demo (app/main.py, app/charts.py), coverageai/app_logic.py (module adapter, pass wrapper, KPIs, field strip, nozzle explorer helpers), .streamlit theme, experiments/demo_screenshots.py (headless Chrome CDP + static fallback), tests/test_app_logic.py |
| 2026-10-18 | WP-13 | Claude Sonnet | vision/unet.py (U-Net B1: pairing, augmentation, train, tiled UNetVerifier, eval CLI), vision/toy_data.py, tests/test_unet.py, notebooks/train_unet_colab.ipynb; CPU toy run only |
| 2026-10-18 | WP-40b | Claude Sonnet 5.5 | Midterm report update with all results: selector (spray vs oracle), B1 CPU-checkpoint rows and idle timing, closed-loop section parsed from closed_loop_summary.md, demo section, status table, corrected schedule and Zenodo wording, build.py changes (cropped figures, B1 loader) |
