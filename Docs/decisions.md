# Decision log

Decisions taken while freezing the v1.0 spec (2026-10-18). Defaults chosen by the orchestrator; team to confirm.

| ID | Issue | Decision |
|---|---|---|
| D1 | Wind direction: Draft spec says "toward", older Parameters PDF says "from" | Use **toward** (Draft is newer). `spec.yaml: wind_direction_convention` |
| D2 | "D" means drift risk in the proposal but deposition in the Draft spec | Four named outputs: `coverage_pct`, `deposition_ul_cm2`, `drift_frac`, `displacement_mm`. Selector minimizes `drift_frac`. |
| D3 | TP65 chart has no droplet class for 65 deg tips | Use the 80 deg column as proxy, flagged in catalog notes, unless the TeeJet catalog gives TP65 classes |
| D4 | DGX40 chart uses 10 in spacing, the rest 20 in | Use flow (gpm) only; recompute GPA at 20 in |
| D5 | Spray angle not on charts | TP65 65, XR 110, DGX40 40, ULD 120, 3D 100 (verify) |
| D6 | Speed and boom height controllable in Draft, treated as x in proposal | v1 selector picks (family, size, pressure); speed control is a stretch goal |
| D7 | Proposal M1/M2 dates passed without code | Compressed schedule in PLAN.md Section 5 |
| D8 | Windows Application Control blocks uv-managed Python 3.12 | Use system Python 3.14 via uv venv |
| D9 | WP-05: the annotated real WSP cards (2 manually annotated cards, 2,754 droplets, plus 127 pre-annotated real squares) are in Zenodo v2 (record 14010262) but its files are **restricted**; only v1 (13995950: 300 synthetic cards, CC BY 4.0, YOLOv8 polygons) downloads anonymously | Develop and test the loader and the B0/B1 vision stack on the 300 synthetic cards; `load_cards(root, "real")` returns [] until real files are placed under the data root. Request the real cards via the Zenodo access button and by email (draft in `reports/email_draft_simoes.md`, not sent). Until granted, the "real <= 5 pp" criterion is reported as untested (or evaluated on own phone photos of WSP). Synthetic px/mm is inferred (width/76 mm); no dpi is published. |
| D10 | With x = U*H/V_t and a 0.5 m threshold, a 2 m/s crosswind drifts ~87% of medium-droplet volume (unrealistic; the relation ignores nozzle exit velocity) | Keep threshold 0.5 m; add nozzle exit velocity v0 = 0.6*sqrt(2P/rho) with linear-drag relaxation (tau = V_t/g), so droplets cover part of H quickly before settling at V_t. Fine droplets still drift, medium/coarse mostly do not. |
