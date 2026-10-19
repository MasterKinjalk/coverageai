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
