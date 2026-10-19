# CoverageVision AI: Execution Plan for Subagents

ABE 598 (Digital Agriculture), Fall 2026. Team: Emery Brewer (remote), Junjie Ao, Kinjalk Parth.
Plan drafted 2026-10-18. Orchestrator: Opus (main session). Executors: Sonnet subagents.

---

## 0. How to use this plan

- The orchestrator runs work packages (WPs) in **waves**. WPs inside a wave are independent and run in parallel, each in its own git worktree (`isolation: "worktree"`).
- Each WP below has: owner (team member accountable), inputs, outputs, acceptance criteria, and a ready-to-paste agent brief. Every agent gets **Section 3 (Contracts)** and **Section 2 (Ground truth)** pasted into its prompt; those are the only shared assumptions.
- After each wave: an Opus review pass (correctness + contract conformance), `uv run pytest` green, `uv run python -m coverageai.e2e --smoke` green, then merge to `main`.
- No agent edits `src/coverageai/contracts.py` or `config/spec.yaml` except WP-00. Contract changes go through the orchestrator and get a version bump.
- Every agent appends one line per session to `docs/ai_usage_log.md` (date, WP, model, what it produced). The course allows AI use; the report should disclose it.

---

## 1. Source documents reviewed

| File | What it holds | Status |
|---|---|---|
| `Docs/CoverageVisionAI_Project_Proposal (1).pdf` | Submitted proposal (9/21). Objectives O1–O4, success criteria (Table 1), eqs. (1)–(3), roles, milestones M1–M5, risks. | **Primary source of truth for scope and success criteria.** |
| `Docs/CoverageAIDraftParameters.pdf` | Emery's system requirements: assumptions, parameter classification, nozzle lookup, outputs C / D / δ, validation via WSP + DepositScan, physics for every input. | **Source of truth for variables, units, ranges, physics relationships.** Equations lost in text extraction; agents must read the PDF visually if they need exact symbols. |
| `Docs/CoverageAIParameters.pdf` | Earlier subset of the draft above (wind→boom height only). | Superseded by the Draft. Contains a contradicting wind-direction convention (see 2.4). |
| `Docs/ABE598ProjectWorkingDoc.docx` | Brainstorm doc: rubric for proposal, early I/O tables, parameter priority, WSP example library idea. | Background. |
| `Docs/ProjectDeliveryProposal.pdf` | Emery's demo vision: interactive closed-loop simulation app; examples (SAAS, a boom sim, a MATLAB spray-pattern model, Ansys). | Sets expectation for the **demo UI**. |
| `Docs/{TP65,XR,DGX40,ULD,3DQ}Guide.*` | Manufacturer rate charts for the five nozzle families. | Raw data for the nozzle catalog (WP-01). ULD and 3D images are low resolution. |
| `Docs/simulation1.png`, `Docs/image (1).jpeg` | Screenshots of Emery's existing sims (boom view, twin-fan coverage map, uniformity coefficient). | Reference for demo visuals; pattern/uniformity model idea. |
| `Docs/image.jpeg` | **Midterm report rubric (15%)**: technical progress and early results 25, project management/teamwork/contribution table (Member, Responsibilities, Work completed, Next milestone) 25, clarity 25, analysis quality 25. | Drives WP-40. |

---

## 2. Ground truth for all agents

### 2.1 Problem in one paragraph
Predict spray performance (coverage, deposition, drift risk, pattern displacement) from pre-spray conditions x and candidate settings u; pick the setting that meets a coverage requirement at lowest predicted drift risk and volume; measure achieved coverage from a water-sensitive-paper (WSP) card image with computer vision; feed the residual back to correct the next segment (fast loop) and refit the model (slow loop). No field access: everything runs in a physics-informed simulator plus public WSP images. The report must say what simulation-based evaluation can and cannot show.

### 2.2 Fixed system assumptions (from Emery's spec)
- Target detection exists and is out of scope.
- Boom width 120 ft (36.58 m), nozzle spacing 20 in (0.508 m), fixed boom architecture.
- Nozzle families limited to: TeeJet TP65, Hypro ULD, Hypro 3D, TeeJet XR, TeeJet DGX40. Manufacturer charts are valid for flow, droplet class, spray angle, operating range.
- One standardized WSP card per segment at a known repeatable location; card is a **local** performance sensor, not a field drift measurement.
- Inputs are time-synchronized and stay inside nozzle operating ranges.

### 2.3 Variables, units, ranges (internal units in parentheses where they differ)

| Symbol | Name | Unit | Range | Type | Class |
|---|---|---|---|---|---|
| U | wind speed | m/s | 0–7 (block application > 15 mph / 6.7 m/s) | continuous | x, measured |
| θ_w | wind direction (toward) | deg | 0–360 | circular | x, measured |
| ψ | vehicle heading | deg | 0–360 | circular | x, measured |
| T | air temperature | °C | 10–40 | continuous | x |
| RH | relative humidity | % | 10–100 | continuous | x |
| v | travel speed | km/h | 6–32 (≈4–20 mph, chart range) | continuous | x in v1 (controllable is a stretch goal) |
| H | boom height | m | 0.30–0.75 | continuous | x in v1 |
| fam | nozzle family | cat | TP65, ULD, 3D, XR, DGX40 | categorical | u |
| size | nozzle size code | cat | per chart | ordinal | u |
| P | pressure | psi | per chart: TP65 30–60, XR 15–60, DGX40 20–70, ULD 20–115, 3D 10–90 | continuous | u |
| Q | flow per nozzle | gpm | lookup | derived | |
| AR | application rate | GPA (and L/ha) | derived | derived | |
| DC | droplet class | ASABE S572.3 ordinal | VF…UC | derived | |
| α_s | spray angle | deg | per family | derived | |
| φ | relative wind angle = θ_w − ψ, wrapped to (−180, 180] | deg | | derived | |
| U_cross, U_along | U·sin φ, U·cos φ | m/s | | derived | |

Outputs: coverage C (% card area), deposition D_ep (µL/cm²), drift risk D (fraction of emitted volume displaced beyond threshold), pattern displacement δ (mm on card), plus card metrics (stains/cm², Dv0.1/0.5/0.9).

Coverage classes (configurable): acceptable ≥ 30%, marginal 15–30%, critical < 15%.

### 2.4 Known discrepancies the orchestrator must resolve (ask the team; defaults below)
1. **Wind direction convention.** Draft says "toward" (90° = wind toward east); the older Parameters PDF says "from". **Default: toward**, per the Draft (newer).
2. **"D" naming.** Proposal uses D̂ = drift risk; Emery's Draft uses D = deposition and δ = displacement. **Default:** keep four outputs with distinct names: `coverage_pct`, `deposition_ul_cm2`, `drift_frac`, `displacement_mm`. The selector minimizes `drift_frac`.
3. **TP65 droplet class.** The TP chart only lists drop size for 80° and 110° tips; TP65 rows have none. **Default:** use the 80° column as proxy and flag it in the catalog `notes` column; try to source a TP65 class from the TeeJet catalog (cat51a) during WP-01.
4. **DGX40 chart spacing is 10 in**, others 20 in. Flow (gpm) is spacing-independent, so recompute GPA at 20 in. Do not copy GPA columns.
5. **Spray angle per family** is not in the charts. Defaults to verify in WP-01: TP65 65°, XR 110° (offer 80°), DGX40 40°, ULD 120°, 3D 100°.
6. **Travel speed and boom height** are "controllable" in the Draft but treated as x (uncontrolled) in the proposal's selector. **Default v1:** selector chooses (family, size, P) only; speed is a stretch-goal control.
7. **Schedule.** Proposal M1 (Oct 2) and M2 (Oct 16) have passed with no code in this folder. Revised dates in Section 5.

### 2.5 Physics the simulator must encode (first-order; documented as such)
- Flow: Q(P) = Q_ref · sqrt(P / P_ref), anchored on the chart row nearest P; interpolate between chart rows otherwise.
- Application rate: GPA = 5940 · GPM / (MPH · W_in), W_in = 20. Also L/ha = 600 · q_Lmin / (w_m · v_kmh) per nozzle.
- Droplet spectrum: log-normal in volume, parameterized by VMD from droplet class (approximate class centers: VF 90, F 145, M 215, C 310, VC 410, XC 525, UC 700 µm; configurable) with small within-class pressure trend d ∝ P^−0.3 clipped to class bounds; relative span configurable (default 1.2).
- Terminal velocity V_t(d): iterative drag correlation (Schiller–Naumann), checked against published values (≈0.012 m/s at 20 µm, ≈0.25–0.28 m/s at 100 µm, ≈0.7 m/s at 200 µm, ≈1.6–2.0 m/s at 400 µm; tolerance ±25%).
- Evaporation: d² law, d² = d0² − λ t, λ ≈ 84.76 · ΔT_wb µm²/s (Williamson–Threadgill, no ventilation term in v1); ΔT_wb from T and RH via a psychrometric approximation (e.g. Stull 2011 wet-bulb). Droplets reaching d ≤ 0 are lost.
- Fall time t = H / V_t; displacement x = U_cross · H / V_t (Emery's first-order relation). Drift fraction = volume fraction with |x| > x_crit (default 0.5 m, configurable) plus fully evaporated volume.
- Deposition on target: D_ep = AR_µL/cm² · (1 − drift_frac) · η_canopy (hidden), with 1 GPA = 0.09354 µL/cm².
- Coverage (analytic, used by simulator): stain diameter s = SF(d) · d with spread factor SF ≈ 1.6–2.2 (sample above datasheet values per proposal risk table); expected coverage C = 1 − exp(−N·E[π s²/4] / A) (Poisson/Boolean model). The renderer's explicit geometry must agree with this within 2 pp on average (cross-check test).
- Displacement δ on card: volume-weighted mean along-track/cross-track shift of deposited droplets, mm.
- Hidden mismatch (never visible to the model): wind gusts (AR(1) multiplicative noise on U, σ configurable), canopy interception η ∈ [0.6, 1.0], nozzle wear (+0–15% flow, −0–10% VMD). Exposed as a single `mismatch_level` ∈ {0, low, med, high} plus per-factor overrides.

These are fit to published trends; the report must not call this a validated drift model.

### 2.6 Success criteria (proposal Table 1, verbatim targets)
| Component | Target |
|---|---|
| End-to-end | Simulated pass runs prediction→correction with no manual steps; every decision logged and visualized. |
| Simulator | Encoded trends hold across ranges: coverage ↑ with volume; drift ↑ with U, H, fine fraction. |
| Predictive model | Coverage MAE ≤ 3 pp on held-out conditions and below linear baseline; grouped-regime error reported; sensitivity sweeps pass. |
| Selector | No mismatch: meets C_req in ≥ 90% of scenarios at lower drift than best fixed setting with same success rate. |
| Verification | Coverage MAE ≤ 2 pp synthetic, ≤ 5 pp real; 3-class agreement ≥ 90% synthetic; ≤ 2 s/card on laptop CPU. |
| Closed loop | Under mismatch, recovers ≥ 50% of open-loop→oracle gap in requirement-met segments, ≤ 10% more volume than oracle; 20 seeds, 95% CI. |

Every criterion gets reported with its measured value, met or missed.

---

## 3. Contracts (frozen in WP-00; paste into every agent brief)

Repo layout:
```
coverageai/
  pyproject.toml            # uv; Python 3.12 pinned (3.14 too new for torch wheels)
  config/spec.yaml          # ranges, thresholds, physics constants, defaults from 2.3–2.5
  data/catalog/nozzles.csv  # WP-01
  data/external/            # Zenodo WSP data (gitignored, fetched by script)
  src/coverageai/
    contracts.py            # dataclasses below + JSON (de)serialization + schema version
    units.py                # mph<->km/h, psi, gpm<->L/min, GPA<->L/ha<->µL/cm²
    catalog.py              # WP-01 lookup API
    physics/                # WP-02 droplet, evaporation, drift, coverage
    simulator.py            # WP-02 simulate_segment(x, u, rng, mismatch) -> Outcome
    scenarios.py            # WP-12 field-pass generators
    model/                  # WP-10 predictive model
    selector.py             # WP-11
    vision/render.py        # WP-03 synthetic WSP renderer
    vision/classical.py     # WP-04 B0
    vision/unet.py          # WP-13 B1
    vision/real_data.py     # WP-05 Zenodo loader
    feedback.py             # WP-06
    loop.py                 # WP-06 harness: runs a pass with pluggable modules
    e2e.py                  # smoke test entry point
    mocks.py                # WP-00 mock model / selector / verifier / simulator
  experiments/              # WP-20/21 scripts, write results/*.json + figures
  app/                      # WP-22 Streamlit demo
  reports/                  # WP-40/41 midterm + final
  tests/
  docs/ai_usage_log.md, docs/decisions.md
```

Core records (`contracts.py`, `SCHEMA_VERSION = "1.0"`):
```python
@dataclass(frozen=True)
class Conditions:            # x
    wind_speed_ms: float; wind_dir_deg: float; heading_deg: float
    temp_c: float; rh_pct: float; travel_speed_kmh: float; boom_height_m: float

@dataclass(frozen=True)
class Setting:               # u
    family: str; size: str; pressure_psi: float

@dataclass(frozen=True)
class Derived:               # catalog + geometry lookups
    flow_gpm: float; app_rate_gpa: float; app_rate_lha: float
    droplet_class: str; vmd_um: float; spray_angle_deg: float
    rel_wind_deg: float; u_cross_ms: float; u_along_ms: float

@dataclass(frozen=True)
class Outcome:               # simulator ground truth for one segment
    coverage_pct: float; deposition_ul_cm2: float; drift_frac: float
    displacement_mm: float; stains_per_cm2: float
    droplets: "DropletSample | None"   # diameters_um, x_mm, y_mm on card, for renderer

@dataclass(frozen=True)
class Prediction:
    coverage_pct: float; coverage_lo: float; coverage_hi: float
    deposition_ul_cm2: float; drift_frac: float; displacement_mm: float

@dataclass(frozen=True)
class CardMeasurement:       # vision output
    coverage_pct: float; stains_per_cm2: float | None
    dv01_um: float | None; dv05_um: float | None; dv09_um: float | None
    coverage_class: Literal["acceptable","marginal","critical"]
    saturated: bool; runtime_s: float

@dataclass
class SegmentLog:            # one row per segment, written to parquet/JSONL
    pass_id: str; seed: int; k: int; policy: str
    x: Conditions; u: Setting; derived: Derived; pred: Prediction
    corrected_pred_cov: float; outcome: Outcome; meas: CardMeasurement
    residual: float; bias: float; flagged_retreat: bool; selector_infeasible: bool
```

Module interfaces (Protocols):
```python
class Model(Protocol):     def predict(self, x: Conditions, u: Setting) -> Prediction
                           def predict_batch(self, xs, us) -> list[Prediction]
                           def fit(self, records: Iterable[tuple[Conditions, Setting, Outcome]]) -> None
class Selector(Protocol):  def select(self, x, model, bias: float, c_req: float) -> tuple[Setting, dict]  # dict: infeasible, candidates scored
class Verifier(Protocol):  def measure(self, image: np.ndarray, px_per_mm: float) -> CardMeasurement
class Simulator(Protocol): def simulate(self, x, u, rng, mismatch) -> Outcome
class Renderer(Protocol):  def render(self, outcome: Outcome, rng, degrade: bool = True) -> tuple[np.ndarray, np.ndarray]  # image RGB uint8, gt mask bool
```

Card geometry default: 76 × 26 mm WSP, rendered at 600 dpi (23.62 px/mm). Image arrays RGB uint8; WSP background yellow, stains blue.

---

## 4. Work packages

Notation: **Owner** = accountable team member (agents build, the owner reviews and presents). **Deps** = WPs that must merge first. All agents: Sonnet, worktree isolation, must add tests, must update `docs/ai_usage_log.md`.

### Wave 0 (sequential, gate for everything)

**WP-00 Scaffold and contracts** (Owner: Kinjalk, sign-off from Emery on spec.yaml)
- Create repo per Section 3 with uv (`uv init`, Python 3.12, deps: numpy, scipy, pandas, pyarrow, scikit-learn, lightgbm, matplotlib, pyyaml, opencv-python-headless, scikit-image, pytest, streamlit; torch + segmentation-models-pytorch as optional extra `[vision]`).
- `contracts.py`, `units.py`, `config/spec.yaml` (all numbers in 2.3–2.5), `mocks.py` (mock simulator returning a smooth closed-form coverage, mock model, greedy mock selector, mock verifier that reads `Outcome.coverage_pct` + noise), `e2e.py --smoke` running 1 pass × 10 segments with mocks.
- `docs/decisions.md` seeded with the seven discrepancies in 2.4 and the chosen defaults.
- `git init`, GitHub repo creation **only after user confirms** (outward-facing).
- Accept: `uv run pytest` passes; smoke run writes a SegmentLog JSONL; units round-trip tests (e.g. 15 GPA = 140.3 L/ha = 1.403 µL/cm²).

### Wave 1 (parallel; deps: WP-00)

**WP-01 Nozzle catalog** (Owner: Emery)
- Transcribe the five charts into `data/catalog/nozzles.csv`: `family,size,pressure_psi,flow_gpm,droplet_class,spray_angle_deg,source,chart_spacing_in,notes`.
- TP65 and XR and DGX40 images are legible. ULD and 3D images are low-res: fetch the original Hypro PDF charts (WebFetch/WebSearch) and transcribe from them; any cell still uncertain goes in `notes` as `UNCERTAIN`.
- Verify each row: flow follows sqrt-pressure within ±5% of neighbors; GPA recomputed at 20 in matches the chart (20-in charts) within rounding. Write a validation script that prints violations.
- `catalog.py`: `lookup(setting) -> Derived-part`, `candidates() -> list[Setting]` on a pressure grid (chart rows plus midpoints) inside each family's range.
- Accept: every chart row present; validator clean or every violation explained in notes; ≥ 200 candidate settings.

**WP-02 Simulator physics core** (Owner: Junjie, physics review by Emery)
- Implement Section 2.5 in `physics/` and `simulator.py`. Pure NumPy, vectorized; one segment ≤ 50 ms without droplet sampling, ≤ 300 ms with sampling for the renderer.
- Uses `catalog.py` through the contract only; until WP-01 merges, use a stub table with 3 rows per family.
- Tests: terminal-velocity checkpoints; d² evaporation monotone in T and inverse in RH; monotonic trends (coverage ↑ with GPA, drift ↑ with U, H, fine fraction; drift ↓ with coarser class) via one-at-a-time sweeps; Poisson coverage formula vs Monte Carlo placement within 2 pp; mismatch_level 0 is deterministic given seed.
- Output a trend figure `experiments/figures/sim_trends.png` for the midterm report.

**WP-03 Synthetic WSP renderer** (Owner: Kinjalk)
- `vision/render.py`: from `Outcome.droplets` produce card image + exact GT mask. Stains as filled ellipses (slight eccentricity), size s = SF·d, colors sampled from a yellow-to-blue palette with edge darkening; optional degradations: illumination gradient, Gaussian blur, sensor noise, JPEG (q 60–95), small perspective warp, scanner vs phone profiles.
- Dataset generator: N cards stratified across coverage bands 0–80% and droplet classes; save PNG + mask + JSON (true coverage, count, Dv stats) under `data/synthetic/v1/` with train/val/test splits by seed.
- Tests: mask coverage equals analytic within 2 pp averaged over 200 cards; deterministic per seed; 1 card ≤ 1 s.

**WP-04 Classical verifier B0** (Owner: Kinjalk)
- `vision/classical.py`: card ROI detect (largest yellow quadrilateral, perspective-correct), convert to Lab/HSV, Otsu (or fixed) threshold on stain channel, morphology cleanup, coverage = stained px / card px; stain count and size via distance-transform watershed; Dv0.1/0.5/0.9 from stain diameters / spread factor; saturated flag when coverage > 60% or merged-blob fraction high; class from thresholds.
- Evaluate on mock renderer output now, on WP-03 data once merged. Report coverage MAE, IoU, count error, class agreement, runtime, stratified by coverage band.
- Accept: ≤ 2 s/card CPU; metrics script writes `results/b0_synthetic.json`.

**WP-05 Real WSP data loader** (Owner: Kinjalk)
- Script to download Simões et al. Zenodo record 10.5281/zenodo.13995950 into `data/external/` (check license CC BY 4.0, record what is actually included: 300 synthetic cards with polygons; whether the 2 real annotated cards are present).
- `vision/real_data.py`: polygons → masks, tiling for evaluation, px/mm from metadata or card size.
- If real annotations are absent: write `docs/decisions.md` entry and a draft email to the authors (do **not** send; user sends).
- Accept: loader returns (image, mask, px_per_mm) for every available card; summary table of dataset contents.

**WP-06 Feedback module and loop harness** (Owner: Kinjalk)
- `feedback.py`: eq. (3) bias update b_{k+1} = (1−α) b_k + α e_k; clipping (|b| ≤ b_max), warm-up (min cards before acting), per-regime biases (regimes = bins of U_cross × droplet class, configurable), critical-card re-treatment flag; slow loop = append (x, u, measured outcome) to a training buffer and call `model.fit` between passes.
- `loop.py`: `run_pass(scenario, policy, modules, seed) -> list[SegmentLog]`, policies: `open`, `fast`, `fast_slow`, `oracle` (selector queries true simulator with mismatch), `best_fixed`. Writes JSONL/parquet.
- Develop entirely against `mocks.py`. Unit tests: bias converges to a constant injected offset; clipping and warm-up behave; regime isolation (a gust in one regime does not move another's bias).
- Accept: `e2e --smoke` runs all five policies with mocks.

### Wave 2 (parallel; deps: Wave 1 merged)

**WP-10 Dataset and predictive model** (Owner: Junjie)
- Generate dataset v1 from the simulator with **mismatch_level 0** (the model learns nominal physics), Latin-hypercube over x × catalog u, ~50k rows; save parquet + datasheet (ranges, seeds, version).
- Features: raw x, Derived, plus physics features (U_cross·H/V_t(VMD), GPA, ΔT_wb). Models: linear baseline (ridge on standardized features), random forest, LightGBM; quantile LightGBM (q 0.1/0.9) for the interval. NN only if trees miss the target.
- Splits: random hold-out; grouped split withholding whole regimes (e.g. U > 5 m/s, or one nozzle family); report MAE/RMSE per output and per regime.
- Sensitivity sweeps: one-at-a-time, assert sign of ∂C/∂GPA > 0, ∂drift/∂U > 0, ∂drift/∂H > 0.
- Implements `Model` protocol; `predict_batch` over ~300 candidates ≤ 20 ms.
- Accept: coverage MAE ≤ 3 pp random split and < ridge; results in `results/model_v1.json`, parity and sensitivity figures.

**WP-11 Parameter selector** (Owner: Emery)
- Eq. (2): minimize λ·norm(drift) + (1−λ)·norm(volume) s.t. Ĉ + bias − m ≥ C_req, over `catalog.candidates()`; m from prediction interval (Ĉ − Ĉ_lo) or a fixed margin; infeasible ⇒ return max-coverage setting and flag.
- Baseline `best_fixed`: single setting with the highest success rate over a scenario set; ties broken by lowest drift.
- Tests: feasibility, monotone response to C_req and λ, runtime ≤ 50 ms per call.
- Accept: on 200 no-mismatch scenarios with the oracle model (true simulator), meets C_req ≥ 90%; then with WP-10 model.

**WP-12 Scenario library** (Owner: Emery)
- `scenarios.py`: field passes of K segments (default 40) with time-varying wind (AR(1) speed, slow direction drift, occasional gust bursts), diurnal T/RH trend, heading reversals per pass (back-and-forth), boom-height noise. Named presets: `calm_morning`, `gusty_afternoon`, `hot_dry`, `crosswind_shift`, `marginal_limits`.
- Each preset documented with an agronomic rationale paragraph (Emery's domain-validation content for the report).
- Accept: seeds reproducible; presets stay inside spec ranges; plot of each preset.

**WP-13 Learned verifier B1 (U-Net)** (Owner: Kinjalk)
- `vision/unet.py`: segmentation-models-pytorch U-Net, lightweight encoder (mobilenet_v2 or resnet18, ImageNet weights), 256–384 px tiles, BCE + Dice, heavy photometric/geometric augmentation. Train on WP-03 synthetic v1; validate on held-out synthetic; test on Simões synthetic (out-of-generator) and real tiles.
- **No local GPU.** Provide (a) a CPU path with a small model and few epochs that fits in ≤ 2 h, and (b) a Colab notebook `notebooks/train_unet_colab.ipynb` for full training. Export weights to `models/unet_v1.pt` (gitignored; publish via release or HF Hub only after user confirms).
- Coverage from mask; counts via watershed on the mask (secondary); optional Cellpose zero-shot comparison for counts if time permits.
- Accept: synthetic coverage MAE ≤ 2 pp, class agreement ≥ 90%, CPU inference ≤ 2 s/card; real-card error reported with sample size, framed as a sanity check.

### Wave 3 (deps: Wave 2 merged)

**WP-20 Integration** (Owner: Kinjalk, all review)
- Wire real modules into `loop.py`: simulator → renderer → verifier (B0 and B1 selectable) → feedback; model + selector. Replace mocks in e2e test (keep mocks for fast unit tests).
- Accept: full pass of 40 segments with renderer + B1 finishes in ≤ 2 min on CPU; logs complete.

**WP-21 Closed-loop experiments** (Owner: Kinjalk)
- Grid: policies {open, fast, fast_slow, oracle, best_fixed} × mismatch {0, low, med, high} × presets × 20 seeds. Verifier: B1 (and B0 as ablation). Slow loop: 3 consecutive passes.
- Metrics: fraction of segments meeting C_req, mean coverage shortfall, applied volume, mean drift_frac, re-treatment flags; gap recovery = (closed − open) / (oracle − open); volume overhead vs oracle. Mean ± 95% CI (t-dist over seeds).
- Ablations: α, warm-up, per-regime vs global bias, verifier B0 vs B1 vs perfect measurement (isolates vision error contribution).
- Accept: `results/closed_loop.json`, figures (requirement-met vs mismatch per policy; residual variance across slow-loop passes); one-paragraph honest limitation statement.

**WP-22 Demo app** (Owner: all; UI build by Kinjalk)
- Streamlit app: choose preset, mismatch, policy; step through segments; show conditions, chosen setting and why (top-5 candidates table), predicted vs measured coverage, rendered card with predicted mask overlay, bias trace, map of segments colored by class, re-treatment flags. Visual style informed by Emery's sims (coverage map, cross-section pattern).
- Accept: runs locally with `uv run streamlit run app/main.py`; no manual steps for a full pass; screenshot set for the report.

### Wave 4: Reports (deps as noted)

**WP-40 Midterm report** (Owner: all; drafted by agent, edited by team). Can start as soon as Wave 1 lands.
- Rubric (Docs/image.jpeg): technical progress and early results (simulator trends, catalog, B0 metrics, feedback unit results), project management with the **contribution table** (Member | Responsibilities | Work completed | Next milestone), clarity, analysis quality.
- Include: revised schedule with reason for slip, decisions log summary (2.4), risks update, AI-use disclosure.
- Format: match the proposal's LaTeX/PDF style (find the proposal source if available; otherwise produce LaTeX + PDF via a local or Overleaf build). Agent writes the draft; humans own the claims about their own work.

**WP-41 Final report and presentation** (Owner: all)
- Report sections mirror the proposal; every Table 1 criterion with measured value, met/missed, cause. Domain-validation section by Emery. Slides: one section per member presenting own module.

---

## 5. Revised schedule (today is Sun 2026-10-18)

**Hard deadline: submission due Wed 2026-10-21.** User decision (10-18): agents build all modules, not only Kinjalk's. Schedule compressed accordingly:

| Date | Gate | Contents |
|---|---|---|
| Sun Oct 18 | R0 | WP-00 merged (done). Wave 1 launched: WP-01–06 plus WP-11, WP-12 (both run against mocks) |
| Mon Oct 19 | R1 | Wave 1 merged; Wave 2 (WP-10 model, WP-13 U-Net) launched |
| Tue Oct 20 | R2 | Wave 2 merged; WP-20 integration, WP-21 experiments, WP-22 demo |
| Wed Oct 21 | Submit | WP-40 report with results, contribution table, AI-use disclosure |
| Nov 20 / Dec 9 | M4 / M5 | Hardening, real-card evaluation, final report and presentation |

Python: Windows Application Control blocks uv-managed interpreters, so the venv uses system Python 3.14 (decision D8). Always run through `uv run`.

Critical path: WP-00 → WP-02 → (WP-03, WP-10) → WP-20 → WP-21. Mocks let WP-06 and WP-04 proceed before WP-02 lands.

---

## 6. Orchestration recipe

1. Opus resolves 2.4 with the user/team, then runs WP-00 itself or via one Sonnet agent, and reviews.
2. Wave 1: launch WP-01…WP-06 as six Sonnet agents in one message, `isolation: "worktree"`, each brief = Section 2 + Section 3 + its WP text + "do not modify contracts.py/spec.yaml; if a contract change is needed, stop and report it".
3. Merge order: WP-01, WP-02, then others (rebase on main). Opus review per PR (`/code-review` or cavecrew-reviewer), fix, merge.
4. Wave 2: WP-10…WP-13 in parallel. WP-13 GPU training step is handed to the user (Colab) if CPU path misses the target.
5. Wave 3: WP-20 alone, then WP-21 and WP-22 in parallel.
6. After each wave: update `reports/progress.md` (feeds the contribution table) and post a short written Teams update (proposal promised this; user posts it).

Agent brief template:
```
You are implementing WP-XX of the CoverageVision AI project (ABE 598). Repo: <path>.
Read PLAN.md Sections 2 and 3 first; they are binding. Your task: <WP text>.
Rules: Python 3.12 via `uv run`; follow contracts exactly; do not edit contracts.py or
config/spec.yaml; add pytest tests; keep functions vectorized where noted; no network
uploads or git pushes; append one line to docs/ai_usage_log.md. Finish with: files
changed, tests run + result, acceptance criteria met/missed with numbers, open questions.
```

---

## 7. Risks specific to execution

| Risk | Mitigation |
|---|---|
| Teammates own WPs that agents implement | Owner reviews and can rewrite; agents leave clear docstrings; contribution table states who built and who reviewed. Confirm with the team which WPs they want to run themselves. |
| Chart transcription errors (ULD, 3D low-res) | Source original PDFs; sqrt-pressure and GPA validators; flag UNCERTAIN cells. |
| Simulator coverage scale unrealistic | Sanity anchor: ~15 GPA with C/M droplets should give ~15–35% card coverage; tune spread factor and class VMDs in spec.yaml, document. |
| No GPU | Small U-Net on CPU + Colab notebook; B0 remains fallback verifier. |
| Real WSP annotations unavailable | Report synthetic + out-of-generator only, state gap (proposal already commits to this). |
| Circular evaluation | Hidden mismatch, oracle comparison, perfect-measurement ablation, explicit limitation text. |
| Python 3.14 default on machine | Pin 3.12 in `.python-version` via uv. |
