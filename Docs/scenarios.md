# Scenario library (WP-12)

Source: `src/coverageai/scenarios.py`. Plot: `experiments/figures/scenarios.png` (`uv run python experiments/plot_scenarios.py`).

A scenario is one field pass of K = 40 segments (`pass.segments` in `config/spec.yaml`). Each segment carries the conditions x = (wind speed, wind direction, heading, temperature, RH, travel speed, boom height). Passes are generated from a `ScenarioSpec` and an integer seed and are reproducible. Everything is clipped to the operating ranges in `config/spec.yaml`. Wind is capped at the label limit (6.7 m/s) in every preset except `marginal_limits`, which may reach the 7 m/s end of the modeling range.

## How a pass is generated

- **Wind speed:** AR(1) series around the base wind (persistence phi, stationary standard deviation sigma), multiplied by gust bursts. A burst starts at each segment with probability `gust_prob`, lasts 1 to 3 segments and scales wind by about `gust_mult`.
- **Wind direction:** direction the air moves toward (decision D1). Linear drift per segment plus a small random-walk meander (3 degrees per segment, damped by 0.5).
- **Temperature and RH:** linear trend between start and end values plus small AR(1) temperature noise. RH responds to the temperature noise (-2.5 percentage points per degC) plus independent noise, so RH falls when temperature rises.
- **Heading:** 0 and 180 degrees alternate every lap (`segments_per_lap` = 10), a back-and-forth pattern. The crosswind component is U sin(wind direction - heading), so a steady wind flips sign between laps.
- **Machine:** travel speed and boom height have AR(1) noise (persistence 0.5) around their set values.

`random_conditions(n, seed)` gives independent Latin-hypercube samples over the full ranges, used for the training dataset and selector evaluation. API: `generate(spec, seed)`, `get(name)`, `list_presets()`, `PRESETS`, `crosswind_ms(x)`.

## Presets

| Preset | Base wind (m/s) | AR(1) phi | Sigma (m/s) | Gust prob / mult | Wind dir start (deg) / drift (deg per seg) | T start to end (C) | RH start to end (%) | Speed (km/h) | Boom (m) |
|---|---|---|---|---|---|---|---|---|---|
| calm_morning | 1.2 | 0.8 | 0.4 | 0.02 / 1.5 | 45 / 0.3 | 14 to 19 | 85 to 72 | 14 (sd 0.8) | 0.50 (sd 0.02) |
| gusty_afternoon | 4.0 | 0.6 | 1.4 | 0.12 / 1.6 | 120 / 0.5 | 27 to 31 | 45 to 38 | 16 (sd 1.0) | 0.55 (sd 0.03) |
| hot_dry | 2.5 | 0.7 | 0.7 | 0.04 / 1.4 | 90 / 0.4 | 33 to 38 | 28 to 16 | 16 (sd 0.8) | 0.60 (sd 0.03) |
| crosswind_shift | 3.0 | 0.7 | 0.6 | 0.04 / 1.4 | 80 / 3.5 | 20 to 24 | 60 to 52 | 15 (sd 0.8) | 0.50 (sd 0.02) |
| marginal_limits | 6.0 | 0.6 | 0.8 | 0.12 / 1.25 | 90 / 0.5 | 24 to 28 | 50 to 44 | 16 (sd 1.0) | 0.65 (sd 0.03) |
| inversion_evening | 0.6 | 0.8 | 0.3 | 0.01 / 1.5 | 200 / 0.2 | 22 to 15 | 55 to 80 | 14 (sd 0.8) | 0.50 (sd 0.02) |

The parameter values are plausible choices for each regime, not measured weather statistics. Replace them with site data if the team has it.

## Agronomic rationale

### calm_morning
Light, steady wind; cool and humid; slow warming.

Early-morning passes usually have the lowest and steadiest wind of the day, so horizontal displacement is small and most droplets reach the target. Cool, humid air slows evaporation, which lets even fine droplets keep their size during the fall (UNL G1773). This is the benign reference: coverage is limited by application volume and droplet spectrum rather than by weather, so it checks that the selector does not over-react when conditions are easy and can choose low-drift settings at modest volume.

### gusty_afternoon
Moderate mean wind with frequent gust bursts; warm, moderately dry.

Afternoon heating produces turbulent, intermittent wind. The mean can sit well below the label limit while individual gusts approach it, so a setting chosen from the mean wind under-predicts displacement during bursts. Because drift distance scales with crosswind and fall time, fine droplets from a high boom are the most exposed (EPA mitigation menu; Nuyttens et al. on boom height). The scenario stresses the feedback loop: card residuals jump when a gust hits and the correction must react without chasing noise.

### hot_dry
Moderate wind, high temperature, low and falling humidity.

High temperature and low relative humidity raise the wet-bulb depression, so water droplets shrink faster on the way down. UNL G1773 notes that hot, dry air speeds evaporation, increases drift potential and can reduce deposition on the target. Droplets below roughly 200 um are the most sensitive, and the smallest can evaporate completely. Coverage therefore falls even when wind is moderate, which favors coarser spectra or higher volume. This preset isolates the temperature-humidity effect on the model and selector.

### crosswind_shift
Moderate wind whose direction rotates about 140 degrees over the pass.

With a back-and-forth pattern the crosswind component U sin(wind direction minus heading) changes sign between laps, so a wind that pushes spray to one side on one lap pushes it to the other on the next. A slowly rotating wind also moves between crosswind and along-track (head or tail wind) geometry, which changes displacement without any change in wind speed. EPA guidance stresses wind direction relative to the application path. The scenario checks that models and selectors use the relative angle rather than wind speed alone.

### marginal_limits
High mean wind near the label limit with gusts that exceed it.

EPA guidance and product labels commonly restrict application at wind speeds in the 10-15 mph range (4.5-6.7 m/s), and the proposed modeling range ends at 7 m/s. This preset sits at that edge, with a mean of about 6 m/s and gust bursts that cross 6.7 m/s, to test behavior where drift risk is highest and a real operator would normally stop spraying. Feasible settings may not exist, so it exercises the selector's infeasible path and the retreat flag. It is the only preset allowed to exceed the limit.

### inversion_evening
Near-calm evening air, cooling and humidifying.

Near sunset the ground cools faster than the air above it and wind often drops toward zero, which favors a surface temperature inversion. Under an inversion, fine droplets can stay suspended in a stable layer and move off target in light, variable air even though wind speed is low; product labels commonly warn against spraying under inversions (EPA label statements). Cooling and rising humidity slow evaporation, so coverage is usually good. The challenge is that low measured wind makes a wind-based drift predictor look safe, so this is a limitation test rather than a regime the simulator models directly.

## Sources (from `Docs/CoverageAIDraftParameters.pdf`)

- EPA mitigation menu measure descriptions: https://www.epa.gov/pesticides/mitigation-menu-measure-descriptions
- EPA PRN 2001-X draft spray and dust drift label statements: https://www.epa.gov/pesticide-registration/prn-2001-x-draft-spray-and-dust-drift-label-statements-pesticide-products
- UNL Extension G1773 (temperature, humidity and evaporation): https://extensionpubs.unl.edu/publication/g1773/na/html/view
- Nuyttens et al., ASABE (boom heights 0.30, 0.50, 0.75 m): https://elibrary.asabe.org/abstract.asp?aid=28220&

Please check that each citation supports the sentence it is attached to before the report is submitted.
