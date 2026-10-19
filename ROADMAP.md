# Model Roadmap

**Goal:** estimate **air quality hour by hour** from low-cost sensors alone. The model infers each gas's true level from the sensor readings, then combines those levels into an hourly air quality rating (AQ).

Each phase changes **one thing** from the phase before it, so every comparison isolates a single modeling choice. All phases share the same data module, discretization levels, time-based train/test split, and evaluation.

| Phase | Structure | Learning | Question it answers |
|---|---|---|---|
| 1 | Single gas: `Hour → Gas → Sᵢ ← W`, i = 1…5 | MLE | Baseline: how well do the sensors recover one gas? |
| 2 | One gas per sensor (plate over gases) | MLE | Does pairing each sensor with its own target gas help? |
| 3 | Phase 2 with latent gas levels | EM | Does using hours with missing reference readings help? |
| 4 | Phase 3 chained hour to hour (HMM) | Baum–Welch | Does temporal structure help? |
| AQ | Computed on top of Phases 2–4 | none (deterministic) | How well can hourly air quality be predicted from sensors alone? |

`W` = weather (temperature `T` and relative humidity `RH`). Gases with reference readings: g ∈ {CO, NMHC, NOₓ, NO₂} (C₆H₆ only via the cross-sensitivity extension).

---

## Phase 1: Single-gas Bayesian network (MLE)

**Structure:** `Hour → Gas`, `Gas → S₁…S₅`, `W → S₁…S₅`. All five sensors are treated as responding to one gas.
**Observed:** everything. `Gas` is that gas's reference reading, treated as ground truth.
**Learning:** maximum likelihood (normalized counts) on hours where the gas and sensors are observed.
**Inference:** P(Gas | Hour, S₁…S₅, W).
**Scope:** fit first for CO; the same model can be fit separately for any gas with reference readings.
**Figure:** `phase-1.jpg`

**Assumptions**
- Sensors are conditionally independent given the gas level and weather.
- Hours are independent of each other.
- The reference analyzer is error-free.

## Phase 2: One gas per sensor (MLE)

**Structure:** a plate over gases. Inside: `Gas_g → S_g`. Outside, shared by all gases: `Hour → every Gas_g`, `W → every S_g`.

| Gas | Reference column | Sensor |
|---|---|---|
| CO | `CO(GT)` | `S₁` (PT08.S1) |
| NMHC | `NMHC(GT)` | `S₂` (PT08.S2) |
| NOₓ | `NOx(GT)` | `S₃` (PT08.S3) |
| NO₂ | `NO2(GT)` | `S₄` (PT08.S4) |

**Learning:** MLE, counting each CPT on the hours where its own variables are observed (avoids discarding hours where an unrelated gas is missing).
**Compared to Phase 1:** CO is inferred from `S₁` alone instead of all five sensors. If Phase 1 does better, the other sensors carry real information about CO (the cross-sensitivity argument).
**Figure:** `phase-2.jpg`

**Open questions for the EDA**
- **NMHC:** if its reference is mostly missing, drop it (and `S₂` loses its gas).
- **`S₅` (ozone):** there is no O₃ reference, so `S₅` has no gas parent.
- **Benzene:** has a reference but no dedicated sensor; see the cross-sensitivity extension.

## Phase 3: Latent gas levels (EM)

**Structure:** Phase 2 with each gas split into a latent true level `Gas_g` and its reference reading: `Gas_g → Ref_g`, `Gas_g → S_g` (inside the plate).
**Observed:** `Hour`, sensors, and `W` always; `Ref_g` sometimes; `Gas_g` never.
**Learning:** EM over all hours. E-step: posterior over the latent levels given whatever is observed (missing references are marginalized). M-step: re-estimate CPTs from expected counts. Initialize P(Ref_g | Gas_g) near the identity so latent states line up with concentration levels.
**Inference:** P(Gas_g | Hour, sensors, W) with references hidden on test hours.
**Additional assumption:** missing reference values are missing at random.
**Figure:** `phase-3.jpg`

### Optional extension: cross-sensitivity edges
Add edges from a gas to sensors other than its nominal one, chosen from EDA correlations (placeholders: `Gas_CO → S₂`, `Gas_C6H6 → S₂`, `Gas_NOx → S₅`). This gives benzene sensor evidence and lets `S₅` contribute. Shown by the dashed edges in the multi-gas figure (`cross-sensitivity.jpg`).

## Phase 4: HMM over hours (Baum–Welch)

**Structure:** Phase 3 unrolled over time. Each hour is one time slice containing the gas plate and `W`. Each gas's latent level carries over to the same gas next hour: `Gas_g,t−1 → Gas_g,t`. `Hour` is dropped; the transitions replace it.
**Consequence:** with no edges between gases, this is **one independent HMM per gas**, each with a small chain (3 states at 3 levels), rather than one joint chain over every combination of gas levels.
**Learning:** Baum–Welch per gas. Missing emissions are skipped (probability 1).
**Inference:** forward–backward for P(Gas_g,t | all hours); Viterbi for the most likely level sequence.
**Hypothesis:** neighboring hours help most during reference-analyzer gaps.
**Figure:** `phase-4.jpg`

## Air quality output

**Node:** `AQ` (or `AQ_t` per hour in Phase 4), a child of the gas-level nodes, outside the gas plate. It is **deterministic**: the worst band across the gases, following how air quality indices rate an hour by its worst pollutant. It has no learned parameters and isn't observed.
**Added on top of:** Phases 2–4 (in Phase 1 it reduces to the single gas's band). Because nothing is learned, it can be computed after training without refitting.
**Query:** P(AQ_t | sensors, W), combining each gas's posterior; in Phase 4, an hour-by-hour AQ timeline.
**Evaluation:** compute the true AQ from reference readings on held-out hours with the same rule, then compare with the prediction from sensors alone.

**Requirements**
- **Standard-based bins** for every gas feeding AQ, so "fair" means the same thing for each gas. This must be decided at the start, since bins are shared by all phases.
- **Only gases with usable hourly standards feed AQ.** NO₂ has hourly bands; CO and benzene standards use 8-hour or long-term averages. Either restrict AQ to those gases or state the averaging mismatch as a limitation.

---

## Shared setup (all phases)

- **Data module:** one loader (`-200` → NaN), one discretization, one time-based split (contiguous blocks, so neighboring hours don't leak between train and test).
- **Evaluation:** on held-out hours with a reference reading, hide the reference, infer it, and report accuracy of the most probable level, held-out log-likelihood, and a confusion matrix. Report AQ accuracy the same way.
- **Configurations to sweep:** discretization levels (3 / 5 / 7); quantile vs. air-quality-standard bins; `T + RH` vs. `T + AH`; with vs. without weather edges; with vs. without cross-sensitivity edges.

---
<sub>**AI use:** Roadmap drafted with assistance from Claude (Anthropic, Claude Opus 5.5) and reviewed and edited by Kiernan.</sub>