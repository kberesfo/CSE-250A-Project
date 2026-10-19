# Model Roadmap

**Goal:** estimate **air quality hour by hour** from low-cost sensors alone. The model infers each gas's true level from the sensor readings, then combines those levels into an hourly air quality rating (AQ).

Each phase builds on the one before it. All phases share the same data module, discretization levels, time-based train/test split, and evaluation.

**Naming used throughout (and in the figures)**
- `Ref_g`: the reference analyzer's reading for gas g. Observed when the analyzer is online.
- `Gas_g`: the **true** level of gas g. Latent; it only exists from Phase 3 on.
- `S₁…S₅`: sensor readings. `W`: weather (temperature `T`, relative humidity `RH`).

| Phase | Structure | Gas level is… | Learning | Question it answers |
|---|---|---|---|---|
| 1 | Single gas: `Hour → Ref → S₁…S₅ ← W` | the reference reading (observed) | MLE | Baseline: how well do the sensors recover one gas? |
| 2 | All gases; each sensor has its nominal gas plus cross-sensitivity gases | the reference reading (observed) | MLE | Does modeling several gases and cross-sensitivity help? |
| 3 | Phase 2 with a latent `Gas_g` inserted above each `Ref_g` | latent; `Ref_g` is a noisy child | EM | Does using hours with missing reference readings help? |
| 3b | Phase 3 + air quality output `AQ` | latent | none added (AQ is computed) | How well can air quality be predicted from sensors alone? |
| 4 | Phase 3b chained hour to hour (HMM) | latent, carried over each hour | Baum–Welch | Does temporal structure help? Hour-by-hour AQ. |

Gases: g ∈ {CO, C₆H₆, NOₓ, NO₂}; NMHC added if the EDA shows enough reference data.

---

## Phase 1: Single-gas Bayesian network (MLE)

**Structure:** `Hour → Ref`, `Ref → S₁…S₅`, `W → S₁…S₅`. All five sensors are treated as responding to one gas.
**Observed:** everything. The gas node is the reference reading, treated as ground truth.
**Learning:** maximum likelihood (normalized counts) on hours where the reference and sensors are observed.
**Inference:** P(Ref | Hour, S₁…S₅, W).
**Scope:** fit first for CO; the same model can be fit separately for any gas with reference readings.
**Figure:** `phase-1.jpg`

**Assumptions**
- Sensors are conditionally independent given the gas level and weather.
- Hours are independent of each other.
- The reference analyzer is error-free.

## Phase 2: All gases with cross-sensitivity (MLE)

**What changes from Phase 1:** several gases at once, and each sensor can respond to more than one gas.

**Structure:** `Hour → every Ref_g`; `W → every sensor`. Each sensor gets:
- **Nominal edge** (solid): the gas it targets.
- **Cross-sensitivity edges** (dashed): other gases it also responds to, chosen from EDA correlations.

| Sensor | Nominal gas | Cross-sensitivity (placeholders until the EDA) |
|---|---|---|
| `S₁` (PT08.S1) | CO | — |
| `S₂` (PT08.S2) | NMHC (if kept) | CO, C₆H₆ |
| `S₃` (PT08.S3) | NOₓ | — |
| `S₄` (PT08.S4) | NO₂ | — |
| `S₅` (PT08.S5) | O₃ (no reference) | NOₓ |

Cross-sensitivity edges are what give benzene sensor evidence (it has no sensor of its own) and let `S₅` contribute.

**Observed:** everything; there is **no latent node yet**.
**Learning:** MLE. Each sensor CPT is counted on the hours where that sensor and **all** of its gas parents' references are observed. Sensors with several gas parents therefore have fewer usable hours, which Phase 3 fixes.
**CPT size:** each sensor's table has (levels)^(gas parents) × (weather levels) rows, so keep cross-sensitivity edges to one or two per sensor. Use Laplace (add-one) smoothing for sparse rows.
**Figure:** `phase-2.jpg` (multi-gas figure with gray gas nodes labeled `Ref_g`, no white or dashed nodes)

**Open questions for the EDA**
- Which cross-sensitivity edges are real (sensor-vs-gas correlations).
- Whether NMHC has enough reference data to include.

## Phase 3: Latent gas levels (EM)

**What changes from Phase 2:** a latent true level `Gas_g` is inserted above each reference reading. The reference stops being "the truth" and becomes one more noisy, sometimes-missing measurement of it.

**Structure:** `Hour → every Gas_g`; `Gas_g → Ref_g`; nominal and cross-sensitivity edges now come from `Gas_g`; `W → every sensor`.
**Observed:** `Hour`, sensors, and `W` always; `Ref_g` sometimes; `Gas_g` never.
**Learning:** EM over **all** hours, including those Phase 2 skipped because a reference was missing. E-step: posterior over the gas levels given whatever is observed. M-step: re-estimate CPTs from expected counts. Initialize P(Ref_g | Gas_g) near the identity so latent states line up with concentration levels.
**Inference:** P(Gas_g | Hour, sensors, W) with references hidden on test hours. With several connected latent gases this is genuine exact inference (variable elimination); the joint over 4 gases at 3 levels is 81 states, so it stays tractable.
**Additional assumption:** missing reference values are missing at random.
**Figure:** `phase-3.jpg` (multi-gas figure: white `Gas_g`, dashed `Ref_g`, solid and dashed sensor edges)

## Phase 3b: Air quality output

**What changes from Phase 3:** a node `AQ` with an edge from every `Gas_g`.

**Node:** `AQ` is **deterministic**: the worst band across the gases, following how air quality indices rate an hour by its worst pollutant. It has no learned parameters and is never observed (white, drawn as a double circle or with a distinct outline).
**Nothing is retrained:** AQ is computed from the Phase 3 posteriors.
**Query:** P(AQ | Hour, sensors, W).
**Evaluation:** compute the true AQ from reference readings on held-out hours with the same rule, then compare with the prediction from sensors alone.

**Requirements**
- **Standard-based bins** for every gas feeding AQ, so "fair" means the same thing for each gas. Decide this at the start, since bins are shared by all phases.
- **Only gases with usable hourly standards feed AQ.** NO₂ has hourly bands; CO and benzene standards use 8-hour or long-term averages. Either restrict AQ to those gases or state the averaging mismatch as a limitation.

**Figure:** `phase-3.jpg` with `AQ` added (blue/pink AQ edges)

## Phase 4: HMM over hours (Baum–Welch)

**What changes from Phase 3b:** each gas's latent level carries over to the next hour.

**Structure:** Phase 3b unrolled over time. Each hour is a time slice containing the gas nodes, references, sensors, `W`, and `AQ_t`. Transitions: `Gas_g,t−1 → Gas_g,t`. `Hour` is dropped; the transitions replace it.
**Coupling:** because cross-sensitivity lets one sensor depend on several gases, the gas chains are **not independent**: this is a factorial HMM. Exact forward–backward over the joint gas state (81 states at 3 levels with 4 gases) is still tractable for ~9,000 hours.
**Fallback:** if the coupled model is too slow or unstable, drop the cross-sensitivity edges for Phase 4. The chains then separate into one small HMM per gas.
**Learning:** Baum–Welch. Missing emissions are skipped (probability 1).
**Inference:** forward–backward for P(Gas_g,t | all hours) and P(AQ_t | all hours); Viterbi for the most likely sequence.
**Output:** an hour-by-hour air quality timeline from the sensors alone.
**Hypothesis:** neighboring hours help most during reference-analyzer gaps.
**Figure:** `phase-4.jpg` (the figure's plate over gases shows the nominal edges; the caption notes the cross-sensitivity edges)

---

## Shared setup (all phases)

- **Data module:** one loader (`-200` → NaN), one discretization, one time-based split (contiguous blocks, so neighboring hours don't leak between train and test).
- **Evaluation:** on held-out hours with a reference reading, hide the reference, infer it, and report accuracy of the most probable level, held-out log-likelihood, and a confusion matrix. Report AQ accuracy the same way.
- **Configurations to sweep:** discretization levels (3 / 5 / 7); quantile vs. air-quality-standard bins; `T + RH` vs. `T + AH`; with vs. without weather edges; nominal-only vs. with cross-sensitivity edges.

---
<sub>**AI use:** Roadmap drafted with assistance from Claude (Anthropic, Claude Opus 5.5) and reviewed and edited by Kiernan.</sub>