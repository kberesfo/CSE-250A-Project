# Model Roadmap

Each phase changes **one thing** from the phase before it, so every comparison isolates a single modeling choice. All phases share the same data module, discretization levels, time-based train/test split, and evaluation.

| Phase | Structure | Learning | Question it answers |
|---|---|---|---|
| 1 | CO only: `Hour → CO → Sᵢ ← W`, i = 1…5 | MLE | Baseline: how well do the sensors recover CO? |
| 2 | One gas per sensor (nominal pairing) | MLE | Does modeling each sensor's own target gas help? |
| 3 | Phase 2 structure, true gas levels latent | EM | Does using hours with missing reference readings help? |
| 4 (reach) | Phase 3's latent CO chained hour to hour | Baum–Welch (EM for HMMs) | Does temporal structure help? |

`W` = weather (temperature `T` and relative humidity `RH`).

---

## Phase 1: CO-only Bayesian network (MLE)

**Structure:** `Hour → CO`, `CO → S₁…S₅`, `W → S₁…S₅`. All five sensors are treated as responding to CO.
**Observed:** everything. `CO` is the reference reading `CO(GT)`.
**Learning:** maximum likelihood (normalized counts) on hours where CO and the sensors are observed.
**Inference:** P(CO | Hour, S₁…S₅, W).
**Figure:** `phase-1.jpg`

**Assumptions**
- Sensors are conditionally independent given CO and weather.
- Hours are independent of each other.
- The reference analyzer is error-free.

## Phase 2: One gas per sensor (MLE)

**Structure:** one node per gas, each with its nominal sensor:

| Gas | Reference column | Sensor |
|---|---|---|
| CO | `CO(GT)` | `S₁` (PT08.S1) |
| NMHC | `NMHC(GT)` | `S₂` (PT08.S2) |
| NOx | `NOx(GT)` | `S₃` (PT08.S3) |
| NO2 | `NO2(GT)` | `S₄` (PT08.S4) |

Plus `Hour → every gas` and `W → every sensor`.
**Learning:** MLE, counting each CPT on the hours where its own variables are observed (avoids discarding hours where an unrelated gas is missing).
**Compared to Phase 1:** CO is now inferred from `S₁` alone instead of all five sensors. If Phase 1 does better, the other sensors carry real information about CO, which is the cross-sensitivity argument.

**Open questions for the EDA**
- **NMHC:** if its reference is mostly missing, Phase 2 may need to drop it (and `S₂` loses its gas).
- **`S₅` (ozone):** there is no O3 reference, so `S₅` has no gas parent in this phase.
- **Benzene:** has a reference (`C6H6(GT)`) but no dedicated sensor, so it can't be estimated here; see the cross-sensitivity extension.

## Phase 3: Latent gas levels (EM)

**Structure:** Phase 2 with each gas split into a latent true level `Z_g` and its reference reading: `Z_g → Ref_g`, `Z_g → nominal sensor`.
**Observed:** `Hour`, sensors, `W` always; `Ref_g` sometimes; `Z_g` never.
**Learning:** EM over all hours. E-step: posterior over the latent levels given whatever is observed (missing references are marginalized). M-step: re-estimate CPTs from expected counts. Initialize P(Ref_g | Z_g) near the identity so latent states line up with concentration levels.
**Inference:** P(Z_g | Hour, sensors, W) with references hidden on test hours.
**Additional assumption:** missing reference values are missing at random.
**Figure:** multi-gas network (solid edges only)

### Optional extension: cross-sensitivity edges
Add edges from a gas to sensors other than its nominal one, chosen from EDA correlations (placeholders: `Z_CO → S₂`, `Z_C6H6 → S₂`, `Z_NOx → S₅`). This gives benzene sensor evidence and lets `S₅` contribute. The dashed edges in the multi-gas figure show this variant.

## Phase 4 (reach): HMM over hours

**Structure:** Phase 3 reduced to CO; `Z_CO,t−1 → Z_CO,t` replaces `Hour → Z_CO`. Each hour's `Ref_CO,t` and sensors are emissions, with `W_t` as an input.
**Learning:** Baum–Welch. Missing emissions are skipped (probability 1).
**Inference:** forward–backward for P(Z_t | all hours); Viterbi for the most likely level sequence.
**Hypothesis:** neighboring hours help most during reference-analyzer gaps.
**Figure:** `phase-3.jpg` (rename to `phase-4.jpg`)

---

## Shared setup (all phases)

- **Data module:** one loader (`-200` → NaN), one discretization, one time-based split (contiguous blocks, so neighboring hours don't leak between train and test).
- **Evaluation:** on held-out hours with a reference reading, hide the reference, infer it, and report accuracy of the most probable level, held-out log-likelihood, and a confusion matrix.
- **Configurations to sweep:** discretization levels (3 / 5 / 7); quantile vs. air-quality-standard bins; `T + RH` vs. `T + AH`; with vs. without weather edges.

---
<sub>**AI use:** Roadmap drafted with assistance from Claude (Anthropic, Claude Opus 5.5) and reviewed and edited by Kiernan.</sub>