# Decisions

One line per decision: date — decision — why. Newest at the bottom.

- 2026-10-12 — Dev container + uv for the environment — identical Python/dependencies on any host; `uv.lock` pins versions.
- 2026-10-12 — GitHub Flow: protected `main`, PRs with 1 review, squash merge — simple enough for a 6-week project, keeps `main` working and history readable.
- 2026-10-12 — Commit datasets to `data/` (when small and license allows) — no per-person download/setup; everyone has identical data.
- 2026-10-14 — Project: UCI Air Quality (Bayes Net + EM / HMM, true vs. sensed pollution) — won round-2 ranked-choice vote outright (4/5 first-choice, beat every option head-to-head). Backup: Adaptive Tutoring (ASSISTments).
- 2026-10-19 — Ultimate output is hourly air quality (AQ) — the practical goal; gas-level estimation is the means to it.
- 2026-10-19 — Roadmap: P1 single-gas BN (MLE) → P2 multi-gas + cross-sensitivity (MLE) → P3 latent gas levels (EM) → P3b AQ output → P4 HMM — each phase changes one thing, so every comparison isolates one modeling choice. Details in `ROADMAP.md`.
- 2026-10-19 — AQ is a deterministic node (worst band across gas levels), not learned — matches how air quality indices work; adds no parameters and needs no AQ labels.
- 2026-10-19 — Naming: `Ref_g` = reference reading (observed), `Gas_g` = true level (latent, P3+) — Phases 2 and 3 were blurring together under one name.
- 2026-10-19 — Cross-sensitivity edges chosen from EDA sensor–gas correlations, 1–2 per sensor max — dataset documents cross-sensitivity; capping parents keeps sensor CPTs estimable.
- 2026-10-19 — Weather node `W` = T + RH; AH tested as an alternative, not added — AH is mostly determined by T and RH, and a third parent triples every sensor CPT.
- 2026-10-19 — Time step stays at 1 hour; "bins" means discretizing values — data is already hourly averages; the HMM's natural step.
- 2026-10-19 — HMM drops the `Hour` node; transitions carry time — keeps one transition matrix per gas instead of one per hour block.
- 2026-10-19 — Time-based train/test split (contiguous blocks), shared by all phases — a random split leaks neighboring hours, especially for the HMM.
- 2026-10-19 — Milestones M0–M6 (proposal → data/baseline → multi-gas → EM+AQ → HMM → final → resubmission) — weekly targets working back from the Nov 23 deadline; HMM can start in parallel with EM.
- 2026-10-19 — Proposal figures and LaTeX on one branch (`task/report`), one PR closing #5 and #6 — mostly LaTeX/figure edits; merge after submission so `main` matches what was turned in.
- 2026-10-19 — AI-use footer on issues and docs drafted with AI help — course GenAI policy requires describing AI use in the report; footers make that easy to compile.
- 2026-11-10 — Phase 2 sensor edges from the cross-sensitivity study (#16): nominal S1←CO, S3←NOx, S4←NO2; cross S2←C6H6, S4←CO+NOx, S5←CO+NOx; NMHC dropped — kept a gas only if partial Spearman (| nominal + T + AH + hour) and held-out CPT log-lik gain (3 bins, seeds 0–2) were both clearly non-zero and consistent, by forward selection, max 2 cross edges. C6H6 is allowed for S2 only: its correlation with every other sensor vanishes once S2 is controlled for, so the C6H6 reference is a transform of S2 (edge is structural, not a measured sensitivity). S5←NO2 (prior guess) rejected: no gain once CO and NOx are in. Details in `notebooks/phase2/cross_sensitivity.ipynb`.
- 2026-11-11 — Phase 3b Event (#19): K = 3, Event-only (hour, is_weekday → Event → gases), sensor edges from #16 — chosen on val log-likelihood among configurations whose EM solution is reproducible across 5 restarts in all 3 split seeds (aligned P(gas | Event) within 0.1). K = 4 (+0.07 nats/h) and every Event + hour model (+0.04 at K = 3) fit slightly better but are not reproducible (max |ΔP| 0.74 and ≈ 0.95), so their states can't be interpreted; day of week adds +0.005 nats/h for a 3.5× larger Event table. K = 3 beats Phase 2 by +0.62 nats/h on val with fewer parameters; states = quiet / moderate / weekday rush hour. Grid via `LayeredBN` (vectorized E-step, tested equal to DiscreteBN). Details in `notebooks/phase3b/eval.ipynb`.
