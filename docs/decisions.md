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